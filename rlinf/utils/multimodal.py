# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Materialize immutable image references for learner microbatches."""

import hashlib
import io
import json
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=2)
def _image_processor(model_path: str):
    """Load the policy's image processor only for file-backed training rows."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True
    ).image_processor


class _ProcessedImageCache:
    """A process-local, byte-bounded cache of immutable CPU processor outputs."""

    def __init__(self):
        self.entries = OrderedDict()
        self.bytes = 0

    def trim(self, budget):
        while self.entries and self.bytes > budget:
            _, (pixels, grid) = self.entries.popitem(last=False)
            self.bytes -= pixels.nbytes + grid.nbytes

    def get(self, key):
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return self.entries[key]

    def put(self, key, value, budget):
        size = sum(tensor.nbytes for tensor in value)
        if size > budget:
            return
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.bytes -= sum(tensor.nbytes for tensor in previous)
        self.entries[key] = value
        self.bytes += size
        self.trim(budget)


_processed_image_cache = _ProcessedImageCache()


def materialize_multimodal_inputs(
    payloads: list[dict], model_path: str, device, cache_bytes: int = 0
) -> dict:
    """Build one microbatch's pixels in row order from tensors or immutable images.

    File-backed rows carry checksums and the rollout token grid. A missing,
    changed or differently processed image is rejected before model forward.
    """
    pixels, grids = [], []
    _processed_image_cache.trim(cache_bytes)
    for entry in payloads:
        if not entry:
            continue
        grid = entry["image_grid_thw"]
        if "image_refs" in entry:
            from PIL import Image

            contents = []
            for reference in entry["image_refs"]:
                content = Path(reference["path"]).read_bytes()
                if hashlib.sha256(content).hexdigest() != reference["sha256"]:
                    raise ValueError(
                        f"Training image checksum mismatch: {reference['path']}"
                    )
                contents.append(content)
            processor = _image_processor(model_path)
            # Cache the complete image sequence: generic processors can depend
            # on batch composition. Never assume per-image preprocessing commutes.
            cached = None
            if cache_bytes:
                key = (
                    model_path,
                    id(processor),
                    json.dumps(processor.to_dict(), sort_keys=True)
                    if hasattr(processor, "to_dict")
                    else None,
                    tuple(reference["sha256"] for reference in entry["image_refs"]),
                )
                cached = _processed_image_cache.get(key)
            if cached is None:
                images = []
                for content in contents:
                    with Image.open(io.BytesIO(content)) as image:
                        images.append(image.convert("RGB"))
                processed = processor(images=images, return_tensors="pt")
                cached = (
                    processed["pixel_values"].to(torch.bfloat16),
                    processed["image_grid_thw"],
                )
                if cache_bytes:
                    _processed_image_cache.put(key, cached, cache_bytes)
            if not torch.equal(grid, cached[1]):
                raise ValueError(
                    "Training image grid differs from the recorded rollout grid"
                )
            pixels.append(cached[0])
        else:
            pixels.append(entry["pixel_values"])
        grids.append(grid)
    if not pixels:
        return {}
    return {
        "pixel_values": torch.cat(pixels).to(device),
        "image_grid_thw": torch.cat(grids).to(device),
    }
