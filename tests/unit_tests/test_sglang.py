# Copyright 2026 The RLinf Authors.
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

"""SGLang request metadata, image processing, and service lifecycle."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from rlinf.agents.rpent.thinking_budget import ThinkingBudget, apply_thinking_budget
from rlinf.scheduler import Channel, Cluster


def test_sglang_scheduler_wrapper_keeps_the_vendor_entrypoint():
    pytest.importorskip("sglang")
    # Engine installation rewrites imported aliases. Check in a fresh process
    # so global patches neither hide this regression nor leak into other tests.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from rlinf.hybrid_engines.sglang.common import sgl_engine, sgl_scheduler; "
            "from sglang.srt.managers import scheduler; "
            "assert scheduler.run_scheduler_process is sgl_scheduler.run_scheduler_process; "
            "assert sgl_scheduler.ORIG__run_scheduler_process is not scheduler.run_scheduler_process; "
            "assert sgl_scheduler.ORIG__run_scheduler_process.__module__ == 'sglang.srt.managers.scheduler'",
        ],
        check=True,
        timeout=60,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


@pytest.fixture
def qwen_image_processor():
    """Real SGLang/HF preprocessing; model assets are optional in CPU CI."""
    model_path = os.environ.get("RLINF_TEST_QWEN_VL_MODEL")
    if not model_path:
        pytest.skip("Set RLINF_TEST_QWEN_VL_MODEL to a local Qwen3.5 processor.")
    pytest.importorskip("sglang")
    from transformers import AutoConfig, AutoProcessor

    from rlinf.hybrid_engines.sglang.multimodal.qwen_vl import TokenizedQwenVLProcessor

    hf = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    server = SimpleNamespace(
        mm_process_config={},
        skip_tokenizer_init=True,
        disable_fast_image_processor=True,
        keep_mm_feature_on_device=False,
        tokenizer_worker_num=1,
    )
    processor = TokenizedQwenVLProcessor(config, server, hf, "cpu", skip_mm_pool=True)
    try:
        yield processor, hf, config
    finally:
        processor.io_executor.shutdown()
        processor.cpu_executor.shutdown()


@pytest.mark.parametrize(
    ("detokenize", "parts"),
    [(False, ["\n", "\n"]), (True, ["\n", "\n"]), (False, ["a", "hello"])],
)
def test_qwen_image_processor_preserves_sampled_tokens(
    qwen_image_processor, detokenize, parts
):
    from PIL import Image
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    processor, hf, config = qwen_image_processor
    processor.server_args.skip_tokenizer_init = not detokenize
    images = [Image.new("RGB", (64, 64), "red"), Image.new("RGB", (96, 64), "blue")]
    separator = "".join(parts)
    text = f"A<|vision_start|><|image_pad|><|vision_end|>{separator}<|im_end|>B<|vision_start|><|image_pad|><|vision_end|>C"
    expected = hf(text=[text], images=images, return_tensors="pt")
    canonical = expected.input_ids[0].tolist()
    at = canonical.index(config.vision_end_token_id) + 1
    separator_size = len(hf.tokenizer.encode(separator, add_special_tokens=False))
    prompt = (
        canonical[:at]
        + [
            token
            for part in parts
            for token in hf.tokenizer.encode(part, add_special_tokens=False)
        ]
        + canonical[at + separator_size :]
    )
    assert hf.tokenizer.decode(prompt) == hf.tokenizer.decode(canonical)
    assert prompt != hf.tokenizer.encode(
        hf.tokenizer.decode(prompt), add_special_tokens=False
    )
    request = SimpleNamespace(
        video_data=None, audio_data=None, rid="token-preservation"
    )
    output = asyncio.run(processor.process_mm_data_async(images, prompt, request))
    assert output.input_ids == prompt
    assert torch.equal(
        torch.cat([item.feature for item in output.mm_items]), expected.pixel_values
    )
    input_ids = torch.tensor([prompt])
    for item in output.mm_items:
        for start, end in item.offsets:
            assert input_ids[0, start : end + 1].eq(config.image_token_id).all()
    # HF's independent position implementation needs only configuration, not weights.
    geometry = SimpleNamespace(config=config)
    geometry.get_vision_position_ids = MethodType(
        Qwen3_5Model.get_vision_position_ids, geometry
    )
    positions, delta = Qwen3_5Model.get_rope_index(
        geometry,
        input_ids,
        (input_ids == config.image_token_id).int(),
        image_grid_thw=expected.image_grid_thw,
    )
    assert torch.equal(output.mrope_positions, positions[:, 0])
    assert torch.equal(output.mrope_position_delta, delta)


@pytest.mark.parametrize("invalid", ["grid", "image_count"])
def test_qwen_image_processor_rejects_inconsistent_images(
    qwen_image_processor, invalid
):
    from PIL import Image

    processor, hf, config = qwen_image_processor
    images = [Image.new("RGB", (64, 64), "red")]
    prompt = (
        hf(
            text=["A<|vision_start|><|image_pad|><|vision_end|>B"],
            images=images,
            return_tensors="pt",
        )
        .input_ids[0]
        .tolist()
    )
    if invalid == "grid":
        prompt.remove(config.image_token_id)
        message = "processor grid needs"
    else:
        images.append(images[0])
        message = "image spans, but received"
    request = SimpleNamespace(video_data=None, audio_data=None, rid="invalid-image")
    with pytest.raises(ValueError, match=message):
        asyncio.run(processor.process_mm_data_async(images, prompt, request))


def test_qwen_image_processor_keeps_text_request_behavior(qwen_image_processor):
    from PIL import Image

    processor, hf, _ = qwen_image_processor
    images = [Image.new("RGB", (64, 64), "red")]
    text = "A<|vision_start|><|image_pad|><|vision_end|>\n\nB"
    expected = hf(text=[text], images=images, return_tensors="pt")
    request = SimpleNamespace(video_data=None, audio_data=None, rid="text-request")
    output = asyncio.run(processor.process_mm_data_async(images, text, request))
    assert output.input_ids == expected.input_ids[0].tolist()
    assert torch.equal(output.mm_items[0].feature, expected.pixel_values)


def test_qwen_image_processor_external_registration(qwen_image_processor, monkeypatch):
    from sglang.srt.managers import multimodal_processor

    processor, _, _ = qwen_image_processor
    monkeypatch.setattr(multimodal_processor, "PROCESSOR_MAPPING", {})
    multimodal_processor.import_processors(
        "rlinf.hybrid_engines.sglang.multimodal", overwrite=True
    )
    for model in processor.models:
        assert multimodal_processor.PROCESSOR_MAPPING[model] is type(processor)


@pytest.mark.parametrize("resume_dir", [None, "/checkpoint"])
@pytest.mark.parametrize("backup", [None, False, True])
def test_sglang_cpu_backup_loads_unsynchronized_weights(
    monkeypatch, resume_dir, backup
):
    import dataclasses

    from omegaconf import OmegaConf

    pytest.importorskip("sglang")
    from rlinf.workers.rollout.sglang import sglang_worker

    # Isolate the vendor engine boundary; exercise RLinf's real argument assembly.
    def server_args(**kwargs):
        cls = dataclasses.make_dataclass(
            "ServerArgs", [(key, object) for key in kwargs]
        )
        return cls(**kwargs)

    monkeypatch.setattr(sglang_worker, "ServerArgs", server_args)
    rollout = OmegaConf.create(
        {
            "enforce_eager": False,
            "validate_weight": False,
            "validate_weight_first_sync": True,
            "model": {
                "model_path": "/model",
                "precision": "bf16",
                "trust_remote_code": True,
            },
            "sglang": {
                "use_torch_compile": False,
                "torch_compile_max_bs": 4,
                "decode_log_interval": 100,
                "attention_backend": "triton",
                "disable_overlap_schedule": True,
            },
            "cuda_graph_max_bs": 4,
            "max_running_requests": 12,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.7,
            "detokenize": False,
        }
    )
    if backup is not None:
        rollout.sglang.enable_weights_cpu_backup = backup
    worker = SimpleNamespace(
        _cfg=SimpleNamespace(runner=SimpleNamespace(resume_dir=resume_dir)),
        _cfg_rollout=rollout,
        _rank=0,
        weight_reload="sync",
        acquire_free_port=lambda: 12345,
        log_on_first_rank=lambda message: None,
    )
    args = sglang_worker.SGLangWorker._build_server_args(worker)
    assert args["disable_overlap_schedule"] is True
    assert args["enable_weights_cpu_backup"] is bool(backup)
    assert args["load_format"] == ("auto" if backup or resume_dir is None else "dummy")


@pytest.mark.parametrize("num_nodes", [1, 2])
def test_sglang_replicas_reserve_both_initialization_ports(monkeypatch, num_nodes):
    import dataclasses
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from itertools import count

    from omegaconf import OmegaConf

    pytest.importorskip("sglang")
    from rlinf.scheduler.manager.lock_manager import PortLockManager
    from rlinf.scheduler.worker.lock import PortLock
    from rlinf.scheduler.worker.worker import Worker
    from rlinf.workers.rollout.sglang import sglang_worker

    # Repeated candidates force contention in the real node-scoped lock.
    manager = object.__new__(PortLockManager)
    manager._port_locks = {}
    manager._lock = threading.Lock()
    candidates = count()
    candidate_lock = threading.Lock()

    def find_free_port(max_port_num=None):
        with candidate_lock:
            return 35000 + next(candidates) // 4

    monkeypatch.setattr(Cluster, "find_free_port", staticmethod(find_free_port))

    def server_args(**kwargs):
        cls = dataclasses.make_dataclass(
            "ServerArgs", [(key, object) for key in kwargs]
        )
        return cls(**kwargs)

    monkeypatch.setattr(sglang_worker, "ServerArgs", server_args)

    def initialize(rank):
        worker = SimpleNamespace(
            _cfg=SimpleNamespace(runner=SimpleNamespace(resume_dir=None)),
            _cfg_rollout=OmegaConf.create(
                {
                    "enforce_eager": True,
                    "validate_weight": False,
                    "model": {
                        "model_path": "/model",
                        "precision": "bf16",
                        "trust_remote_code": False,
                    },
                    "sglang": {
                        "use_torch_compile": False,
                        "torch_compile_max_bs": 4,
                        "decode_log_interval": 100,
                        "attention_backend": "triton",
                    },
                    "cuda_graph_max_bs": 4,
                    "max_running_requests": 12,
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.7,
                    "detokenize": False,
                }
            ),
            _cluster_node_rank=rank % num_nodes,
            _worker_name=f"rollout-{rank}",
            _rank=rank,
            weight_reload="sync",
            log_on_first_rank=lambda message: None,
        )
        port_lock = object.__new__(PortLock)
        port_lock._worker = worker
        port_lock._lock_manager = manager
        worker._port_lock = port_lock
        worker.acquire_free_port = lambda: Worker.acquire_free_port(worker)
        return worker, sglang_worker.SGLangWorker._build_server_args(worker)

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(initialize, range(16)))

    assert len(manager._port_locks) == 32
    for worker, args in results:
        nccl_port = args["nccl_port"]
        init_port = int(args["dist_init_addr"].rsplit(":", 1)[1])
        assert nccl_port != init_port
        for port in (nccl_port, init_port):
            assert manager._port_locks[worker._cluster_node_rank, port] == (
                worker._worker_name
            )
    assert next(candidates) > 32

    # Exercise the installed vendor handoff without launching an engine.
    from sglang.srt import server_args as vendor

    def unexpected_free_port():
        pytest.fail("SGLang must use the port reserved by RLinf.")

    monkeypatch.setattr(vendor, "get_free_port", unexpected_free_port)
    args = results[0][1]
    vendor_args = object.__new__(vendor.ServerArgs)
    vendor_args.nccl_port = args["nccl_port"]
    vendor_args.dist_init_addr = args["dist_init_addr"]
    ports = vendor.PortArgs.init_new(vendor_args)
    try:
        assert ports.nccl_port == args["nccl_port"]
    finally:
        for value in vars(ports).values():
            if isinstance(value, str) and value.startswith("ipc://"):
                Path(value.removeprefix("ipc://")).unlink(missing_ok=True)


@pytest.mark.parametrize("cache_kind", ["legacy", "radix", "mamba", "swa"])
@pytest.mark.parametrize("num_running", [0, 3])
def test_sglang_running_state_uses_native_hybrid_pool_accounting(
    cache_kind, num_running
):
    from types import MethodType
    from unittest.mock import Mock

    pytest.importorskip("sglang")
    from sglang.srt.managers.scheduler import Scheduler as VendorScheduler

    from rlinf.hybrid_engines.sglang.common.sgl_scheduler import Scheduler

    cache = SimpleNamespace(evictable_size=lambda: 20)
    allocator = SimpleNamespace(available_size=lambda: 50, size=100)
    state = SimpleNamespace(
        max_total_num_tokens=100,
        max_running_requests=8,
        token_to_kv_pool_allocator=allocator,
        tree_cache=cache,
        running_batch=SimpleNamespace(reqs=[object()] * num_running),
        waiting_queue=[object(), object()],
    )
    expected_used = 30
    if cache_kind != "legacy":
        state.is_hybrid_ssm = cache_kind == "mamba"
        state.is_hybrid_swa = cache_kind == "swa"
        state.enable_hisparse = False
        state.get_pool_stats = MethodType(VendorScheduler.get_pool_stats, state)
        for name in ("_get_token_info", "_get_mamba_token_info", "_get_swa_token_info"):
            setattr(state, name, MethodType(getattr(VendorScheduler, name), state))
    if cache_kind in ("mamba", "swa"):
        cache.evictable_size = Mock(side_effect=NotImplementedError)
        cache.full_evictable_size = lambda: 20
    if cache_kind == "mamba":
        cache.supports_mamba = lambda: True
        cache.is_tree_cache = lambda: True
        cache.mamba_evictable_size = lambda: 2
        state.req_to_token_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(size=10, available_size=lambda: 3)
        )
    elif cache_kind == "swa":
        allocator.available_size = Mock(side_effect=NotImplementedError)
        allocator.full_available_size = lambda: 30
        allocator.swa_available_size = lambda: 15
        cache.swa_evictable_size = lambda: 5
        state.full_tokens_per_layer = 100
        state.swa_tokens_per_layer = 60
        expected_used = 50

    assert Scheduler.get_scheduler_running_state(state) == {
        "num_running_reqs": num_running,
        "max_running_reqs": 8,
        "num_used_tokens": expected_used,
        "max_total_num_tokens": 100,
        "token_usage": expected_used / 100,
        "num_queue_reqs": 2,
    }
    if cache_kind in ("mamba", "swa"):
        cache.evictable_size.assert_not_called()
    if cache_kind == "swa":
        allocator.available_size.assert_not_called()


@pytest.mark.placement
@pytest.mark.parametrize("replica_affinity", [False, True])
def test_serverless_shutdown_unblocks_the_replica_queue(replica_affinity):
    from omegaconf import OmegaConf

    pytest.importorskip("sglang")
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

    Cluster(num_nodes=1)
    incoming = Channel.create(f"shutdown-input-{replica_affinity}")
    outgoing = Channel.create(f"shutdown-output-{replica_affinity}")
    assert incoming.qsize() == outgoing.qsize() == 0
    worker = object.__new__(SGLangWorker)
    worker._rank = 3
    worker._is_model_instance_entry = True
    worker._placement = SimpleNamespace(rollout_model_instance_id=lambda rank: rank)
    worker._inflight_generates = set()
    worker._dispatch_gate = asyncio.Event()
    worker._dispatch_gate.set()
    worker._cfg_rollout = OmegaConf.create({"replica_affinity": replica_affinity})
    request_key = worker._rank if replica_affinity else "default_queue"

    async def run():
        consumer = asyncio.create_task(worker.rollout_serverless(incoming, outgoing))
        try:
            while not hasattr(worker, "_serverless_input_channel"):
                await asyncio.sleep(0)
            await asyncio.sleep(0.05)
            await worker.stop_serverless()
            await asyncio.wait_for(consumer, timeout=2)
            for _ in range(100):
                if incoming.empty(key=request_key):
                    break
                await asyncio.sleep(0.01)
            assert incoming.empty(key=request_key)
            assert incoming.empty()
        finally:
            # Release a receiver even when an incorrect stop key fails this test.
            incoming.put(None, key=request_key)
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)

    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.mark.parametrize("do_sample", [True, False])
@pytest.mark.parametrize("presence_penalty", [None, 0.0, 1.5])
def test_sglang_sampling_keeps_explicit_presence_penalty(do_sample, presence_penalty):
    from omegaconf import OmegaConf

    pytest.importorskip("sglang")
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

    params = {
        "do_sample": do_sample,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "max_new_tokens": 32768,
    }
    if presence_penalty is not None:
        params["presence_penalty"] = presence_penalty
    actual = SGLangWorker.get_sampling_param_from_config(OmegaConf.create(params))
    if presence_penalty is None:
        assert "presence_penalty" not in actual
    else:
        assert actual["presence_penalty"] == presence_penalty
    assert actual["max_new_tokens"] == 32768
    if do_sample:
        assert actual["repetition_penalty"] == 1.0


def test_sglang_packet_metadata_stays_aligned_with_cumulative_tokens():
    from rlinf.hybrid_engines.sglang.common.token_metadata import (
        accumulate_token_metadata,
        decode_generation_result,
    )

    states = {"episode": SimpleNamespace()}
    for version, count, forced in [(1, 3, False), (2, 2, False), (2, 1, True)]:
        packet = SimpleNamespace(
            rids=["episode", "cancelled"],
            customized_info={
                "rollout_weight_versions": [[version] * count, [0]],
                "forced_token_mask": [[forced] * count, [False]],
            },
        )
        accumulate_token_metadata(packet, states)
    assert packet.customized_info["rollout_weight_versions"][0] == [1, 1, 1, 2, 2, 2]
    assert packet.customized_info["forced_token_mask"][0] == [False] * 5 + [True]

    result = decode_generation_result(
        {
            "output_ids": [10, 11, 12, 13, 14, 15],
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 8,
                "forced_token_mask": packet.customized_info["forced_token_mask"][0],
                "rollout_weight_versions": packet.customized_info[
                    "rollout_weight_versions"
                ][0],
                "output_token_logprobs": [
                    (-0.5, token, None) for token in range(10, 16)
                ],
            },
        },
        return_logprobs=True,
    )
    assert result["forced_token_positions"] == [5]
    assert (
        len(result["logprobs"])
        == len(result["weight_versions"])
        == len(result["output_ids"])
    )


def test_sglang_rejects_truncated_forced_token_metadata():
    from rlinf.hybrid_engines.sglang.common.token_metadata import (
        decode_generation_result,
    )

    with pytest.raises(ValueError, match="align with output IDs"):
        decode_generation_result(
            {
                "output_ids": [1, 2],
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "prompt_tokens": 1,
                    "forced_token_mask": [True],
                },
            },
            return_logprobs=False,
        )


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_sglang_weight_install_is_independent_of_graph_capture(
    enforce_eager, monkeypatch
):
    pytest.importorskip("sglang")
    from rlinf.hybrid_engines.sglang.common.io_struct import SyncHFWeightInput
    from rlinf.hybrid_engines.sglang.common.sgl_scheduler import Scheduler
    from rlinf.scheduler import Worker

    monkeypatch.setattr(Worker.torch_platform, "synchronize", lambda: None)
    received = {"weight": torch.tensor([1.0, 2.0])}
    installed = {}
    # Transport and model-loader boundaries; the scheduler owns publication.
    scheduler = SimpleNamespace(
        cfg=SimpleNamespace(rollout=SimpleNamespace(enforce_eager=enforce_eager)),
        weight_reload="sync",
        _rlinf_worker=SimpleNamespace(recv=lambda **kwargs: received),
        _actor_group_name="actor",
        actor_weight_rank=0,
        is_weight_offloaded=False,
        batch_load_hf_weight=installed.update,
        weight_norm_dict=None,
        flush_cache=lambda: True,
        _handle_cache_after_sync=lambda mode: None,
        _rlinf_weight_version=None,
    )
    result = Scheduler.sync_hf_weight(scheduler, SyncHFWeightInput(weight_version=7))
    torch.testing.assert_close(installed["weight"], received["weight"])
    assert result.weight_version == scheduler._rlinf_weight_version == 7


@pytest.mark.parametrize("batched", [False, True])
def test_server_request_preserves_images_and_logit_processors(batched):
    import base64
    import copy

    from rlinf.utils.http_client import InferenceHTTPClient

    params = {"temperature": 0.7, "custom_logit_processor": "thinking-budget"}
    if batched:
        params = [params, {"temperature": 0.7}]
    original = copy.deepcopy(params)
    images = [[b"png-image"], []] if batched else [b"png-image"]
    body = InferenceHTTPClient._generate_body(
        None, [[1, 2], [3]] if batched else [1, 2], params, True, images
    )
    expected_processor = ["thinking-budget", None] if batched else "thinking-budget"
    assert body["custom_logit_processor"] == expected_processor
    clean_params = body["sampling_params"] if batched else [body["sampling_params"]]
    assert all("custom_logit_processor" not in item for item in clean_params)
    encoded = base64.b64encode(b"png-image").decode()
    assert body["image_data"] == ([[encoded], []] if batched else [encoded])
    assert params == original
    assert images == ([[b"png-image"], []] if batched else [b"png-image"])


def test_token_space_http_and_generation_metadata_preserve_prompt_ids():
    from rlinf.hybrid_engines.sglang.common.token_metadata import (
        decode_generation_result,
    )
    from rlinf.utils.http_client import InferenceHTTPClient

    body = InferenceHTTPClient._generate_body(
        None,
        [1, 9, 9, 2],
        {"max_new_tokens": 2},
        True,
        [b"image"],
        True,
        4,
    )
    assert body["return_prompt_token_ids"] is True
    assert body["mm_token_expansion_start_len"] == 4
    assert body["image_data"] == ["aW1hZ2U="]
    result = decode_generation_result(
        {
            "output_ids": [3],
            "prompt_token_ids": body["input_ids"],
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 4,
                "output_token_logprobs": [(-0.5, 3, None)],
            },
        },
        return_logprobs=True,
    )
    assert result["prompt_token_ids"] == [1, 9, 9, 2]
    assert result["logprobs"] == [-0.5]
    legacy = InferenceHTTPClient._generate_body(None, [1, 2], {}, False)
    assert "return_prompt_token_ids" not in legacy
    assert "mm_token_expansion_start_len" not in legacy


def _thinking_params(budget, output):
    # SGLang's external Req object supplies these three fields to its hook.
    request = SimpleNamespace(
        origin_input_ids=[0] * budget.prompt_tokens,
        output_ids=output,
        customized_info=None,
    )
    return {**budget.custom_params(), "__req__": request}


def test_preserved_history_and_current_tokenizer_ids():
    start, end = 248068, 248069
    prompt = [start, 8, end, 12, start, 198]
    budget = ThinkingBudget.from_prompt(prompt, 512, start, end)
    params = _thinking_params(budget, [7] * 512)
    logits = torch.ones(1, 248320)
    apply_thinking_budget(logits, [params])
    assert torch.isfinite(logits).sum().item() == 1
    assert logits[0, end].item() == 0
    assert params["__req__"].customized_info["forced_token_mask"] == [False] * 512 + [
        True
    ]


def test_natural_end_keeps_every_sampled_token():
    budget = ThinkingBudget(4, 7, 100000)
    output = [2, 7, 3, 4, 5, 6]
    logits = torch.randn(1, 9)
    original = logits.clone()
    params = _thinking_params(budget, output)
    apply_thinking_budget(logits, [params])
    torch.testing.assert_close(logits, original)
    assert budget.response_mask(output, []) == [1] * len(output)


def test_retracted_request_and_tool_continuation():
    budget = ThinkingBudget(2, 7, 100)
    params = _thinking_params(budget, [2, 3])
    apply_thinking_budget(torch.zeros(1, 9), [params])
    apply_thinking_budget(torch.zeros(1, 9), [params])
    request = params["__req__"]
    assert request.customized_info["forced_token_mask"] == [False, False, True]
    request.output_ids.extend([7, 4, 5, 6])
    logits = torch.randn(1, 9)
    original = logits.clone()
    apply_thinking_budget(logits, [params])
    torch.testing.assert_close(logits, original)
    assert budget.response_mask(request.output_ids, [2]) == [1, 1, 0, 1, 1, 1]
    assert request.output_ids == [2, 3, 7, 4, 5, 6]


def test_forced_token_has_zero_loss_gradient_and_preserves_normalization():
    budget = ThinkingBudget(2, 5, 10)
    ids = torch.tensor([1, 2, 5, 3])
    mask = torch.tensor(budget.response_mask(ids.tolist(), [2]))
    logits = torch.arange(32, dtype=torch.float64).reshape(4, 8).requires_grad_()
    logprobs = logits.log_softmax(-1).gather(-1, ids[:, None]).squeeze(-1)
    loss = -(logprobs * mask).sum() / mask.sum()
    reference = -logprobs[[0, 1, 3]].mean()
    grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(reference, logits)[0]
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(grad, reference_grad)
    assert torch.count_nonzero(grad[2]).item() == 0
    assert all(torch.count_nonzero(grad[row]).item() for row in [0, 1, 3])


def test_batch_requests_have_independent_budgets_and_disabled_path():
    params = [_thinking_params(ThinkingBudget(2, 7, 10), [3, 4]), None]
    logits = torch.randn(2, 9)
    baseline = logits.clone()
    apply_thinking_budget(logits, params)
    assert torch.isfinite(logits[0]).sum().item() == 1
    torch.testing.assert_close(logits[1], baseline[1])
    original = baseline.clone()
    apply_thinking_budget(baseline, None)
    torch.testing.assert_close(baseline, original)


@pytest.mark.parametrize("limit", [-1, True, 2.5])
def test_invalid_limits_are_rejected(limit):
    with pytest.raises(ValueError, match="nonnegative int"):
        ThinkingBudget(limit, 7, 10)


def test_wrong_prompt_or_missing_engine_evidence_is_rejected():
    with pytest.raises(ValueError, match="open thinking span"):
        ThinkingBudget.from_prompt([6, 2, 7], 2, 6, 7)
    budget = ThinkingBudget(2, 7, 10)
    params = _thinking_params(budget, [2, 3])
    params["__req__"].origin_input_ids.append(0)
    with pytest.raises(ValueError, match="prompt length"):
        apply_thinking_budget(torch.zeros(1, 9), [params])
    with pytest.raises(ValueError, match="closing token"):
        budget.response_mask([2, 3, 4], [2])
    with pytest.raises(ValueError, match="did not record"):
        budget.response_mask([2, 3, 7, 4], [])
    with pytest.raises(ValueError, match="outside the thinking boundary"):
        budget.response_mask([2, 7, 4], [2])


def test_token_space_rejects_ambiguous_spans_and_changed_training_prefix():
    from rlinf.hybrid_engines.sglang.multimodal.token_space import collapse_image_spans
    from rlinf.utils.multimodal import materialize_multimodal_inputs

    assert collapse_image_spans([1, 9, 9, 2, 9, 3], [(1, 3), (4, 5)], 9) == [
        1,
        9,
        2,
        9,
        3,
    ]
    for spans in ([(1, 2)], [(1, 4), (4, 5)], [(4, 5), (1, 3)]):
        with pytest.raises(ValueError):
            collapse_image_spans([1, 9, 9, 2, 9, 3], spans, 9)
    row = {
        "format": "sglang-token-space-v1",
        "prompt_token_ids": [1, 2],
        "image_refs": [],
    }
    ids = torch.tensor([[1, 3, 7]])
    with pytest.raises(ValueError, match="Training tokens"):
        materialize_multimodal_inputs(
            [row],
            "unused",
            "cpu",
            input_ids=ids,
            attention_mask=ids.ne(0),
            token_space=True,
        )
    with pytest.raises(ValueError, match="Legacy media"):
        materialize_multimodal_inputs(
            [{"image_grid_thw": torch.tensor([[1, 4, 4]])}],
            "unused",
            "cpu",
            input_ids=ids,
            attention_mask=ids.ne(0),
            token_space=True,
        )
    with pytest.raises(ValueError, match="requires token-space training"):
        materialize_multimodal_inputs([row], "unused", "cpu")


def test_token_space_text_only_rows_keep_padding_out_of_positions():
    from rlinf.utils.multimodal import materialize_multimodal_inputs

    ids = torch.tensor([[1, 2, 3, 0], [0, 4, 5, 6]])
    inputs = materialize_multimodal_inputs(
        [], "unused", "cpu", input_ids=ids, attention_mask=ids.ne(0), token_space=True
    )
    expected = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 2]])
    assert set(inputs) == {"multimodal_position_ids"}
    assert torch.equal(inputs["multimodal_position_ids"], expected.expand(3, -1, -1))


def test_native_token_space_multiturn_replay_and_cache(tmp_path):
    """Optional CPU integration: real SGLang core and local Qwen processor assets."""
    model_path = os.environ.get("RLINF_TOKEN_SPACE_TEST_MODEL")
    if not model_path:
        pytest.skip(
            "Set RLINF_TOKEN_SPACE_TEST_MODEL to local Qwen3.5 processor assets"
        )
    import io

    from PIL import Image
    from transformers import AutoConfig

    from rlinf.agents.rpent.history import ImagePrompt
    from rlinf.hybrid_engines.sglang.multimodal.token_space import SGLangTokenSpace
    from rlinf.utils.multimodal import materialize_multimodal_inputs

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    settings = {
        "model_path": model_path,
        "tokenizer_path": model_path,
        "tokenizer_mode": "auto",
        "tokenizer_backend": "huggingface",
        "trust_remote_code": False,
        "revision": None,
        "model_config_parser": "auto",
        "json_model_override_args": "{}",
        "image_processor_backend": "pil",
        "disable_fast_image_processor": True,
        "mm_process_config": {},
    }
    processor = SGLangTokenSpace(settings)
    images = []
    for mode, size, color in [
        ("RGB", (64, 96), "red"),
        ("RGBA", (96, 64), (0, 0, 255, 80)),
    ]:
        data = io.BytesIO()
        Image.new(mode, size, color).save(data, format="PNG")
        images.append(data.getvalue())
    marker = [
        config.vision_start_token_id,
        config.image_token_id,
        config.vision_end_token_id,
    ]
    first = asyncio.run(
        processor.prepare(
            [42, *marker, 43], images[:1], {"prefix_length": 0, "image_spans": [(2, 3)]}
        )
    )
    sampled = first["prompt_token_ids"] + [44, 45]
    compact = sampled + marker + [46]
    second = asyncio.run(
        processor.prepare(
            compact,
            images,
            {
                "prefix_length": len(sampled),
                "image_spans": [
                    *first["image_spans"],
                    (len(sampled) + 1, len(sampled) + 2),
                ],
            },
        )
    )
    assert second["prompt_token_ids"][: len(sampled)] == sampled
    assert second["image_spans"][0] == first["image_spans"][0]
    assert len(second["image_spans"]) == 2
    refs = ImagePrompt(
        second["prompt_token_ids"], images, second["image_grid_thw"]
    ).store_images(tmp_path)
    entry = {**refs, **second}
    ids = torch.tensor([second["prompt_token_ids"] + [47, 48, 0, 0]])
    mask = ids.ne(0)

    def materialize(value):
        return materialize_multimodal_inputs(
            [value],
            model_path,
            "cpu",
            cache_bytes=8 * 1024 * 1024,
            input_ids=ids,
            attention_mask=mask,
            token_space=True,
        )

    first_inputs = materialize(entry)
    again = materialize(entry)
    for key in first_inputs:
        assert torch.equal(first_inputs[key], again[key])
    assert first_inputs["multimodal_position_ids"].shape == (3, 1, ids.shape[1])
    # Cache reuse cannot hide changed placement, pixels, or files.
    with pytest.raises(ValueError, match="pixel_values differs"):
        materialize({**entry, "pixel_values_sha256": "changed"})
    with pytest.raises(ValueError, match="mrope_positions differs"):
        materialize({**entry, "mrope_sha256": "changed"})
    Path(entry["image_refs"][1]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        materialize(entry)
    processor.shutdown()
