# Share One Reason Environment

Use the Linux x86_64 `reason` image on every model host. The offline package
extends `/opt/venv/reason`; RPent, LIBERO, SAM3, SGLang and Megatron use that
same interpreter. It does not install OpenPI, upgrade the driver or install FA4.

```bash
export STACK_ROOT=/mnt/public/xzxuan
bash "$STACK_ROOT/envs/offline_package/install.sh"
source "$STACK_ROOT/envs/offline_package/activate.sh"
python -m pip check
```

The package checks the image and file hashes, backs up replaced packages and
installs local wheels without an index or Git access. Activation raises the process
file-descriptor soft limit to at least 65536 before Ray or services start.
Rerunning the installer keeps the same package versions. The installer retains
its pip-check report; runtime qualification uses
real rollout and optimizer tests. Copy the complete package to a node with the same image;
model checkpoints and LIBERO assets remain separate shared inputs.

| Component | Version |
| --- | --- |
| Python | 3.12.12 |
| Torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128, retained from the image |
| FA2 / Transformer Engine | 2.8.3 / 2.17.0, retained from the image |
| SGLang | 0.5.12.post1 |
| Megatron-Core / Bridge | 0.18.0+rpent.cu128.1 / 0.5.0+rpent.cu128.1 |
| Transformers / NumPy | 5.8.1 / 1.26.4 |
| MuJoCo / robosuite | 3.3.0 / 1.5.2 |

The unmodified SGLang release uses RLinf's external Qwen image processor.
It captures the actual serving prompt IDs, processed pixels, image spans,
grid and M-RoPE. Learners replay these exact inputs and run their own vision
encoders. No SGLang source backport or import override is installed. Native
TokenSpace remains available for older runs through upstream's separate profile;
new H200 runs use release capture and fresh rollout state.

The Bridge wheel uses FLA core for Qwen3.5 and the image's FlashInfer and Hydra
versions. The Core wheel uses installed distribution metadata for namespace NVRx packages
and selects the standard checkpoint implementation when their optional async
API is unavailable. It includes the CPU-offload checkpoint repair shipped in
`requirements/agentic/megatron-core-r0.18.0-cpu-offload.patch`. These builds
target the RPent Qwen3.5 path; they do not qualify every optional Bridge model.

Project and simulator wheels carry a `+rpent.h200.1` version suffix so
matching PyPI release numbers cannot replace the forked source.

## Start Local Services

Run one service manager on each node. Use a fresh output directory for each start:

```bash
python -m robots.rl.services \
  --sam3-checkpoint "$SAM3_CHECKPOINT_PATH" \
  --envs-per-gpu 3 --output-dir "$STACK_ROOT/logs/services-node0"
```

It starts one SAM3 per GPU, a local pool on port 28424 and an environment
backend on port 28490. The service manager stops its child processes on exit. Eight GPUs with
three slots each provide room for a node's share of the 80-candidate smoke;
the cohort size remains 80. Resource capacity must be checked for a different
batch or context size.

Start Ray with this environment and a distinct `RLINF_NODE_RANK` per node.
Launch training only from the head:

```bash
cd "$RLINF_ROOT"
bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_h200_smoke_grpo \
  --config-dir "$CONFIG_DIR" +deployment@_global_=local
```

The smoke selects 16 complete groups of four from 20 candidate groups,
runs one update and saves a checkpoint. The production GRPO recipe selects
32 complete groups from 40 candidate groups, with eight episodes per group.
Both use TP2/CP1; the four-node actor has DP16.
Use `/opt/venv/reason/bin/python` for `tools.rpent.python`. The deployment
profile supplies model, task, image-cache and output paths.

Before accepting a deployment, check a real 27B trajectory, the complete
4B optimizer update, saved optimizer/RNG/data state and post-update generation.
A successful import, a completed cohort or a zero-advantage batch alone does
not demonstrate effective learning.
