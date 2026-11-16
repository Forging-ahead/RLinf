RPent Installation And Deployment
=================================

Prepare the paired sources, model assets and services used by :doc:`rpent_libero_rl`. Run these steps once per deployment; use the main example for training and evaluation.

Install Code And Environments
-----------------------------

Use Linux NVIDIA hosts with a compatible driver, CUDA 12.8 toolkit, EGL/OpenGL libraries, Git and ``uv``. Place the paired RLinf and RPent checkouts under ``$STACK_ROOT/repos/RLinf`` and ``$STACK_ROOT/repos/RPent``. Use the same revisions and paths on every node, and record both revisions before installation.

.. code-block:: bash

   export STACK_ROOT="$HOME/rpent-stack"
   mkdir -p "$STACK_ROOT/repos" "$STACK_ROOT/envs" "$STACK_ROOT/models" "$STACK_ROOT/data"
   git clone --branch dev/rpent-rlinf-integration https://github.com/Sonorous281/RLinf.git "$STACK_ROOT/repos/RLinf"
   git clone --branch dev/rpent-rlinf-integration https://github.com/Sonorous281/RPent.git "$STACK_ROOT/repos/RPent"
   git -C "$STACK_ROOT/repos/RLinf" rev-parse HEAD
   git -C "$STACK_ROOT/repos/RPent" rev-parse HEAD
   git clone https://github.com/NVIDIA/Megatron-LM.git "$STACK_ROOT/repos/Megatron-LM"
   git -C "$STACK_ROOT/repos/Megatron-LM" checkout dc0ee41ada177d4c74ba46bb501bba5967264815
   export MEGATRON_PATH="$STACK_ROOT/repos/Megatron-LM"
   cd "$STACK_ROOT/repos/RLinf"
   git -C "$MEGATRON_PATH" apply \
     "$STACK_ROOT/repos/RLinf/requirements/agentic/megatron-core-r0.18.0-cpu-offload.patch"
   UV_TORCH_BACKEND=cu128 UNINSTALL_FA4=1 bash requirements/install.sh agentic \
     --venv "$STACK_ROOT/envs/rlinf" --python 3.12.12 --torch 2.11.0 \
     --engine sglang --sglang 0.5.12.post1 --transformers 5.6.1 \
     --use-mirror --no-root --install-rlinf

The installer honors an existing ``MEGATRON_PATH``. It temporarily edits dependency metadata while resolving, so use an idle checkout and a new environment. ``--no-root`` requires native system dependencies to be installed already. ``UNINSTALL_FA4=1`` selects the validated FA2 route. See ``bash requirements/install.sh --help`` for other platforms.

The patch above preserves CPU-offloaded master weights and Adam step counters when restoring a checkpoint. Apply it once to the pinned Megatron checkout before use. The installer applies it automatically to its own Megatron Core 0.18 checkout when ``MEGATRON_PATH`` is unset. If you set ``MEGATRON_PATH``, apply the patch yourself. The patch covers the Torch-format checkpoint path used by this recipe. Other checkpoint formats require separate validation.

Install the separate simulator/SAM3 environment from the paired RPent checkout. Its requirements install RLinf from the sibling checkout, so both environments use the same source revision. Verify the printed module paths before launching:

.. code-block:: bash

   cd "$STACK_ROOT/repos/RPent"
   uv venv --python 3.12 "$STACK_ROOT/envs/rpent"
   source "$STACK_ROOT/envs/rpent/bin/activate"
   uv pip install --prerelease=allow --torch-backend cu128 -r requirements/libero-rl.txt
   python -c 'import rlinf, robots.libero_rl; print(rlinf.__file__, robots.libero_rl.__file__)'

The tested stack uses these versions. This table records compatibility, rather than promising that every future dependency resolution reproduces the same environment.

.. list-table::
   :header-rows: 1

   * - Component
     - Tested versions
   * - RLinf
     - Python 3.12.12; Torch 2.11.0+cu128; Transformers 5.6.1; SGLang 0.5.12.post1
   * - Training kernels
     - Megatron Core 0.18.3 at the commit above; Bridge 0.5.0; Transformer Engine 2.17.0; FlashAttention 2.8.3 built for Torch 2.11
   * - Qwen linear attention
     - flash-linear-attention/fla-core 0.6.0; TileLang 0.1.8; Triton 3.6.0
   * - Simulator environment
     - Torch 2.11.0+cu128; Transformers 4.57.6; NumPy 1.26.4; MuJoCo 3.3.0; robosuite 1.5.2; rpent-libero/rpent-liberopro 0.2.0; SAM3 0.1.0

Check the active environment with ``uv pip freeze``, ``python -c 'import torch; print(torch.__version__, torch.version.cuda)'`` and ``git -C "$MEGATRON_PATH" rev-parse HEAD``. Preserve a package inventory after validating it. A matching CUDA compiler is needed for source builds; a networked CPU-only machine can download wheels/sources, but does not necessarily build the CUDA extensions. For offline nodes, prepare the complete dependency set on a compatible Linux/Python/CUDA build host, transfer it and the Git sources, then install with ``uv pip install --no-index --find-links /path/to/wheelhouse ...``. A partial wheel cache is not a complete offline installation.

All Ray nodes need the same RLinf environment, source revision and model files. Keep RPent in its own environment; do not upgrade the RLinf Transformers installation to match RPent's. When supplying custom activation profiles, preserve the installer's Megatron, CUDA and NCCL paths. This recipe uses FA2 and disables Torch compilation; set ``NVTE_FLASH_ATTN=1``, ``NVTE_FUSED_ATTN=0`` and ``TORCH_COMPILE_DISABLE=1`` before starting Ray. EGL must work on the environment node.

For an exact replay, check out the two commit IDs recorded with that run before installing. A branch name identifies the delivery channel, not an immutable experiment version.

PPO and SAO training use full-vocabulary sampling by default, with temperature 1, top-k -1, top-p 1, presence penalty 0 and repetition penalty 1. Evaluation retains the standalone planner sampling settings; explicitly use the same sampling settings when comparing policies. Thinking is enabled, with no forced thinking-token limit by default (``agentloop.thinking_token_budget: null``). Set this option explicitly to impose a limit.

Training uses the SGLang Server backend by default. Set ``rollout.sglang.backend_type=engine`` to use the in-process Engine backend. Both use the same rollout and weight synchronization interfaces. The SAO recipe collects episodes asynchronously with ``runner.synchronous: false``; actor and critic updates remain sequential. Rollout can overlap training when they use separate GPUs. Shared GPUs require rollout to pause and offload before training. This recipe does not use the critic-free ``runner.async_mode`` path.

Enable Logprob Recalculation and TIS
---------------------------------------------

To test SAO with training-side logprobs and truncated importance sampling (TIS), add these settings to the SAO config. The default SAO recipe leaves both options disabled.

.. code-block:: yaml

   algorithm:
     recompute_logprobs: true
     importance_sampling_fix: true
     importance_sampling_clip: 2.0
   rollout:
     return_logprobs: true
     max_weight_staleness: 1

Each batch retains its sampling logprobs and per-token policy versions. Before the actor update, the learner recalculates logprobs with the current actor weights. TIS corrects for both backend differences and policy lag. The DIS loss filters training tokens by their probability ratios, relative to the actor weights used for recalculation. With these settings, an episode is discarded if any token was sampled with weights more than one update behind the latest published version. Its task is then retried.

.. warning::

   Enabling these options changes how SAO computes its loss and adds an actor forward pass. Check TIS weight clipping, effective sample size and policy lag during a small run before scaling. Optimizer updates, weight synchronization and checkpoint recovery with these options have not been validated. Logprob recalculation and TIS do not establish numerical agreement between SGLang and Megatron.

Try Token-Space Image Inputs (Experimental)
-------------------------------------------

Token-space mode uses SGLang's image preprocessing for rollout and training. It checks the prompt IDs returned by inference, replays images from checksum-protected files, and checks the M-RoPE positions at the training model's language input. This mode supports Qwen3.5 image inputs with PIL preprocessing and Megatron-Bridge. It requires PP=1 and CP=1. Literal vision markers in text and generated responses are unsupported; video and audio are outside this path.

.. warning::

   This path is experimental. The default SGLang 0.5.12.post1 environment above cannot run it. The candidate dependency file, ``requirements/agentic/sglang-token-space.txt``, pins SGLang commit ``18a44326f6de5134fe921927d760d81f8ee48fc9``, Torch 2.14.1 and Transformers 5.19.0. Source installation requires a Rust toolchain. The pinned dependency profile uses CUDA 13; check driver compatibility before installing it. Prepare a separate environment with matching CUDA and Megatron-Bridge kernels; GPU serving, weight synchronization and training still need to be validated together in this environment. Keep the existing training environment unchanged.

After validating that environment, enable the following settings in the deployment profile. Remove ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`` from the rollout environment, including inherited shell settings. Keep ``output_dir`` accessible at the same path on rollout, actor and critic nodes.

.. code-block:: yaml

   rollout:
     sglang:
       enable_token_space_processor: true
   tools:
     rpent:
       store_images_on_disk: true
       keep_episodes: all

Rollout initialization fails if the TokenSpace APIs are unavailable. Both Server and Engine must return the prompt token IDs they actually use. An image preprocessing pass runs before sampling to calculate output and thinking budgets from the expanded prompt; its throughput cost has not been measured.

Start with fresh rollout state when switching modes. Async checkpoint recovery rejects a different mode, and the learner rejects changed processor fingerprints or mixed media formats. Do not assume optimizer state is portable across the Torch/kernel upgrade. Retain image files for checkpoint recovery.

To validate the stack for training, first verify multi-turn images, compaction and packed M-RoPE in the deployment environment. Then run the fixed 96-point logprob comparison. The mean absolute error must not exceed 0.03, and the maximum absolute error must not exceed 0.10. Compare base model logprobs separately from probabilities modified by sampling penalties, top-k and top-p. CPU input checks do not establish GPU numerical parity.

Reduce Repeated Image Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Token-space mode can share preprocessed images between rollout, actor and critic. The cache contains processed pixels and grid metadata. Each backend still runs its own frozen vision encoder; visual embeddings are not shared.

Shared preprocessing caches and CPU prefetching are disabled by default. To enable them, add the following to the deployment profile above. Replace ``<cache-dir>`` with the absolute path to a directory reserved for this job. Every process in the job must be able to access the directory at that path.

.. code-block:: yaml

   rollout:
     sglang:
       enable_token_space_preprocess_cache: true
       token_space_preprocess_cache_dir: <cache-dir>
       token_space_preprocess_cache_trust_url: true
       token_space_preprocess_cache_check_rate: 0.0
   tools:
     rpent:
       training_image_prefetch_batches: 2
       training_image_prefetch_bytes: 268435456

Cached image paths must always refer to the same image contents. The cache stores each processor fingerprint in a separate directory. The check rate is between 0 and 1: increasing it recomputes a fraction of cache hits to check their contents. Keep this setting identical across processes.

The prefetch settings limit the number and total bytes of prepared microbatches per learner rank, including the current microbatch. One CPU thread prepares the following batch while the training thread runs the current forward. The 256 MiB example excludes processor scratch space and the separate ``training_image_cache_bytes`` cache. Multiply the per-rank budgets by the number of actor and critic ranks, and measure peak host memory before increasing them. A microbatch larger than the budget fails with an error. Prefetching requires token-space trajectories that record the size of the preprocessed pixel data in bytes. Set ``training_image_prefetch_batches: 0`` when replaying older trajectories.

.. warning::

   Only trusted processes may write the cache directory: cache reads deserialize Python objects. The native disk cache has no eviction or disk-size limit. Use a job-owned directory with a filesystem quota or external capacity monitoring. Remove it only after all job processes have exited; image files must remain available for checkpoint recovery. A new process rebuilds its CPU prefetch queue from the saved image references.

CPU tests cover cache reuse, image integrity, prefetch ordering and thread cleanup. Validate GPU logprobs, training gradients, throughput and training recovery in a new process in the deployment environment. Compare each optimization against the same backend with the option disabled; the separate 96-point cross-backend acceptance limits above still apply.

Prepare Models And Standard LIBERO Assets
-----------------------------------------

Place the full Qwen3.5-4B model, or a compatible full SFT export, under ``$STACK_ROOT/models/``. It must include the vision weights, tokenizer, processor and chat template. A language-only extraction cannot serve adaptive image turns. Download the base model on a networked machine, for example:

.. code-block:: bash

   hf download Qwen/Qwen3.5-4B --local-dir "$STACK_ROOT/models/Qwen3.5-4B"

Obtain the SAM3 checkpoint following the `SAM3 repository <https://github.com/facebookresearch/sam3>`__, and set ``SAM3_CHECKPOINT_PATH`` to its local ``sam3.pt``. Assets are not bundled in the Python packages. In the RPent environment, download the standard assets and write a dedicated benchmark config:

.. code-block:: bash

   source "$STACK_ROOT/envs/rpent/bin/activate"
   export LIBERO_ASSETS="$STACK_ROOT/data/libero/assets"
   export LIBERO_CONFIG_PATH="$STACK_ROOT/data/libero-standard-config"
   env -u LIBERO_ASSET_PATH libero-download-assets \
     --assets-dir "$LIBERO_ASSETS" --skip-existing --no-cache
   mkdir -p "$LIBERO_CONFIG_PATH"
   python - <<'PY'
   import os
   from pathlib import Path
   import yaml
   import libero.libero as libero
   assets = Path(os.environ['LIBERO_ASSETS']).resolve()
   package_assets = Path(libero.__file__).resolve().parent / 'assets'
   # LIBERO also constructs scene/mesh paths relative to its installed package.
   if not package_assets.exists():
       if package_assets.is_symlink():
           raise RuntimeError(f'Broken LIBERO assets link: {package_assets}')
       package_assets.symlink_to(assets, target_is_directory=True)
   for scene in ['libero_tabletop_base_style.xml', 'libero_floor_base_style.xml',
                 'libero_kitchen_tabletop_base_style.xml',
                 'libero_living_room_tabletop_base_style.xml']:
       assert (package_assets / 'scenes' / scene).is_file(), scene
   config = libero.get_default_path_dict()
   config['assets'] = str(assets)
   Path(os.environ['LIBERO_CONFIG_PATH'], 'config.yaml').write_text(yaml.safe_dump(config))
   PY

The installed package's assets link is necessary because LIBERO constructs some XML/mesh paths relative to its source, independently of config.yaml. Preserve an existing complete assets directory; repair a broken link instead of replacing scene geometry. Keep the package's benchmark/BDDL/initial-state paths from ``get_default_path_dict()``. The ``train_qwen3.5_4b_libero_rl_sao_3node`` recipe passes this config explicitly to episode subprocesses, so a parent process configured for LIBERO-PRO cannot silently change the standard benchmark.

Create the standard 40-task training list with the RPent interpreter:

.. code-block:: bash

   cd "$STACK_ROOT/repos/RLinf"
   python examples/agent/rpent/prepare_tasks.py --workflow libero_rl \
     --libero-type standard --libero-config-path "$LIBERO_CONFIG_PATH" \
     --seeds 1-32 --output-dir "$STACK_ROOT/data/libero-rl-tasks"

These are initial-state splits over the same 40 tasks, not held-out task families. Check the SFT training manifest for overlap before calling them independent holdouts. This writes 1,280 rows and a manifest containing task/state counts, seeds and hashes. Seed 0 is reserved for evaluation. The generator rejects seeds outside each task's initial-state range instead of silently reusing an initial state through modulo indexing. For separate dev/test manifests, use ``--seeds 33-39 --split dev`` or ``--seeds 40-49 --split test`` with a separate output directory. The old memory-based generator remains the default ``--workflow libero`` and still requires ``--memory-dir``.

Start Three Ray Nodes And SAM3
------------------------------

Before ``ray start`` on **each node**, source the RLinf environment and export ``MEGATRON_PATH``, ``NVTE_FLASH_ATTN=1``, ``NVTE_FUSED_ATTN=0``, ``TORCH_COMPILE_DISABLE=1``, and a suitable ``RLINF_TIMEOUT`` such as ``10800``. Choose the actual network interface; ``RLINF_COMM_NET_DEVICES=eth0`` is an example, not a universal interface name. Ray workers inherit this environment at startup.

.. code-block:: bash

   source "$STACK_ROOT/envs/rlinf/bin/activate"
   export PYTHONPATH="$STACK_ROOT/repos/RPent:$STACK_ROOT/repos/RLinf:$MEGATRON_PATH${PYTHONPATH:+:$PYTHONPATH}"
   export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rlinf.hybrid_engines.sglang.multimodal
   export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 TORCH_COMPILE_DISABLE=1
   export RLINF_TIMEOUT=10800
   export HEAD_IP=10.0.0.10  # reachable head-node address
   export RLINF_NODE_RANK=0
   ray start --head --port 6387 --node-ip-address="$HEAD_IP" --num-gpus 8

On model node 1 and environment node 2, respectively, use the same source/environment and set the corresponding rank and local address:

.. code-block:: bash

   export RLINF_NODE_RANK=1  # use 2 on the environment node
   export NODE_IP=10.0.0.11 # use the environment node's address there
   ray start --address="$HEAD_IP:6387" --node-ip-address="$NODE_IP" --num-gpus 8

Run ``ray status --address="$HEAD_IP:6387"`` and verify three nodes and 24 GPUs. Do not start a second conflicting Ray cluster on an occupied node. Verify multi-node NCCL communication before a long job; network/firewall/RDMA settings depend on the cluster.

In a separate terminal on node 2, start SAM3 using the RPent environment:

.. code-block:: bash

   source "$STACK_ROOT/envs/rpent/bin/activate"
   cd "$STACK_ROOT/repos/RPent"
   export SAM3_CHECKPOINT_PATH="$STACK_ROOT/models/sam3/sam3.pt"
   python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8101

The agent workers and episode processes all run on node 2, so ``http://127.0.0.1:8101`` reaches their local SAM3 service. The local policy HTTP endpoint forwards model requests through RLinf channels to the two model nodes. There is no requirement to expose SAM3 across nodes.

