RPent Reinforcement Learning
============================

This example trains a multimodal Qwen3.5 policy with LIBERO feedback, using RPent to execute each episode and RLinf to optimize the policy.

Choose The Integration
----------------------

The launcher defaults to the original ``libero`` workflow. Its ``train_qwen3.5_4b_ppo`` and ``eval_qwen3.5_4b`` configs retain the original Pi0.5, memory and language-only model setup. See the :doc:`original LIBERO example </rst_source/examples/agentic/rpent>` for that workflow.

The separate ``libero_rl`` workflow uses a **complete multimodal** Qwen checkpoint, adaptive observation and six tools: ``move_to``, ``set_gripper``, ``view_env``, ``segment``, ``back_project`` and ``finish``. SAM3 supplies segmentation; the motion tools execute directly in LIBERO. Select it explicitly with one of these configs:

.. list-table::
   :header-rows: 1

   * - Config name, without ``.yaml``
     - Purpose
   * - ``train_qwen3.5_4b_libero_rl_ppo``
     - Synchronous PPO template
   * - ``train_qwen3.5_4b_libero_rl_sao``
     - Asynchronous SAO template
   * - ``train_qwen3.5_4b_libero_rl_sao_local``
     - Required deployment paths to fill locally
   * - ``train_qwen3.5_4b_libero_rl_sao_3node``
     - Shared 16-GPU SAO, plus a separate environment node
   * - ``eval_qwen3.5_4b_libero_rl``
     - Multimodal evaluation template

Install Code And Environments
-----------------------------

Use Linux NVIDIA hosts with a compatible driver, CUDA 12.8 toolkit, EGL/OpenGL libraries, Git and ``uv``. Place the paired RLinf and RPent checkouts under ``$STACK_ROOT/repos/RLinf`` and ``$STACK_ROOT/repos/RPent``. Use the same revisions and paths on every node, and record both revisions before installation.

.. code-block:: bash

   export STACK_ROOT="$HOME/rpent-stack"
   mkdir -p "$STACK_ROOT/repos" "$STACK_ROOT/envs" "$STACK_ROOT/models" "$STACK_ROOT/data"
   git -C "$STACK_ROOT/repos/RLinf" rev-parse HEAD
   git -C "$STACK_ROOT/repos/RPent" rev-parse HEAD
   git clone https://github.com/NVIDIA/Megatron-LM.git "$STACK_ROOT/repos/Megatron-LM"
   git -C "$STACK_ROOT/repos/Megatron-LM" checkout dc0ee41ada177d4c74ba46bb501bba5967264815
   export MEGATRON_PATH="$STACK_ROOT/repos/Megatron-LM"
   cd "$STACK_ROOT/repos/RLinf"
   UV_TORCH_BACKEND=cu128 UNINSTALL_FA4=1 bash requirements/install.sh agentic \
     --venv "$STACK_ROOT/envs/rlinf" --python 3.12.12 --torch 2.11.0 \
     --engine sglang --sglang 0.5.12.post1 --transformers 5.6.1 \
     --use-mirror --no-root --install-rlinf

The installer honors an existing ``MEGATRON_PATH``. It temporarily edits dependency metadata while resolving, so use an idle checkout and a new environment. ``--no-root`` requires native system dependencies to be installed already. ``UNINSTALL_FA4=1`` selects the validated FA2 route. See ``bash requirements/install.sh --help`` for other platforms.

Install the separate simulator/SAM3 environment from the paired RPent checkout. Its requirements point at a moving RLinf branch; the override below selects the delivered checkout instead. Verify the printed module paths before launching:

.. code-block:: bash

   cd "$STACK_ROOT/repos/RPent"
   uv venv --python 3.12 "$STACK_ROOT/envs/rpent"
   source "$STACK_ROOT/envs/rpent/bin/activate"
   python - <<'PYTHON' > "$STACK_ROOT/envs/rpent-source-overrides.txt"
   import os
   from pathlib import Path
   print("rlinf @ " + (Path(os.environ["STACK_ROOT"]) / "repos/RLinf").as_uri())
   PYTHON
   uv pip install --prerelease=allow --torch-backend cu128 \
     -r requirements/libero-rl.txt --override "$STACK_ROOT/envs/rpent-source-overrides.txt"
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

Trajectory Format
-----------------

Each JSONL row contains ``prompt`` and an ``answer`` encoded as a JSON string with ``suite``, ``task`` and ``seed``. The generator writes the task/initial-state manifest alongside the data. An episode produces ordered model requests, sampled token IDs, behavior logprobs, tool results, images and the native terminal reward (success 1, failure 0).

RLinf merges turns only while their token prefixes agree. A row records its prompt/response lengths, image inputs, per-token policy versions and response mask. Tool observations and forced closing tokens have mask 0; they receive neither policy loss nor a critic target. Actor and Critic inference results carry the same rollout identity even when two samples have identical text and different images.

Configure Shared 16-GPU SAO
---------------------------

Use the four clean-workflow profiles as an inheritance chain: ``train_qwen3.5_4b_libero_rl_ppo`` → ``train_qwen3.5_4b_libero_rl_sao`` → ``train_qwen3.5_4b_libero_rl_sao_3node`` → ``train_qwen3.5_4b_libero_rl_sao_local``. The first inherits the original ``train_qwen3.5_4b_ppo``, which remains available for the original LIBERO integration. Common task, sampling and model settings live in the clean PPO profile; SAO overrides its update rules, the three-node profile overrides placement, and the local profile requires deployment paths.

Fill ``examples/agent/rpent/config/train_qwen3.5_4b_libero_rl_sao_local.yaml``, or pass the required paths as launch overrides below. Unfilled ``???`` values are errors, not fallback paths. Set the full multimodal model directory, training JSONL, RPent interpreter, LIBERO configuration, episode directory and training output directory. The parent profiles require these paths instead of supplying machine-specific defaults. Setting ``rollout.model.model_path`` also sets the actor/critic tokenizer and HF import paths.

.. list-table::
   :header-rows: 1

   * - Resource or parameter
     - Configuration
   * - Nodes 0–1, GPUs 0–15
     - Rollout, critic and actor share all 16 GPUs in successive phases; weights, gradients and optimizer states offload between phases
   * - Learners
     - TP2, CP1, DP8; sequence parallel; BF16; activation recomputation; response-only logits
   * - Rollout
     - 16 TP1 SGLang replicas; ``replica_affinity=false``; ``max_running_requests=128`` per replica; memory fraction 0.7
   * - Node 2, GPUs 0–7
     - Eight agent workers, 32 episode slots each; one SAM3 service on GPU 0
   * - Completion scheduling
     - Batch 128; ``max_in_flight_rollouts=null`` and ``rollout_buffer_capacity=null`` each resolve to 256
   * - Training budget
     - 10 dataset epochs, no separate step cap; 1,280 input rows yield 100 learner updates; save every five updates
   * - Optimization
     - One actor minibatch and two critic update epochs per learner update; microbatch 1; actor LR ``1e-6``, critic LR ``5e-6``; language parameters train, vision encoder/projector remain frozen
   * - Episode
     - Simple prompt, adaptive observation, thinking, 100 turns, 10,000 env steps, four-hour budget
   * - Generation
     - Context 262144, output 32768; temperature 1, top-p 0.95, top-k 20, presence penalty 1.5, repetition penalty 1.0
   * - Context management
     - Native message window 80→40, trigger fraction 0.7; no extra tool clearing or summarization

The training and evaluation templates set ``rollout.sglang.disable_overlap_schedule=true``: presence penalties and optional thinking limits need the current CPU token history. Changing this setting requires verifying sampled logprobs and forced-token alignment with the installed SGLang version.

``agentloop.chat_template_kwargs.enable_thinking`` accepts ``true`` or ``false``; this
recipe defaults to ``true``. Compare native task success and generation cost on the
same checkpoint and task seeds before changing the mode for training. Start new
episodes for the comparison; do not remove thinking tokens from an existing
trajectory or its recorded behavior logprobs.

``agentloop.prompt=simple_brief`` selects RPent's optional concise-reasoning prompt,
which targets 256 reasoning tokens per decision without forcing decoder tokens.
It requires an RPent version that provides this prompt variant. Keep thinking
history and compare native success and measured cost before adopting it.

Optional Thinking And Projection Limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To cap the first thinking span already opened by the chat template, set ``agentloop.thinking_token_budget=512`` and ``rollout.sglang.enable_custom_logit_processor=true``, keeping ``enable_thinking=true`` and ``rollout.sglang.disable_overlap_schedule=true``. Configuration validation rejects a budget with overlap enabled because its CPU token history can lag decoding. The default budget is ``null``. The engine records which closing token it forces; that token has zero training mask, while a naturally generated closing token remains trainable. A later reopened thinking span is not capped by this option. Keep the returned IDs, behavior logprobs, forced-token flags and weight versions together through trajectory assembly.

For full projection chunking, set ``actor.model.lm_head_chunk_size=256``; the default ``0`` retains the normal projection. The chunked path projects scored response positions in blocks and recomputes them during backward. It supports MBridge Qwen3.5 with PP1/CP1, including TP and sequence parallel. Use the recorded Megatron version; this path replaces that model's postprocess operation temporarily. Entropy is a detached diagnostic only, so an entropy loss bonus is unsupported. FP8, MTP, deferred embedding gradients, muP and autocast are also rejected for this mode. These constraints protect supported execution semantics; they do not establish a throughput gain or learning benefit.

SAO keeps up to 256 episodes in flight and consumes the next 128 completed trajectories, rather than waiting for a fixed synchronous batch to finish. An episode's slot includes model waits and tool execution; eight workers with 32 slots can start all 256 in-flight episodes. ``max_env_initializations_per_worker=8`` bounds cold starts to 64 processes across the eight workers. A permit is released after the first environment state is saved, not at episode completion, so all 256 environments can remain active after warmup. This limits CPU/import/CUDA-context pressure without reducing rollout capacity. The buffer holds complete trajectories. Additional pending prompts can be queued, but do not increase the in-flight limit. Each episode normally has one outstanding model request, so a per-replica limit of 128 is an admission ceiling, not 128 continuously active requests on every GPU.

Shared GPUs still require a phase switch: pause/retract model requests, offload rollout memory, train the critic and actor, synchronize weights, then resume the waiting requests. Environment work may continue during training, but model generation pauses. SAO also changes the learning rules: DIS actor loss, clip bounds 0.3/5.0, ``sao_gae`` with alpha 1.5, and the SAO critic objective. It uses behavior logprobs saved during rollout instead of recomputing old-policy logprobs. The task, model, prompt, sampling and compression settings remain inherited from clean PPO.

Tool results and images remain in subsequent prompts. Only policy-generated assistant tokens receive policy loss; user/tool/history tokens do not. Prefix-compatible turns merge, while context compression/history rewriting starts a new row. The critic and actor use the same response-position mapping. Response-only logits require CP1.

When enabling ``tools.rpent.store_images_on_disk=true``, every rollout and learner process must read the **same absolute image paths**. ``tools.rpent.output_dir`` must be shared across model and environment nodes and retained while trajectories or saved checkpoints reference it. A same-named directory on separate local disks is insufficient. Leave the default ``false`` for installations without shared image storage; it retains image bytes in trajectories at a higher memory cost. Do not prune episode artifacts needed by a recoverable checkpoint.

Preserve Image-Turn Token Boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For merged Qwen3.5 image turns, enable the token-preserving processor on each model node **before starting Ray**:

.. code-block:: bash

   export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rlinf.hybrid_engines.sglang.multimodal

The adaptive train/eval templates also pass this variable through ``rollout.env_vars``, so worker launch does not depend solely on the terminal environment. This uses SGLang's external processor registry. Image requests carry already expanded prompt IDs and raw images. The adapter validates the number of images and each image's pad span against its processed grid, retains the exact text/action IDs, and lets SGLang derive offsets and M-RoPE from that sequence. Decoding and re-encoding sampled tokens can otherwise change their boundaries, even when the decoded text is identical. String requests retain the upstream behavior. Token-ID requests with video, audio or precomputed image inputs are rejected by this image-only adapter.

Run the processor regressions with a local model directory containing its config, tokenizer and processor files:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES='' SGLANG_USE_CUDA_IPC_TRANSPORT=0 \
     RLINF_TEST_QWEN_VL_MODEL=/path/to/Qwen3.5-4B \
     PYTHONPATH=.:tests/unit_tests \
     python -m pytest tests/unit_tests/test_agent_ppo.py -k qwen_image_processor

These CPU checks cover input IDs, image tensors, offsets and HF/SGLang M-RoPE agreement. They do not replace real-engine generation/logprob checks or a complete learner update and checkpoint recovery before deploying a new processor version.

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

Launch, Inspect And Resume
--------------------------

After filling the local profile, launch it from the head node:

.. code-block:: bash

   source "$STACK_ROOT/envs/rlinf/bin/activate"
   cd "$STACK_ROOT/repos/RLinf"
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_sao_local \
     runner.output_dir="$STACK_ROOT/logs/training" \
     data.train_data_paths="[$STACK_ROOT/data/libero-rl-tasks/train.jsonl]" \
     tools.rpent.python="$STACK_ROOT/envs/rpent/bin/python" \
     tools.rpent.libero_config_path="$LIBERO_CONFIG_PATH" \
     tools.rpent.output_dir="$STACK_ROOT/data/episodes" \
     rollout.model.model_path="$STACK_ROOT/models/Qwen3.5-4B"

Hydra overrides follow the config name, for example ``rollout.model.model_path="$STACK_ROOT/models/Qwen3.5-4B-SFT"``. Append ``--cfg job --resolve`` to inspect the resolved configuration without starting training. Use ``runner.resume_dir=/path/to/checkpoints/global_step_N`` to resume a complete SAO checkpoint, retaining both model/optimizer states and its buffered/pending rollout state. Use only a completed checkpoint with actor/critic state and the final ``data/data.pt`` marker. Pending prompts are requeued; this is not a persisted simulator state for halfway-finished episodes. Keep the dataset, algorithm, world size and referenced images compatible. A synchronous PPO checkpoint cannot provide the SAO queue state. Actor/critic checkpoints and logs live under ``runner.output_dir``/the experiment name; episode requests, tool images and results live under ``tools.rpent.output_dir``.

Check finite losses/gradients, nonzero parameter updates, actual replica activity, task success and infrastructure errors. Use Hydra overrides for an isolated short smoke run, rather than maintaining another smoke YAML. A successful short smoke test establishes connectivity and optimizer execution; it does not establish that the longest 262144-token workload fits in memory. Keep failure evidence and diagnose memory limits before changing sharding or budgets.

For independent evaluation, use a separate available deployment; the default evaluation placement expects one eight-GPU model node and a separate environment node. Do not launch this alongside training on the same occupied GPUs. Its raw template uses LIBERO-PRO, a 1200-step budget and a shorter history window, so it is **not** directly comparable to the standard training recipe. The overrides below align those settings. Adjust placement to the resources actually assigned to evaluation.

The export needs the same full VLM used to initialize training, because it retains the frozen vision weights and processor. Replace ``--vlm`` accordingly when training from SFT. Use a new output directory; an HF export is for inference and cannot replace the optimizer/queue checkpoint for resume.

.. code-block:: bash

   # Generate the same fixed cases once, using the simulator environment.
   "$STACK_ROOT/envs/rpent/bin/python" examples/agent/rpent/prepare_tasks.py \
     --workflow libero_rl --libero-type standard --libero-config-path "$LIBERO_CONFIG_PATH" \
     --seeds 0 --split eval --output-dir "$STACK_ROOT/data/libero-rl-eval"

   # Export a completed checkpoint into a new full-model directory.
   python examples/agent/rpent/checkpoint.py export \
     --checkpoint /path/to/checkpoints/global_step_5 \
     --vlm "$STACK_ROOT/models/Qwen3.5-4B" \
     --output "$STACK_ROOT/models/policy-step5"

   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_libero_rl \
     runner.output_dir="$STACK_ROOT/logs/eval-step5" \
     data.val_data_paths="[$STACK_ROOT/data/libero-rl-eval/eval.jsonl]" \
     rollout.model.model_path="$STACK_ROOT/models/policy-step5" \
     tools.rpent.python="$STACK_ROOT/envs/rpent/bin/python" \
     tools.rpent.output_dir="$STACK_ROOT/data/eval-episodes-step5" \
     +tools.rpent.libero_config_path="$LIBERO_CONFIG_PATH" \
     tools.rpent.libero_type=standard tools.rpent.max_episode_steps=10000 \
     agentloop.episode_timeout_s=14400 tools.rpent.process_timeout_s=15000 \
     agentloop.history_compaction.max_messages=80 \
     agentloop.history_compaction.keep_messages=40 \
     +algorithm.sampling_params.presence_penalty=1.5

Run the same evaluation again for the starting model by changing only the model and output paths. Keep the manifest, task/initial-state IDs, prompt, sampling, turn/action budgets and compaction fixed. Report native success with numerator/denominator, turn-budget exhaustion and infrastructure failures separately. Training completion-queue success is not independent evaluation, and a fixed 40-case panel alone cannot establish a small learning gain.

Metrics And Validation Boundaries
---------------------------------

Watch ``rpent/success_rate``, ``rpent/turns``, ``rpent/infra_retry_rate`` and the actor/critic loss and gradient metrics. Report native successes over completed cases and keep infrastructure failures separate. The template permits recovered retries (``agentloop.max_infra_retry_rate=1.0``); exhaustion of the per-episode retry limit still fails. Sync PPO and SAO apply the configured retry policy through the same task hook.

These templates are starting configurations, not recorded learning results. CPU regressions cover trajectory identity, action masks, packet metadata, stale-data admission and tool/context boundaries. Verify real SGLang requests, learner updates and checkpoint recovery on the intended deployment. Optional thinking closure and LM Head chunking remain disabled by default; the historical forced-close numerical probe did not satisfy its diagnostic tolerance. Neither these tests nor the training completion curve establish a learning gain.
