RPent Multimodal Reinforcement Learning
=======================================

Train a Qwen3.5-4B planner on LIBERO with RPent and RLinf. RPent executes six end-effector tools and collects observations; RLinf samples model responses and updates the actor and critic. This workflow trains the planner with images and text, while the vision encoder and projector stay frozen.

.. important::

   The default image path preserves sampled tokens and checks image grids and positions. Experimental TokenSpace also checks processor, pixel-value and M-RoPE replay. These checks establish input consistency, not measured SGLang/Megatron logprob parity for either path. TokenSpace is off by default and requires the dependency versions in the installation guide.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model

      Full multimodal Qwen3.5-4B checkpoint

   .. grid-item-card:: Algorithm

      Synchronous PPO or asynchronous SAO

   .. grid-item-card:: Tools

      Direct end-effector control and SAM3 segmentation

   .. grid-item-card:: Hardware

      The deployment example uses two 8-GPU model nodes and one 8-GPU environment node

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Workflow
     - Scope
   * - This page: ``libero_rl``
     - Standard LIBERO, 40 tasks; seeds 1–32 for training, seed 0 for evaluation. These are initial-state splits of the same tasks.
   * - :doc:`Original Planner PPO <rpent>`
     - LIBERO-PRO with Pi0.5, task memory and text observations. Use its own configurations and installation instructions.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Item
     - Contract
   * - Observation
     - Measured state after initialization/actions; ``view_env`` returns camera images.
   * - Action
     - ``move_to``, ``set_gripper``, ``view_env``, ``segment``, ``back_project`` and ``finish``.
   * - Reward
     - Native environment success: 1 for success, 0 for failure. A planner ``finish`` call alone does not establish success.
   * - Task prompt
     - Task language with the selected tool prompt; default ``simple``.

Installation
------------

Use the paired ``dev/rpent-rlinf-integration`` branches of ``Sonorous281/RLinf`` and ``Sonorous281/RPent``. Complete :doc:`rpent_libero_rl_setup` for the two Python environments, full model and SAM3 weights, simulator assets, task data and Ray services. The baseline branch's README specifies a different RPent revision and must not be used for this deployment.

The setup page uses three nodes with eight GPUs each. Adjust placement and concurrency to your allocated resources before launch; this template is not a minimum-memory guarantee. The standalone RPent collector uses a different serving environment and defaults to LIBERO-PRO.

Prepare the Task Data
---------------------

The task generator writes ``train.jsonl`` and a manifest with task IDs, initial-state seeds, counts and hashes. Each input line has a task instruction and a JSON-encoded ``answer``:

.. code-block:: json

   {"prompt": "open the bottom drawer", "answer": "{\"suite\": \"libero_goal\", \"task\": 0, \"seed\": 1}"}

The suite/task pair selects the simulator task; use the generator to obtain its actual instruction. Rollouts retain sampled token IDs, behavior logprobs, image inputs, policy versions and the native outcome. Only model-generated response tokens contribute to the policy loss. Tool observations and forced closing tokens have zero loss mask.

Configure Deployment Paths
--------------------------

After installation, copy the deployment path template outside the repository:

.. code-block:: bash

   source "$STACK_ROOT/envs/rlinf/bin/activate"
   cd "$STACK_ROOT/repos/RLinf"
   mkdir -p "$STACK_ROOT/config/deployment"
   cp examples/agent/rpent/config/deployment/paths.yaml "$STACK_ROOT/config/deployment/local.yaml"

Edit ``$STACK_ROOT/config/deployment/local.yaml`` and replace every ``???`` with an absolute path. Set ``data.train_data_paths`` and ``data.val_data_paths`` to YAML lists of training and evaluation JSONL files. Fill the remaining fields with the log directory, RPent interpreter, LIBERO config directory, episode directory and full-model directory. Training, evaluation and resume share this file; command-line overrides still take precedence. Retain it with Hydra's resolved configuration to reproduce the run.

Run One Episode First
---------------------

This configuration uses two GPUs on one node: GPU 0 serves the model, and GPU 1 handles simulator rendering and SAM3. Complete the installation and asset steps first, but start a single-node Ray cluster with ``RLINF_NODE_RANK=0``. Start SAM3 separately on GPU 1 and set ``tools.rpent.sam3_endpoint`` to its address. This two-GPU configuration has not passed GPU validation; measure its memory requirements before increasing the workload.

Select one fixed evaluation case and cap it at 3 turns, 40 environment steps and 1024 generated tokens per request:

.. code-block:: bash

   "$STACK_ROOT/envs/rpent/bin/python" examples/agent/rpent/prepare_tasks.py \
     --workflow libero_rl --libero-type standard --libero-config-path "$LIBERO_CONFIG_PATH" \
     --seeds 0 --split eval --output-dir "$STACK_ROOT/data/libero-rl-eval"
   head -n 1 "$STACK_ROOT/data/libero-rl-eval/eval.jsonl" > "$STACK_ROOT/data/libero-rl-eval/smoke.jsonl"
   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_libero_rl_smoke \
     --config-dir "$STACK_ROOT/config" +deployment@_global_=local \
     data.val_data_paths="[$STACK_ROOT/data/libero-rl-eval/smoke.jsonl]"

Append ``--cfg job --resolve`` to the ``run_eval.sh`` command to inspect the resolved configuration without loading a model. A completed run writes ``runner.output_dir/rpent-libero-rl-smoke/eval_results.json`` with ``summary.episodes`` equal to 1, and retains requests, outcomes and logs in the episode directory. Inspect tool calls and the native outcome, and confirm the episode subprocess has exited. The task can fail within this budget: the check covers collection and cleanup, not task success, numerical parity or learning.

Run It
------

After the single-episode check, allocate the three-node placement and start Ray as described in the installation guide, then launch SAO:

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_sao_local \
     --config-dir "$STACK_ROOT/config" +deployment@_global_=local

For synchronous PPO, use ``train_qwen3.5_4b_libero_rl_ppo`` and adjust placement and batch size for the allocated resources. SAO also changes the algorithm: it uses DIS loss and SAO GAE, updates the critic, recomputes values, then updates the actor. It is more than asynchronous scheduling of PPO. Check finite losses and gradients, parameter updates, weight synchronization and recovery before scaling. A short smoke run does not establish capacity for the full 262144-token workload.

Visualization and Results
-------------------------

.. list-table::
   :header-rows: 1

   * - Evidence
     - What to check
   * - Run outputs
     - Training logs and actor/critic checkpoints under ``runner.output_dir`` / experiment name; episode requests, images and results under ``tools.rpent.output_dir``.
   * - Runtime checks
     - Finite losses/gradients, nonzero parameter updates, successful weight synchronization and checkpoint recovery on the deployed revision.
   * - Task results
     - ``rpent/success_rate`` reports rollout outcomes. Report native successes / completed cases and infrastructure failures separately; use matched independent evaluation to measure learning.
   * - CPU component regression
     - Covers prefixes, masks, trajectory merging, Gloo data exchange, process cleanup and config composition. Tests skipped for missing Megatron/SGLang dependencies do not count as passes.
   * - Real GPU numerical checks
     - Not accepted. Validate the default and TokenSpace paths separately on the fixed 96 points (mean ≤ 0.03, max ≤ 0.10), multi-turn images and M-RoPE. Recomputing logprobs or using TIS does not eliminate native logprob error.
   * - Checkpoint and recovery
     - Full training acceptance is pending for this branch. Verify both optimizers, post-update weight synchronization and fresh-process resume; SAO also needs buffer/pending-state and policy-version checks.
   * - Independent evaluation
     - No result establishes a learning gain. Compare native success before and after training with matched tasks, initial states and budgets.

Also inspect ``rpent/turns`` and ``rpent/infra_retry_rate``. The template sets ``agentloop.max_infra_retry_rate=1.0`` to accept recovered retries; exhausted per-episode retries still fail. Earlier numerical checks with forced closing tokens exceeded the diagnostic tolerance and do not validate this deployment.

Resume and Evaluate
-------------------

Resume Training
~~~~~~~~~~~~~~~

Use ``runner.resume_dir=/path/to/checkpoints/global_step_N`` to resume a complete SAO checkpoint, retaining both model/optimizer states and its buffered/pending rollout state. Use only a completed checkpoint with actor/critic state and the final ``data/data.pt`` marker. Pending prompts are requeued; this is not a persisted simulator state for halfway-finished episodes. Keep the dataset, algorithm, world size and referenced images compatible. A synchronous PPO checkpoint cannot provide the SAO queue state.

Independent Evaluation
~~~~~~~~~~~~~~~~~~~~~~

Run the commands below on separately allocated resources. The default placement uses one eight-GPU model node and a separate environment node; adjust it for your evaluation deployment.

The template matches the training task, sampling, episode budgets and history window. Apply any training overrides to evaluation too. RPent's standalone batch collector is for serving and data collection; ``eval_qwen3.5_4b`` belongs to the original LIBERO workflow.

The export needs the same full VLM used to initialize training, because it retains the frozen vision weights and processor. Replace ``--vlm`` accordingly when training from SFT. Use a new output directory; an HF export is for inference and cannot replace the optimizer/queue checkpoint for resume.

.. code-block:: bash

   # Export a completed checkpoint into a new full-model directory.
   python examples/agent/rpent/checkpoint.py export \
     --checkpoint /path/to/checkpoints/global_step_5 \
     --vlm "$STACK_ROOT/models/Qwen3.5-4B" \
     --output "$STACK_ROOT/models/policy-step5"

   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_libero_rl \
     --config-dir "$STACK_ROOT/config" +deployment@_global_=local \
     runner.output_dir="$STACK_ROOT/logs/eval-step5" \
     rollout.model.model_path="$STACK_ROOT/models/policy-step5" \
     tools.rpent.output_dir="$STACK_ROOT/data/eval-episodes-step5"

Run the same evaluation again for the starting model by changing only the model and output paths. Keep the manifest, task/initial-state IDs, prompt, sampling, turn/action budgets and compaction fixed. Report native success with numerator/denominator, turn-budget exhaustion and infrastructure failures separately. Training completion-queue success is not independent evaluation, and a fixed 40-case panel alone cannot establish a small learning gain.

Choose a Training Configuration
-------------------------------

.. list-table::
   :header-rows: 1

   * - Config (without ``.yaml``)
     - Use
   * - ``train_qwen3.5_4b_libero_rl_ppo``
     - Synchronous PPO
   * - ``train_qwen3.5_4b_libero_rl_sao``
     - SAO with completed-episode scheduling
   * - ``train_qwen3.5_4b_libero_rl_sao_3node``
     - SAO placement across three nodes
   * - ``train_qwen3.5_4b_libero_rl_sao_local``
     - Three-node SAO with required local paths
   * - ``eval_qwen3.5_4b_libero_rl``
     - Independent multimodal evaluation

The launchers default to the original ``libero`` configurations. Always pass the ``libero_rl`` config name explicitly. The PPO → SAO → three-node → local profiles inherit common settings.

.. list-table::
   :header-rows: 1

   * - Setting
     - Template value
   * - Model placement
     - Nodes 0–1: actor, critic and rollout share 16 GPUs in successive phases. Learners use TP2/CP1/DP8 and sequence parallel; rollout uses 16 TP1 replicas.
   * - Environment placement
     - Node 2: eight workers, 32 episode slots each, one SAM3 service on GPU 0. Eight simultaneous initializations per worker limit cold-start load.
   * - Scheduling
     - Batch 128; 256 in-flight episodes and buffer capacity. SAO consumes completed episodes. Generation pauses during learner updates on shared GPUs; environment work may continue.
   * - Training
     - 10 dataset epochs; save every five updates. With 1,280 rows, 100 learner updates. Actor LR 1e-6; critic LR 5e-6; one actor minibatch and two critic epochs per SAO update.
   * - Episode
     - 100 turns, 10,000 environment steps, four-hour budget; context 262144, output 32768.
   * - Sampling
     - Thinking enabled; temperature 1, top-p 0.95, top-k 20, presence penalty 1.5, repetition penalty 1.0.
   * - History
     - Message window 80→40 at 70% of context; no extra summarization.
   * - Inference capacity
     - Memory fraction 0.7 and 128 requests per replica are admission settings, not a measured throughput or memory-fit result.

SAO uses DIS actor loss (clip bounds 0.3/5.0), ``sao_gae`` (alpha 1.5), the SAO critic loss and recorded behavior logprobs. The fully asynchronous :doc:`Search-R1 <searchr1/async>` recipe has separate rollout GPUs and a different update schedule.

For long updates, ``rollout.drain_before_training=true`` finishes active episodes before pausing generation. Its buffer capacity must cover the in-flight limit. This changes admission timing, not task timeouts.

Optional Runtime Settings
-------------------------

Training and evaluation set ``rollout.sglang.disable_overlap_schedule=true`` so presence penalties and optional thinking limits read the current CPU token history. Validate sampling logprobs and forced-token alignment on the deployed SGLang version before changing it.

Set ``agentloop.chat_template_kwargs.enable_thinking=false`` to disable thinking, or ``agentloop.prompt=simple_brief`` for prompt guidance toward about 256 reasoning tokens per decision. The prompt does not enforce a decoder limit. Compare native success and cost from fresh episodes with fixed models and seeds; do not remove thinking tokens or behavior logprobs from recorded trajectories.

To cap the first thinking span already opened by the chat template, set ``agentloop.thinking_token_budget=512`` and ``rollout.sglang.enable_custom_logit_processor=true``, keeping ``enable_thinking=true`` and ``rollout.sglang.disable_overlap_schedule=true``. Configuration validation rejects a budget with overlap enabled because its CPU token history can lag decoding. The default budget is ``null``. The engine records which closing token it forces; that token has zero training mask, while a naturally generated closing token remains trainable. A later reopened thinking span is not capped by this option. Keep the returned IDs, behavior logprobs, forced-token flags and weight versions together through trajectory assembly.

Set ``actor.model.lm_head_chunk_size=256`` to calculate the output layer in blocks for response tokens that contribute to the loss. The default ``0`` disables chunking. Results are recomputed during the backward pass. It supports MBridge Qwen3.5 with PP1/CP1, including TP and sequence parallel. Use the Megatron version listed on the setup page. Entropy is a detached diagnostic only, so an entropy loss bonus is unsupported. FP8, MTP, deferred embedding gradients, muP and autocast are also rejected for this mode. These are support constraints, not evidence of higher throughput or learning gains.

These recipes use the in-process SGLang backend by default. To use the main
branch's HTTP server backend, add ``+rollout.sglang.backend_type=server`` to the
same launcher. Both paths carry images, thinking-budget processors and installed
weight versions. The server runs as a local child process and binds to loopback;
actor weights still transfer directly through the scheduler.

Tool results and images remain in subsequent prompts. Only policy-generated assistant tokens receive policy loss; user/tool/history tokens do not. Prefix-compatible turns merge, while context compression/history rewriting starts a new row. The critic and actor use the same response-position mapping. Shared trajectory packing keeps rows separate if even one scored token overlaps: merging would overwrite a learning target. This applies to text-only SearchR1 as well as RPent; packing regressions compare both losses and gradients. Response-only logits require CP1.

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
     python -m pytest tests/unit_tests/test_sglang.py -k qwen_image_processor

These CPU checks cover input IDs, image tensors, offsets and HF/SGLang M-RoPE agreement. They do not replace real-engine generation/logprob checks or a complete learner update and checkpoint recovery before deploying a new processor version.

.. toctree::
   :hidden:

   rpent_libero_rl_setup
