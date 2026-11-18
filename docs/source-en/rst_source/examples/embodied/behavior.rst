RL with Behavior Benchmark
==========================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/behavior.jpg
   :align: center
   :width: 90%

   The BEHAVIOR benchmark (image: `BEHAVIOR <https://behavior.stanford.edu>`__).

`BEHAVIOR <https://behavior.stanford.edu>`__ is a benchmark of everyday household
activities built on NVIDIA IsaacSim / OmniGibson. A dual-arm R1 Pro robot performs
long-horizon manipulation; RLinf uses it to RL-fine-tune vision-language-action (VLA)
policies.

Overview
--------

RL-finetune a VLA on BEHAVIOR household tasks with PPO.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      100 tasks in the 2026 challenge

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · ray-tracing GPUs

| **You'll do:** install IsaacSim deps → download assets + base model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · IsaacSim 5.1 + BEHAVIOR-1K assets (>30 GB) · a base checkpoint (steps below).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Detail
   * - Tasks
     - The `2026 BEHAVIOR Challenge <https://behavior.stanford.edu/challenge/tasks/index.html>`_ includes 100 household tasks. Select one with ``omni_config.task.activity_name`` and set its ``omni_config.scene.scene_model``.
   * - Robot
     - Dual-arm R1 Pro on IsaacSim / OmniGibson.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - Head-camera RGB (720×720) plus left/right wrist RealSense RGB (480×480).
   * - Action
     - 23-dim continuous: 3-DOF base (x, y, rz), 4-DOF torso, 2×7-DOF arms, and 2×1-DOF parallel-jaw grippers.


Installation
------------

.. warning::

   Check the IsaacSim software and hardware requirements before installing:

   - https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
   - https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html

   Isaac Sim 5.1 requires an RTX-capable GPU. NVIDIA lists the RTX 4080 as the minimum for x86-64 and does not support the A100 or H100, which lack RT cores.

.. include:: _setup_common.rst

**Option 1: Docker image** — the RLinf images for BEHAVIOR 3.9.3 (OpenVLA-OFT and OpenPI) are pending release.

**Option 2: Custom environment** — the RLinf model and simulator packages have conflicting PyTorch requirements. OmniGibson 3.9.3 (``v3.9.3-post2``) uses Isaac Sim 5.1 with PyTorch 2.7.0. OpenVLA-OFT requires 2.2.0 and OpenPI requires 2.7.1. For OpenPI evaluation, launch RLinf from the model environment and assign the simulation worker to a node group whose ``env_configs`` sets ``python_interpreter_path`` to the simulator environment's Python. See :doc:`the node group configuration example <../../guides/hetero>`.


Download the Assets
-------------------

Choose a data directory with space for more than 30 GB of downloads. Use the same directory when you launch RLinf.

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   mkdir -p "$OMNIGIBSON_DATA_PATH"
   # Optional Hugging Face mirror for mainland China: export HF_ENDPOINT=https://hf-mirror.com

**From a fresh checkout (Linux, Conda).** Follow the `official development setup <https://behavior.stanford.edu/getting_started/installation.html#setup>`_ with the `SDK revision used by RLinf <https://github.com/StanfordVL/BEHAVIOR-1K/tree/v3.9.3-post2>`_. The SDK creates a Python 3.11 environment, so use RLinf's Python 3.11-compatible LeRobot revision in this checkout.

.. code-block:: bash

   git clone --branch v3.9.3-post2 https://github.com/StanfordVL/BEHAVIOR-1K.git
   (
     cd BEHAVIOR-1K
     # Clear paths inherited from a standalone Isaac Sim installation.
     unset ISAAC_PATH EXP_PATH CARB_APP_PATH
     sed -i 's|wensi-ai/lerobot@release/b1k|wensi-ai/lerobot@436812bd8ee39b768c645c248c91f1330834e687|' OmniGibson/setup.py
     ./setup.sh --new-env --omnigibson --bddl --dataset
   )
   conda activate behavior

Follow the license prompts. The ``--dataset`` option downloads the robot assets, scenes and objects, and 2026 challenge task instances. The ``behavior`` environment contains the simulator. See Installation for RLinf model dependencies. Run the RLinf commands below from the RLinf checkout.

**With an existing OmniGibson 3.9.3 environment.** To download missing assets into the same data root, run the following in the active simulator environment:

.. code-block:: bash

   python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
   python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets()"
   python -c "from omnigibson.utils.asset_utils import download_2026_challenge_task_instances; download_2026_challenge_task_instances()"

Both methods use the `official asset repository <https://huggingface.co/datasets/behavior-1k/zipped-datasets/tree/main>`_. Check that your data root contains ``omnigibson-robot-assets/``, ``behavior-1k-assets/``, ``2026-challenge-task-instances/``, and the dataset key ``omnigibson.key``.

For unattended runs, export ``OMNI_KIT_ACCEPT_EULA=YES`` in the launch shell after accepting the `NVIDIA license <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html#running-isaac-sim>`_. Interactive runs can accept the license at the first Isaac Sim import.

Download the Model
------------------

Download the checkpoint for your model family (either method works):

.. code-block:: bash

   # Method 1: git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-Behavior
   git clone https://huggingface.co/RLinf/RLinf-Pi0-Behavior

   # Method 2: huggingface-hub (set HF_ENDPOINT=https://hf-mirror.com in mainland China)
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-Behavior --local-dir RLinf-OpenVLAOFT-Behavior
   hf download RLinf/RLinf-Pi0-Behavior --local-dir RLinf-Pi0-Behavior

.. include:: _model_path.rst

Run It
------

.. warning::

   Place BEHAVIOR env workers on GPUs starting from 0. IsaacSim can hang when env
   workers start on later GPU ranks.

Each recipe is a YAML config under ``examples/embodiment/config/``:

.. list-table::
   :header-rows: 1
   :widths: 34 26 40

   * - Model / purpose
     - Algorithm
     - Config
   * - OpenVLA-OFT
     - PPO
     - ``behavior_ppo_openvlaoft.yaml``
   * - π₀
     - PPO
     - ``behavior_ppo_openpi.yaml``
   * - π₀.₅
     - PPO
     - ``behavior_ppo_openpi_pi05.yaml``

Launch a config with ``run_embodiment.sh``:

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   bash examples/embodiment/run_embodiment.sh behavior_ppo_openvlaoft

**What this command does:**

1. Loads ``examples/embodiment/config/behavior_ppo_openvlaoft.yaml`` and its shared env config ``examples/embodiment/config/env/behavior_r1pro.yaml``.
2. Starts Ray workers for the actor, rollout, and BEHAVIOR env placement.
3. Runs PPO training and writes logs/checkpoints under ``runner.logger.log_path``.

.. admonition:: Configure further
   :class: note

   - BEHAVIOR throughput → increase env GPU count first, then tune ``env.train.num_env_subprocess`` and ``env.train.total_num_envs``.
   - Each BEHAVIOR process can use roughly 10 GiB of VRAM; tune subprocess count for your GPU memory.
   - Cached task instances → generate them with ``rlinf/envs/sim/behavior/instance_generator.py`` and ``examples/embodiment/config/env/behavior_r1pro.yaml``.
   - Placement and throughput → :doc:`Placement <../../concepts/placement>` and :doc:`Execution modes <../../concepts/execution_modes>`
   - Metric definitions and logging backends → :doc:`Training metrics <../../reference/metrics>`

.. warning::

   Known issue: under the current BEHAVIOR setup, training success rate
   (``env/success_once``) may stay at 0 for OpenVLA-OFT / π₀.
   This issue will be fixed in a later release.

Standalone Evaluation
~~~~~~~~~~~~~~~~~~~~~

In principle, any ``pi05`` checkpoint that has non-zero success rate on
Behavior and has been converted to PyTorch format can be used for evaluation
with this config. We use OpenPI-Comet only as an example source:

- https://huggingface.co/sunshk/openpi_comet/tree/main

After download, you can use the following repository to convert weights to
PyTorch format:

- https://github.com/mli0603/openpi-comet

Thanks to the OpenPI-Comet authors for open-sourcing the model and tools, which
helps reproducibility and evaluation in RLinf.

After conversion, update ``behavior_openpi_pi05_eval.yaml`` as follows:

1. Set ``rollout.model.model_path`` to the converted model directory.
2. Increase ``max_episode_steps`` and ``max_steps_per_rollout_epoch`` in
   ``env.eval`` if you need longer trajectories (for example, ``4096``).

.. code-block:: yaml

   env:
     eval:
       max_episode_steps: 4096
       max_steps_per_rollout_epoch: 4096

Run standalone evaluation through the :doc:`BEHAVIOR-1K evaluation guide <../../evaluations/guides/behavior>`.
It owns the required ``OMNIGIBSON_DATA_PATH`` setup, the
``behavior_openpi_pi05_eval`` launch command, and result interpretation.


Configure Further
-----------------

The BEHAVIOR env is driven by ``examples/embodiment/config/env/behavior_r1pro.yaml``.
RLinf first loads OmniGibson's base config (``base_config_name``) and then applies the
``omni_config`` overrides.
The fields below control reset behavior, scene loading, simulator frequencies, and
throughput — most have sensible defaults and only need tuning for custom tasks or
performance.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Key
     - Meaning
   * - ``base_config_name``
     - Base OmniGibson config (e.g. ``r1pro_behavior``) loaded before ``omni_config`` overrides.
   * - ``omni_config.task.type`` / ``omni_config.scene.type``
     - Keep ``BehaviorTask`` / ``InteractiveTraversableScene`` explicit so the intended
       upstream OmniGibson classes are selected after overrides are applied.
   * - ``omni_config.task.activity_name``
     - Set the task by name. For example, task index ``0`` in the task list corresponds to ``turning_on_radio``. With ``instance_resample_mode: disabled``, ``activity_instance_id`` selects the fixed starting layout.
   * - ``omni_config.task.instance_resample_mode``
     - Reset-time instance switching: ``disabled`` (load the fixed ``activity_instance_id``),
       ``offline`` (scan ``activity_instance_dir`` once and sample a cached instance per reset —
       ``*_template.json`` use the heavy scene-reload path, ``*_template-tro_state.json`` the
       lighter in-place path), or ``online`` (resample the scene at reset; see the :doc:`evaluation guide <../../evaluations/guides/behavior>` for settings).
   * - ``omni_config.task.activity_instance_dir``
     - Directory of cached instance JSONs (``*_template.json`` / ``*_template-tro_state.json``),
       used by ``offline`` mode and by fixed-id loading when the mode is ``disabled``.
   * - ``omni_config.task.instance_file_format``
     - Cached-instance format: ``template`` (full reload) or ``tro_state`` (light, task-relevant
       only). RLinf accepts ``tro_state`` files without ``robot_poses``; it then clears stale
       cached robot-pose metadata and the reset falls back to the task's default robot pose.
   * - ``omni_config.scene.partial_scene_load``
     - When ``true``, loads only the rooms assigned to the task in the 2026 challenge scene.
   * - ``camera.head_resolution`` / ``camera.wrist_resolution``
     - Head / wrist camera resolutions (defaults 720×720 / 480×480, applied to R1Pro sensors).
   * - ``omni_config.env.action_frequency`` / ``rendering_frequency`` / ``physics_frequency``
     - Action / render / physics stepping frequency (common default 60 / 60 / 120). Higher is slower.
   * - ``omni_config.env.automatic_reset``
     - Keep ``False`` — reset is controlled explicitly by the RLinf train/eval loop.
   * - ``omni_config.env.flatten_obs_space`` / ``flatten_action_space``
     - Keep ``False`` to preserve structured observation / action spaces.
   * - ``omni_config.macro.use_gpu_dynamics``
     - ``False`` usually improves performance; enable only for particles / fluids.
   * - ``omni_config.macro.enable_object_states``
     - Keep ``True`` — ``BehaviorTask`` depends on object states.
   * - ``omni_config.macro.enable_transition_rules``
     - ``True`` enables transition-rule state changes (e.g. slicing, cooking).
   * - ``omni_config.macro.use_numpy_controller_backend``
     - ``True`` uses the numpy controller backend, usually faster in single-/moderate-parallel runs.
   * - ``skip_intermediate_renders``
     - When ``True``, skips intermediate renders and returns the observation at the end of the action chunk. The simulator still reads sensors at each step.
   * - ``num_env_subprocess``
     - Number of simulator processes per local environment worker (preset default ``1``). For example, with four local environments and two processes, each process runs two scenes. Stages share this pool. Each stage needs its own simulator process because a physics step advances every scene in that process. Use one process per environment for independent resets.

Generating cached task instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``rlinf/envs/sim/behavior/instance_generator.py`` generates ``*_template.json`` and
``*_template-tro_state.json`` files directly from ``behavior_r1pro.yaml`` (it reads
``scene_model``, ``activity_name``, ``activity_definition_id``, the robot config, and
room-loading settings and resamples from the task's canonical scene template). It writes to
``activity_instance_dir`` if set, otherwise to ``OMNIGIBSON_DATA_PATH``'s default
``2026-challenge-task-instances`` directory for the selected scene and task; use ``--output-dir`` to override.

.. code-block:: bash

   cd /path/to/RLinf

   python -m rlinf.envs.sim.behavior.instance_generator \
     --config examples/embodiment/config/env/behavior_r1pro.yaml \
     --output-format template \
     --start-idx 1 --end-idx 50

   python -m rlinf.envs.sim.behavior.instance_generator \
     --config examples/embodiment/config/env/behavior_r1pro.yaml \
     --output-format tro_state \
     --start-idx 1 --end-idx 50

Generated filenames follow
``<scene_model>_task_<activity_name>_<activity_definition_id>_<activity_instance_id>_template(.json|-tro_state.json)``,
so ``--start-idx`` / ``--end-idx`` set the ``activity_instance_id`` range. ``tro_state``
outputs include top-level ``robot_poses`` only when the task metadata provides them; otherwise
the key is omitted and reset falls back to the task's default robot pose. BEHAVIOR-1K's upstream
``multiply_b1k_tasks.py`` still works, but RLinf's generator is recommended because it reads the
RLinf YAML directly and preserves ``activity_definition_id``.

--------------

**5. Evaluate a JAX-aligned Pi0.5 checkpoint**

BEHAVIOR evaluation uses ``model_type: openpi``. For weights from
:doc:`sft_openpi` or the OpenPI checkpoint convertor, the eval config is:

- ``evaluations/behavior/behavior_openpi_pi05_eval.yaml``

This config runs in eval-only mode (``runner.only_eval: True``) and consumes a
converted checkpoint, i.e. one produced by the OpenPI checkpoint
convertor (``ckpt_convertor.openpi`` ``openpi_pytorch_to_openpi`` /
``sft_to_openpi``). Set the model
paths directly in the config as ``/path/to/...`` placeholders:

- ``rollout.model.model_path``: the eval checkpoint.

Normalization statistics are loaded from the converted checkpoint's bundled
``physical-intelligence/behavior/norm_stats.json`` asset.

.. code:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval

.. note::

   Evaluation runs the flow-matching action head with non-deterministic
   (random) sampling noise, so per-run trajectories and success counts will vary
   slightly between repeated runs.


Visualization and Results
-------------------------

Launch TensorBoard to watch training live:

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

The key signal to watch is **``env/success_once``** — the task success rate. For every
logged metric, see :doc:`Training metrics <../../reference/metrics>`.

To save evaluation videos, enable them in the config:

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval


For the Behavior experiment, we were inspired by
`Behavior-1K baselines <https://github.com/StanfordVL/b1k-baselines.git>`_,
with only minor modifications. We thank the authors for releasing their open-source code.
