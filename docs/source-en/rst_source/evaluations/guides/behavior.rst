BEHAVIOR-1K Evaluation
======================

BEHAVIOR-1K is a large-scale household scene simulation benchmark built on OmniGibson and Isaac Sim. It tasks a dual-arm R1 Pro robot with manipulation skills such as pick-and-place, stacking, and tidying. RLinf supports parallel evaluation of OpenPI and other VLA policies in BEHAVIOR environments and reports metrics such as ``eval/success_once``.

Related training doc: :doc:`../../examples/embodied/behavior`

Environment Setup
-----------------

**Install dependencies**

These evaluation steps assume a working OpenPI and BEHAVIOR runtime. The published OpenPI package requires PyTorch 2.7.1 and Isaac Sim 5.1 requires 2.7.0, so the standard single-environment installation is blocked. See :doc:`../../examples/embodied/behavior` for model compatibility.

``evaluations/behavior/`` currently ships an OpenPI π₀.₅ example only. Training also supports OpenVLA-OFT; you can derive an eval YAML from ``examples/embodiment/config/`` (see :doc:`../reference/configuration`).

**Hardware and Isaac Sim**

BEHAVIOR depends on Isaac Sim 5.1 and has additional GPU and driver requirements; see the `Isaac Sim requirements <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_ in the training doc. Key points:

- Isaac Sim 5.1 requires an RTX-capable GPU. NVIDIA lists the RTX 4080 as the minimum for x86-64 and does not support the A100 or H100, which lack RT cores.

The RLinf OpenPI image for BEHAVIOR 3.9.3 is pending release.

**Environment variables**

Set the data root to the directory populated during asset setup:

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data

OmniGibson 3.9.3 looks for ``behavior-1k-assets/``, ``omnigibson-robot-assets/``, ``omnigibson.key``, and ``2026-challenge-task-instances/`` under this root. Isaac Sim 5.1 resolves its installation path when Python imports the package. See “Download the Assets” in :doc:`../../examples/embodied/behavior` for setup.

Example Configs
---------------

The following example is available under ``evaluations/behavior/``:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Config file
     - Env preset
     - Model
   * - ``behavior_openpi_pi05_eval.yaml``
     - ``behavior_r1pro``
     - π₀.₅ (``openpi``)

Use the evaluation config for π₀.₅ checkpoints. It specifies the rollout model type and action chunk length, and the evaluation entry point enables ``runner.only_eval`` automatically.

End-to-End Workflow
-------------------

**Step 1: Activate the environment and set the data path**

.. code-block:: bash

   source .venv/bin/activate
   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data

**Step 2: Prepare the model**

Recommended checkpoint: `RLinf/RLinf-Pi0-Behavior <https://huggingface.co/RLinf/RLinf-Pi0-Behavior>`_ (download commands in the training doc). Third-party OpenPI weights (e.g. OpenPI-Comet) must be converted to PyTorch format before setting ``rollout.model.model_path``.

**Step 3: Edit the config**

Copy or edit the target YAML and set at least ``rollout.model.model_path``. Generic ``env.eval`` fields are documented in :doc:`../reference/configuration` (:ref:`env-eval-fields`); BEHAVIOR-specific fields and the evaluation protocol are covered in :ref:`behavior-eval-config` below.

The OpenPI fields in ``behavior_openpi_pi05_eval.yaml`` must match training (``action_dim: 23``, ``num_action_chunks: 32``, ``openpi.config_name: pi05_behavior``, etc.). ``num_action_chunks`` is the env-executed chunk; the network horizon comes from official ``TrainConfig.model.action_horizon`` for ``pi05_behavior`` (**32**) unless ``openpi.action_horizon`` is set.

**Step 4: Launch evaluation**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model

**Step 5: Check results**

The terminal prints ``eval/success_once``; see :doc:`../reference/results` for logs and videos.

.. _behavior-eval-config:

Evaluation Configuration
--------------------------

BEHAVIOR evaluation runs **one task per launch** (selected by ``omni_config.task.activity_name``). The fields below control parallel scale, trajectory length, and initial scene instances.

Evaluation protocol
~~~~~~~~~~~~~~~~~~~

The `2026 BEHAVIOR Challenge <https://behavior.stanford.edu/challenge/tasks/index.html>`_ includes 100 household tasks. Set ``omni_config.scene.scene_model`` to the scene for the selected task. The ``behavior_r1pro`` preset defaults to ``turning_on_radio`` on scene ``house_double_floor_lower``.

Each evaluation trajectory is determined by:

- ``omni_config.task.activity_name``: task name (language instruction and BDDL definition);
- ``omni_config.task.activity_definition_id``: task definition variant (usually ``0``);
- ``omni_config.task.activity_instance_id`` and ``instance_resample_mode``: initial object layout and robot pose.

``instance_resample_mode`` supports three values:

- ``disabled`` (default): every reset loads the fixed instance for ``activity_instance_id``; if ``activity_instance_dir`` is set, the matching JSON is read from that directory.
- ``offline``: every reset **randomly** picks a cached instance from the configured split or ``activity_instance_dir`` (download official ``2026-challenge-task-instances`` or generate files with ``instance_generator.py``).
- ``online``: generate a fresh arrangement at each reset. In the preset, keep ``activity_instance_dir: null`` and omit ``instance_split``. These settings select saved layouts in the other modes. For online sampling, set ``online_object_sampling: True`` and ``use_presampled_robot_pose: False``, and run one environment per simulator process.

.. note::

   A single launch does not automatically sweep all tasks or init states. To evaluate multiple tasks, change ``activity_name`` and rerun, or wrap launches in a batch script. For multiple instances, use ``instance_resample_mode: offline`` and average over ``rollout_epoch``.

Generic ``env.eval`` fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - BEHAVIOR guidance
   * - ``total_num_envs``
     - Global parallel env count. Each BEHAVIOR env uses roughly **10 GiB** VRAM; the example defaults to ``4``.
   * - ``rollout_epoch``
     - Number of eval rounds with the same config; metrics are averaged. The example defaults to ``4``.
   * - ``max_episode_steps``
     - Max steps per trajectory. The π₀.₅ example uses ``4096`` (the preset default ``2000`` may be too short for long-horizon tasks).
   * - ``max_steps_per_rollout_epoch``
     - Total interaction steps per rollout round; **must be divisible by** ``rollout.model.num_action_chunks``. Without ``auto_reset``, usually equals ``max_episode_steps``.
   * - ``num_env_subprocess``
     - Simulator processes per local environment worker (preset default ``1``). Use one process per environment for automatic resets or online sampling.
   * - ``skip_intermediate_renders``
     - When ``True``, skips intermediate observations inside action chunks for faster stepping; saved videos only contain chunk-boundary frames.

Key ``omni_config`` fields
~~~~~~~~~~~~~~~~~~~~~~~~~~

These live under ``env.eval.omni_config`` (inherited from ``examples/embodiment/config/env/behavior_r1pro.yaml`` and overridable in eval YAML):

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
           activity_name: turning_on_radio
           activity_definition_id: 0
           activity_instance_id: 0
           activity_instance_dir: null          # directory of cached instance JSON files
           instance_file_format: tro_state        # template | tro_state
           instance_resample_mode: disabled       # disabled | offline | online
         scene:
           scene_model: house_double_floor_lower
           partial_scene_load: true               # load task-relevant rooms only

For full field descriptions (``partial_scene_load``, ``instance_generator.py``, etc.), see "behavior_r1pro.yaml key settings" in :doc:`../../examples/embodied/behavior`.

GPU and cluster placement
~~~~~~~~~~~~~~~~~~~~~~~~~

BEHAVIOR stepping is slow; allocate enough GPUs to env workers and share or split placement with rollout:

.. code-block:: yaml

   cluster:
     component_placement:
       rollout,env: all          # share all GPUs (example default)

You can also place env and rollout on separate GPUs to reduce memory pressure; see "Key cluster settings" in the training doc.

Advanced Usage
--------------

**Switch evaluation task**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model \
     env.eval.omni_config.task.activity_name=picking_up_trash

**Random offline instance sampling**

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
           activity_instance_dir: ${oc.env:OMNIGIBSON_DATA_PATH}/2026-challenge-task-instances/scenes/house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances
           instance_file_format: tro_state
           instance_resample_mode: offline
       rollout_epoch: 5

**Adjust parallel scale**

This example assumes one environment worker hosts all four environments and runs them in two simulator processes.

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model \
     env.eval.total_num_envs=4 \
     env.eval.num_env_subprocess=2

**Evaluate a trained π₀.₅ checkpoint**

Set the path to your RLinf-format OpenPI checkpoint. If the checkpoint uses the upstream OpenPI format, convert it as described in :doc:`../../examples/embodied/sft_openpi` first.

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/rlinf_openpi_checkpoint

FAQ
---

- **Data download:** BEHAVIOR assets are large; complete Isaac Sim, OmniGibson asset, and license setup per :doc:`../../examples/embodied/behavior` before evaluation.
- **Simulator import fails:** Activate the environment containing OmniGibson 3.9.3 and Isaac Sim 5.1 before running evaluation.
- **Headless mode:** ``run_eval.sh`` sets ``OMNIGIBSON_HEADLESS=1`` by default.
- **Out of memory:** Lower ``total_num_envs`` or ``num_env_subprocess``; each env uses about 10 GiB VRAM.
- **Blurry or blocky rendering:** Check that the GPU meets the Isaac Sim 5.1 ray-tracing requirements listed above.
- **Very slow startup:** First load of a large scene is expensive; keep ``partial_scene_load: true`` to load only task-relevant rooms.
- **Fewer video frames than expected:** ``skip_intermediate_renders: True`` skips intermediate chunk frames and keeps only observations consumed by the policy.
- **Instance load failure:** JSON filenames under ``activity_instance_dir`` must match ``activity_name``, ``activity_definition_id``, and ``scene_model``; see ``rlinf/envs/sim/behavior/instance_loader.py``.
- **Step count validation error:** ``max_steps_per_rollout_epoch`` must be divisible by ``rollout.model.num_action_chunks``.
