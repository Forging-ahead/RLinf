BEHAVIOR-1K Evaluation
======================

Evaluate an OpenPI π₀.₅ policy on household tasks with the R1Pro robot. RLinf
owns the simulation loop, uses the env worker to step OmniGibson, and sends
the head and wrist camera observations to its rollout worker.

Environment and Model
---------------------

Follow the :doc:`installation and asset instructions <../../examples/embodied/behavior>`
in the BEHAVIOR example. The simulator requires OmniGibson 3.9.3 and the matching
2026 task instances. The example uses the public
`PT50-CS32 weights <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_.
Other PyTorch π₀.₅ checkpoints need compatible BEHAVIOR inputs, normalization
statistics, and 23-dimensional actions.

Run the Evaluation
------------------

From the RLinf checkout, activate the model environment and set the paths to
your 2026 data and local weights:

.. code-block:: bash

   source .venv/bin/activate
   export OMNI_KIT_ACCEPT_EULA=YES
   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   export CHECKPOINT=/path/to/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32
   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     "rollout.model.model_path='$CHECKPOINT'"

``evaluations/behavior/behavior_openpi_pi05_eval.yaml`` uses the
``behavior_r1pro`` preset, including the three-camera input, and selects public
instance 301 of ``turning_on_radio``. By default, the env group runs on GPU 0,
the rollout group runs on GPU 1, and the evaluation uses four environments,
four rollout epochs, and a 4,096-step episode limit. Adjust
``cluster.component_placement.env.placement`` and
``cluster.component_placement.rollout`` for a different allocation. Configure
the simulator interpreter with ``BEHAVIOR_ENV_PYTHON``.

Check a Short Run
-----------------

To verify that the simulator, model, and RLinf workers communicate before
starting longer episodes, evaluate a single instance for two 32-action chunks:

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     "rollout.model.model_path='$CHECKPOINT'" \
     runner.max_epochs=1 env.eval.rollout_epoch=1 env.eval.total_num_envs=1 \
     env.eval.max_episode_steps=64 env.eval.max_steps_per_rollout_epoch=64 \
     env.eval.omni_config.env.action_frequency=30 env.eval.omni_config.env.rendering_frequency=30 \
     'env.eval.omni_config.camera.head_resolution=[128,128]' \
     'env.eval.omni_config.camera.wrist_resolution=[128,128]' \
     env.eval.video_cfg.fps=30 rollout.enable_offload=False

The log should report one trajectory and an episode length of 64 when the step
limit is reached. ``eval/q_score`` records the task's goal-condition score,
and ``eval/success_once`` records whether the task succeeded at least once.
A 64-step trajectory may have zero on both metrics. With
``skip_intermediate_renders: True``, the video contains chunk-boundary frames.
Logs and videos are stored under ``runner.logger.log_path``. See the
:doc:`results reference <../reference/results>` for details.

Change the Task or Budget
-------------------------

Set ``env.eval.omni_config.task.activity_name``, the corresponding
``env.eval.omni_config.scene.scene_model``, and the split and instance ID
together. Optional ``activity_instance_dir`` overrides the default 2026 instance
directory. Increase ``env.eval.max_episode_steps`` and
``env.eval.max_steps_per_rollout_epoch`` for longer tasks. Use a multiple of
``rollout.model.num_action_chunks`` for the latter. Reduce
``env.eval.total_num_envs`` if the simulator runs out of GPU memory.
