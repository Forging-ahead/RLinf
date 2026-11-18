BEHAVIOR-1K 评测
================

使用 OpenPI π₀.₅ 评测仿真机器人 R1Pro 的家庭操作任务。RLinf 的 env worker 推进 OmniGibson 仿真，并把头部与双腕相机的观测交给 rollout worker 生成下一批动作。

环境和权重
----------

按 :doc:`BEHAVIOR 示例中的安装与资产说明 <../../examples/embodied/behavior>` 准备文件。仿真端使用 OmniGibson 3.9.3 和配套的 2026 年任务实例。示例使用公开的 `PT50-CS32 权重 <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_。其他 PyTorch 格式的 π₀.₅ 权重需要匹配 BEHAVIOR 的观测、归一化统计量和 23 维动作。

启动评测
--------

在 RLinf 仓库根目录激活模型环境，并指定 2026 年仿真数据与本地权重路径：

.. code-block:: bash

   source .venv/bin/activate
   export OMNI_KIT_ACCEPT_EULA=YES
   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   export CHECKPOINT=/path/to/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32
   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     "rollout.model.model_path='$CHECKPOINT'"

``evaluations/behavior/behavior_openpi_pi05_eval.yaml`` 使用 ``behavior_r1pro`` 的三相机配置，默认任务为 ``turning_on_radio``，公开测试实例编号为 301。默认由 GPU 0 运行仿真、GPU 1 运行模型，使用 4 个环境、4 个 rollout epoch，每个 episode 上限 4,096 步。可通过 ``cluster.component_placement.env.placement`` 和 ``cluster.component_placement.rollout`` 调整 GPU 分配。``BEHAVIOR_ENV_PYTHON`` 指定仿真端解释器。

验证 64 步短流程
----------------

在长时间评测前，可以先运行一个实例，让模型预测两批动作，每批 32 步，检查仿真、模型与 RLinf worker 之间的通信：

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     "rollout.model.model_path='$CHECKPOINT'" \
     runner.max_epochs=1 env.eval.rollout_epoch=1 env.eval.total_num_envs=1 \
     env.eval.max_episode_steps=64 env.eval.max_steps_per_rollout_epoch=64 \
     env.eval.omni_config.env.action_frequency=30 env.eval.omni_config.env.rendering_frequency=30 \
     'env.eval.omni_config.camera.head_resolution=[128,128]' \
     'env.eval.omni_config.camera.wrist_resolution=[128,128]' \
     env.eval.video_cfg.fps=30 rollout.enable_offload=False

达到步数上限时，日志应显示 1 条轨迹，episode 长度为 64。``eval/q_score`` 是任务目标条件得分，``eval/success_once`` 记录是否至少成功过一次。短流程的两项指标都可能为 0。设置 ``skip_intermediate_renders: True`` 时，视频记录动作块边界的画面。日志和视频保存在 ``runner.logger.log_path`` 下，详见 :doc:`结果说明 <../reference/results>`。

更换任务和调整步数
------------------

切换任务时，同步设置 ``env.eval.omni_config.task.activity_name``、对应的 ``env.eval.omni_config.scene.scene_model``、数据划分和实例编号。如果设置 ``activity_instance_dir``，会优先读取指定目录的实例。需要更长的测试时，增加 ``env.eval.max_episode_steps`` 和 ``env.eval.max_steps_per_rollout_epoch``，后者应是 ``rollout.model.num_action_chunks`` 的整数倍。显存不足时可以调小 ``env.eval.total_num_envs``。
