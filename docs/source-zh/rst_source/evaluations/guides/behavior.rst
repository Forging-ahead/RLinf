BEHAVIOR-1K 评测
================

BEHAVIOR-1K 是基于 OmniGibson 与 Isaac Sim 的大规模家居场景仿真基准，控制双臂 R1 Pro 机器人完成抓取、放置、整理等家庭操作任务。RLinf 支持在 BEHAVIOR 环境中并行评测 OpenPI 等 VLA 策略，并输出 ``eval/success_once`` 等指标。

相关训练文档：:doc:`../../examples/embodied/behavior`

环境准备
--------

**安装依赖**

以下评测步骤供已备好模型和仿真环境的用户使用。已发布的 OpenPI 和 Isaac Sim 5.1 分别要求 PyTorch 2.7.1 和 2.7.0，按 RLinf 原有流程在同一环境安装时会遇到依赖冲突。模型兼容情况见 :doc:`../../examples/embodied/behavior`。

当前 ``evaluations/behavior/`` 仅提供 OpenPI π₀.₅ 示例；训练侧还支持 OpenVLA-OFT，可从 ``examples/embodiment/config/`` 中的训练配置派生评测 YAML（见 :doc:`../reference/configuration`）。

**硬件与 Isaac Sim**

BEHAVIOR 依赖 Isaac Sim 5.1，对 GPU 与驱动有额外要求，详见训练文档中的 `Isaac Sim 依赖说明 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_。要点如下：

- Isaac Sim 5.1 需要支持光线追踪的 GPU。NVIDIA 在 x86-64 平台上列出的最低规格为 RTX 4080；A100 和 H100 缺少 RT 核心，不在支持范围内。

适配 BEHAVIOR 3.9.3 的 RLinf OpenPI 镜像待发布。

**环境变量**

将数据根目录指向下载资产时使用的目录：

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data

OmniGibson 3.9.3 在这个目录下查找场景和物体资产 ``behavior-1k-assets/``、机器人资产 ``omnigibson-robot-assets/``、密钥 ``omnigibson.key`` 和任务实例 ``2026-challenge-task-instances/``。Python 导入 Isaac Sim 5.1 时会自动定位其安装目录。安装和下载步骤见 :doc:`../../examples/embodied/behavior` 中的“下载资产”。

示例配置
--------

``evaluations/behavior/`` 目录下已有以下示例：

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - 配置文件
     - 环境 preset
     - 模型
   * - ``behavior_openpi_pi05_eval.yaml``
     - ``behavior_r1pro``
     - π₀.₅（``openpi``）

评测 π₀.₅ 权重时使用上表中的配置，其中已指定推理模型类型和每次执行的动作步数。评测入口会自动启用 ``runner.only_eval``。

完整评测流程
------------

**Step 1：激活环境并设置路径**

.. code-block:: bash

   source .venv/bin/activate
   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data

**Step 2：准备模型**

推荐预训练权重：`RLinf/RLinf-Pi0-Behavior <https://huggingface.co/RLinf/RLinf-Pi0-Behavior>`_（下载命令见训练文档）。若使用第三方 OpenPI 权重（如 OpenPI-Comet），需先转换为 PyTorch 格式后再填入 ``rollout.model.model_path``。

**Step 3：编辑配置**

复制或编辑目标 YAML，至少修改 ``rollout.model.model_path``。通用 ``env.eval`` 字段见 :doc:`../reference/configuration` 中的 :ref:`env-eval-fields`；BEHAVIOR 特有字段与评测协议见下文 :ref:`behavior-eval-config`。

示例 ``behavior_openpi_pi05_eval.yaml`` 中还需保持与训练一致的 OpenPI 参数（``action_dim: 23``、``num_action_chunks: 32``、``openpi.config_name: pi05_behavior`` 等）。``num_action_chunks`` 是环境实际执行的 chunk；网络 horizon 来自 ``pi05_behavior`` 官方 ``TrainConfig.model.action_horizon``\ （**32**），除非另外设置 ``openpi.action_horizon``。

**Step 4：启动评测**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model

**Step 5：查看结果**

终端输出 ``eval/success_once``；日志与视频见 :doc:`../reference/results`。

.. _behavior-eval-config:

评测配置详解
------------

BEHAVIOR 评测 **每次运行对应单个任务** （由 ``omni_config.task.activity_name`` 指定）。以下字段共同决定并行规模、单条轨迹长度与初始场景实例。

评测协议概述
~~~~~~~~~~~~

`2026 BEHAVIOR 挑战 <https://behavior.stanford.edu/challenge/tasks/index.html>`_ 包含 100 个家居任务。请把 ``omni_config.scene.scene_model`` 设为该任务对应的场景。``behavior_r1pro`` preset 默认任务为 ``turning_on_radio``，场景为 ``house_double_floor_lower``。

每条评测轨迹由以下配置唯一确定：

- ``omni_config.task.activity_name``：任务名称，决定语言指令与 BDDL 定义；
- ``omni_config.task.activity_definition_id``：任务定义变体编号（通常为 ``0``）；
- ``omni_config.task.activity_instance_id`` 与 ``instance_resample_mode``：初始物体布局与机器人位姿。

``instance_resample_mode`` 支持三种模式：

- ``disabled`` （默认）：每次 reset 加载 ``activity_instance_id`` 对应的固定实例；若设置了 ``activity_instance_dir``，则从该目录读取对应 JSON。
- ``offline``：每次 reset 从选定的 split 或 ``activity_instance_dir`` 中 **随机** 选取一个缓存实例（需先下载官方 ``2026-challenge-task-instances`` 或用 ``instance_generator.py`` 生成）。
- ``online``：每次重置时现场生成物体布局。保持预设中的 ``activity_instance_dir: null``，省略 ``instance_split``；这两个字段用于选择已保存的布局。在线采样时设置 ``online_object_sampling: True`` 和 ``use_presampled_robot_pose: False``，每个仿真进程只运行一个环境。

.. note::

   单次启动不会自动遍历全部任务或 init state。要评测多个任务，需分别修改 ``activity_name`` 并多次运行，或编写批量脚本；要评测多个 instance，请使用 ``instance_resample_mode: offline`` 并配合 ``rollout_epoch`` 取平均。

``env.eval`` 通用字段
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - BEHAVIOR 设置建议
   * - ``total_num_envs``
     - 全局并行环境数。单个 BEHAVIOR 环境约占 **10 GiB** 显存，请按 GPU 显存酌情设置；示例默认 ``4``。
   * - ``rollout_epoch``
     - 评测轮次，相同配置下重复运行并取指标平均；示例默认 ``4``。
   * - ``max_episode_steps``
     - 单条轨迹步数上限。π₀.₅ 示例使用 ``4096`` （preset 默认 ``2000`` 偏短，长 horizon 任务建议调大）。
   * - ``max_steps_per_rollout_epoch``
     - 每轮 rollout 的总交互步数，**必须能被** ``rollout.model.num_action_chunks`` **整除**。无 ``auto_reset`` 时通常等于 ``max_episode_steps``。
   * - ``num_env_subprocess``
     - 每个本地环境 worker 启动的仿真进程数，预设值为 ``1``。自动重置或在线采样时，为每个环境分配一个进程。
   * - ``skip_intermediate_renders``
     - 设为 ``True`` 时跳过 chunk 内中间 observation，显著加速环境步进；保存的视频仅含 chunk 边界帧。

``omni_config`` 关键字段
~~~~~~~~~~~~~~~~~~~~~~~~

以下字段位于 ``env.eval.omni_config`` 下（继承自 ``examples/embodiment/config/env/behavior_r1pro.yaml``，可在评测 YAML 中覆盖）：

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
           activity_name: turning_on_radio
           activity_definition_id: 0
           activity_instance_id: 0
           activity_instance_dir: null          # 指向缓存 instance JSON 目录
           instance_file_format: tro_state        # template | tro_state
           instance_resample_mode: disabled       # disabled | offline | online
         scene:
           scene_model: house_double_floor_lower
           partial_scene_load: true               # 仅加载任务相关房间，缩短启动时间

更完整的字段说明（``partial_scene_load``、``instance_generator.py`` 用法等）见 :doc:`../../examples/embodied/behavior` 中的「behavior_r1pro.yaml 关键配置说明」。

GPU 与集群布局
~~~~~~~~~~~~~~

BEHAVIOR 环境步进较慢，通常建议给 env 分配足够 GPU，并与 rollout 共享或分离 placement：

.. code-block:: yaml

   cluster:
     component_placement:
       rollout,env: all          # env 与 rollout 共享全部 GPU（示例默认）

也可将 env 与 rollout 分到不同 GPU 以缓解显存压力，详见训练文档「关键集群配置」一节。

进阶用法
--------

**切换评测任务**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model \
     env.eval.omni_config.task.activity_name=picking_up_trash

**使用离线 instance 随机采样**

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
           activity_instance_dir: ${oc.env:OMNIGIBSON_DATA_PATH}/2026-challenge-task-instances/scenes/house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances
           instance_file_format: tro_state
           instance_resample_mode: offline
       rollout_epoch: 5

**调整并行规模**

以下示例假设同一个环境 worker 负责四个环境，并用两个仿真进程运行它们。

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model \
     env.eval.total_num_envs=4 \
     env.eval.num_env_subprocess=2

**评测训练得到的 π₀.₅ 权重**

将模型路径指向 RLinf 格式的 OpenPI checkpoint 目录。如果权重仍是上游 OpenPI 格式，请先按 :doc:`../../examples/embodied/sft_openpi` 中的步骤转换。

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/rlinf_openpi_checkpoint

常见问题
--------

- **数据下载：** BEHAVIOR 资源体积较大，请按 :doc:`../../examples/embodied/behavior` 完成 Isaac Sim、OmniGibson 资产与 license 配置后再评测。
- **仿真模块导入失败：** 启动评估前激活已安装 OmniGibson 3.9.3 和 Isaac Sim 5.1 的环境。
- **Headless 模式：** ``run_eval.sh`` 默认设置 ``OMNIGIBSON_HEADLESS=1``。
- **显存不足：** 降低 ``total_num_envs`` 或 ``num_env_subprocess``；单个环境约占 10 GiB 显存。
- **渲染马赛克/模糊：** 请核对 GPU 是否满足上文的 Isaac Sim 5.1 光线追踪要求。
- **启动极慢：** 首次加载大场景耗时较长；保持 ``partial_scene_load: true`` 可只加载任务相关房间。
- **视频帧数少于预期：** ``skip_intermediate_renders: True`` 会跳过 chunk 内中间帧，仅保留策略实际消费的 observation 对应帧。
- **instance 加载失败：** ``activity_instance_dir`` 中的 JSON 文件名须与 ``activity_name``、``activity_definition_id``、``scene_model`` 匹配；详见 ``rlinf/envs/sim/behavior/instance_loader.py``。
- **步数校验失败：** ``max_steps_per_rollout_epoch`` 必须能被 ``rollout.model.num_action_chunks`` 整除。
