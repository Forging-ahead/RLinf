基于Behavior评测平台的强化学习训练
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/behavior.jpg
   :align: center
   :width: 90%

   BEHAVIOR 基准（图片来源：`BEHAVIOR <https://behavior.stanford.edu>`__）。

`BEHAVIOR <https://behavior.stanford.edu>`__ 是一个基于 NVIDIA IsaacSim / OmniGibson 的日常
家居活动基准。双臂 R1 Pro 机器人执行长程操作任务；RLinf 借助它对视觉-语言-动作（VLA）策略进行
强化学习微调。

概览
----------------------------------------

在 BEHAVIOR 家居任务上用 PPO 对 VLA 进行强化学习微调。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      2026 挑战的 100 个任务

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 支持光追的 GPU

| **你将完成：** 安装 IsaacSim 依赖 → 下载资产与基座模型 → 运行 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · IsaacSim 5.1 与 BEHAVIOR-1K 资产（>30 GB）· 基座检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 说明
   * - 任务
     - `2026 BEHAVIOR 挑战 <https://behavior.stanford.edu/challenge/tasks/index.html>`_ 包含 100 个家居任务。用 ``omni_config.task.activity_name`` 选择任务，并在 ``omni_config.scene.scene_model`` 填写对应场景。
   * - 机器人
     - IsaacSim / OmniGibson 上的双臂 R1 Pro。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 观测 (Observation)
     - 头部相机 RGB（720×720）以及左右腕部 RealSense RGB（480×480）。
   * - 动作 (Action)
     - 23 维连续动作：3 自由度底盘（x, y, rz）、4 自由度躯干、2×7 自由度手臂、2×1 自由度平行夹爪。


安装
----------------------------------------

.. warning::

   安装前先检查 IsaacSim 软硬件要求：

   - https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
   - https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html

   Isaac Sim 5.1 需要支持光线追踪的 GPU。NVIDIA 在 x86-64 平台上列出的最低规格为 RTX 4080；A100 和 H100 缺少 RT 核心，不在支持范围内。

.. include:: _setup_common.rst

**选项 1：Docker 镜像** — 适配 BEHAVIOR 3.9.3 的 RLinf 镜像（OpenVLA-OFT、OpenPI）待发布。

**选项 2：自定义环境** — OmniGibson 3.9.3（``v3.9.3-post2``）使用 Isaac Sim 5.1，需要 PyTorch 2.7.0。RLinf 的 OpenVLA-OFT 和 OpenPI 分别要求 PyTorch 2.2.0 和 2.7.1，按发布的依赖安装时，两种模型都会与仿真器冲突。使用 OpenPI 评测时，从模型环境启动 RLinf，将仿真 worker 分配到单独的节点组，在该组的 ``env_configs`` 中将 ``python_interpreter_path`` 指向仿真环境的 Python。配置示例见 :doc:`异构环境指南 <../../guides/hetero>`。


下载资产
----------------------------------------

选择一个数据目录，下载量超过 30 GB。之后启动 RLinf 时也指向这个目录。

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   mkdir -p "$OMNIGIBSON_DATA_PATH"
   # 国内可选用 Hugging Face 镜像：export HF_ENDPOINT=https://hf-mirror.com

**首次安装（Linux、Conda）。** 按照 `官方开发环境安装指南 <https://behavior.stanford.edu/getting_started/installation.html#setup>`_ 获取 `RLinf 使用的 SDK 版本 <https://github.com/StanfordVL/BEHAVIOR-1K/tree/v3.9.3-post2>`_。SDK 创建的是 Python 3.11 环境，因此先在这份新克隆的源码中将 LeRobot 固定到 RLinf 使用的兼容版本。

.. code-block:: bash

   git clone --branch v3.9.3-post2 https://github.com/StanfordVL/BEHAVIOR-1K.git
   (
     cd BEHAVIOR-1K
     # 清除旧版独立安装的 Isaac Sim 路径。
     unset ISAAC_PATH EXP_PATH CARB_APP_PATH
     sed -i 's|wensi-ai/lerobot@release/b1k|wensi-ai/lerobot@436812bd8ee39b768c645c248c91f1330834e687|' OmniGibson/setup.py
     ./setup.sh --new-env --omnigibson --bddl --dataset
   )
   conda activate behavior

按提示确认许可。``--dataset`` 会下载机器人资产、场景与物体资产，以及 2026 挑战任务实例。这里创建的 ``behavior`` 环境用于仿真，RLinf 模型依赖见上方“安装”一节。后续 RLinf 命令从 RLinf 仓库根目录运行。

**已有 OmniGibson 3.9.3 环境。** 如果只缺资产，在已激活的仿真环境中执行以下命令，使用同一个数据根目录：

.. code-block:: bash

   python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
   python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets()"
   python -c "from omnigibson.utils.asset_utils import download_2026_challenge_task_instances; download_2026_challenge_task_instances()"

两种方式都从 `官方资产仓库 <https://huggingface.co/datasets/behavior-1k/zipped-datasets/tree/main>`_ 下载。完成后，数据根目录中应有 ``omnigibson-robot-assets/``、``behavior-1k-assets/``、``2026-challenge-task-instances/`` 三个目录和数据集密钥 ``omnigibson.key``。

首次导入 Isaac Sim 时可以按提示接受 `NVIDIA 许可协议 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html#running-isaac-sim>`_。无人值守运行时，同意许可后可在启动终端设置 ``export OMNI_KIT_ACCEPT_EULA=YES``。


下载模型
----------------------------------------

下载对应模型族的检查点（任选一种方式）：

.. code-block:: bash

   # 方法 1：git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-Behavior
   git clone https://huggingface.co/RLinf/RLinf-Pi0-Behavior

   # 方法 2：huggingface-hub（国内用户可设置 HF_ENDPOINT=https://hf-mirror.com）
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-Behavior --local-dir RLinf-OpenVLAOFT-Behavior
   hf download RLinf/RLinf-Pi0-Behavior --local-dir RLinf-Pi0-Behavior

.. include:: _model_path.rst


运行
----------------------------------------

.. warning::

   将 BEHAVIOR env worker 放在从 0 开始的 GPU 上。IsaacSim 在 env worker 从更靠后的 GPU rank 启动时可能卡住。

每个配方都是 ``examples/embodiment/config/`` 下的一个 YAML 配置：

.. list-table::
   :header-rows: 1
   :widths: 34 26 40

   * - 模型 / 用途
     - 算法
     - 配置
   * - OpenVLA-OFT
     - PPO
     - ``behavior_ppo_openvlaoft.yaml``
   * - π₀
     - PPO
     - ``behavior_ppo_openpi.yaml``
   * - π₀.₅
     - PPO
     - ``behavior_ppo_openpi_pi05.yaml``

使用 ``run_embodiment.sh`` 启动一个配置：

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   bash examples/embodiment/run_embodiment.sh behavior_ppo_openvlaoft

**这个命令会：**

1. 加载 ``examples/embodiment/config/behavior_ppo_openvlaoft.yaml`` 及共享环境配置 ``examples/embodiment/config/env/behavior_r1pro.yaml``。
2. 按组件放置配置启动 actor、rollout 和 BEHAVIOR env 的 Ray worker。
3. 运行 PPO 训练，并把日志和检查点写入 ``runner.logger.log_path``。

.. admonition:: 进一步配置
   :class: note

   - BEHAVIOR 吞吐调优 → 先增加 env GPU 数量，再调 ``env.train.num_env_subprocess`` 和 ``env.train.total_num_envs``。
   - 每个 BEHAVIOR 进程可能占用约 10 GiB 显存；请按 GPU 显存调节 subprocess 数量。
   - 缓存任务实例 → 使用 ``rlinf/envs/sim/behavior/instance_generator.py`` 和 ``examples/embodiment/config/env/behavior_r1pro.yaml`` 生成。
   - 组件放置和吞吐调优 → :doc:`组件放置 <../../concepts/placement>` 与 :doc:`执行模式 <../../concepts/execution_modes>`
   - 指标定义和日志后端 → :doc:`训练指标 <../../reference/metrics>`

.. warning::

   已知问题：在当前 BEHAVIOR 设置下，OpenVLA-OFT / π₀ 的训练成功率
   （``env/success_once``）可能保持为 0。该问题会在后续版本修复。

独立评测
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

原则上，任何在 BEHAVIOR 上有非零成功率、并且已经转换为 PyTorch
格式的 ``pi05`` checkpoint 都可以使用该配置评估。我们只将
OpenPI-Comet 作为示例来源：

- https://huggingface.co/sunshk/openpi_comet/tree/main

下载后，可参考以下仓库将权重转换为 PyTorch 格式：

- https://github.com/mli0603/openpi-comet

感谢 OpenPI-Comet 作者开源模型和工具，这有助于 RLinf 中的可复现评估。

转换完成后，按如下方式更新 ``behavior_openpi_pi05_eval.yaml``：

1. 将 ``rollout.model.model_path`` 设置为转换后的模型目录。
2. 如需更长轨迹，提高 ``env.eval`` 的 ``max_episode_steps`` 与 ``max_steps_per_rollout_epoch``，例如设置为 ``4096``。

.. code-block:: yaml

   env:
     eval:
       max_episode_steps: 4096
       max_steps_per_rollout_epoch: 4096

独立评估请走 :doc:`BEHAVIOR-1K 评测指南 <../../evaluations/guides/behavior>`。该指南负责 ``OMNIGIBSON_DATA_PATH`` 设置、``behavior_openpi_pi05_eval`` 启动命令和结果解读。

--------------

**5. 评估 JAX 对齐的 Pi0.5 checkpoint**

BEHAVIOR 评估使用 ``model_type: openpi``。若权重来自 :doc:`sft_openpi` 或 OpenPI checkpoint 转换器，评估配置为：

- ``evaluations/behavior/behavior_openpi_pi05_eval.yaml``

该配置以纯评估模式运行（``runner.only_eval: True``），并消费转换后的
checkpoint，即由 OpenPI checkpoint 转换器
（``ckpt_convertor.openpi`` 的 ``openpi_pytorch_to_openpi`` /
``sft_to_openpi``）产出的权重。
将模型路径以 ``/path/to/...`` 占位符的形式直接写在配置中：

- ``rollout.model.model_path``：评估 checkpoint。

归一化统计从转换后 checkpoint 中打包的
``physical-intelligence/behavior/norm_stats.json`` 读取。

.. code:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval

.. note::

   评估时流匹配动作头使用非确定性（随机）采样噪声，因此重复运行之间，单次轨迹与
   成功次数会有轻微差异。


配置参考
----------------------------------------

BEHAVIOR 环境由 ``examples/embodiment/config/env/behavior_r1pro.yaml`` 驱动。RLinf 先加载
OmniGibson 的基础配置（``base_config_name``），再应用 ``omni_config`` 覆盖项。下表中的字段控制 reset 行为、场景
加载、仿真器频率与吞吐，大多有合理默认值，仅在自定义任务或调优性能时需要修改。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 配置项
     - 含义
   * - ``base_config_name``
     - 在 ``omni_config`` 覆盖前加载的 OmniGibson 基础配置（如 ``r1pro_behavior``）。
   * - ``omni_config.task.type`` / ``omni_config.scene.type``
     - 显式保留 ``BehaviorTask`` / ``InteractiveTraversableScene``，以确保覆盖后选中预期的上游
       OmniGibson 类。
   * - ``omni_config.task.activity_name``
     - 填任务名称，例如任务表中编号 ``0`` 对应的 ``turning_on_radio``。在固定实例模式（``instance_resample_mode: disabled``）下，``activity_instance_id`` 用于指定初始布局。
   * - ``omni_config.task.instance_resample_mode``
     - reset 时的实例切换：``disabled``\ （加载固定的 ``activity_instance_id``）、``offline``\ （启动时扫描一次 ``activity_instance_dir``，每次 reset 采样一个缓存实例——``*_template.json`` 走较重的场景重载路径，``*_template-tro_state.json`` 走较轻的原地路径）、或 ``online``\ （每次重置时重新采样场景，设置见 :doc:`评测指南 <../../evaluations/guides/behavior>`）。
   * - ``omni_config.task.activity_instance_dir``
     - 缓存实例 JSON 目录（``*_template.json`` / ``*_template-tro_state.json``），供 ``offline``
       模式以及 ``disabled`` 模式下的固定 id 加载使用。
   * - ``omni_config.task.instance_file_format``
     - 缓存实例格式：``template``\ （完整重载）或 ``tro_state``\ （仅任务相关、轻量）。RLinf 接受不含
       ``robot_poses`` 的 ``tro_state`` 文件，此时会清除过期的缓存机器人位姿元数据，reset 回退到
       任务默认机器人位姿。
   * - ``omni_config.scene.partial_scene_load``
     - 设为 ``true`` 时，只加载 2026 挑战场景中分配给该任务的房间。
   * - ``camera.head_resolution`` / ``camera.wrist_resolution``
     - 头部 / 腕部相机分辨率（默认 720×720 / 480×480，应用到 R1Pro 传感器）。
   * - ``omni_config.env.action_frequency`` / ``rendering_frequency`` / ``physics_frequency``
     - 动作 / 渲染 / 物理步进频率（常用默认 60 / 60 / 120）。越高越慢。
   * - ``omni_config.env.automatic_reset``
     - 保持 ``False``——reset 由 RLinf 训练/评估循环显式控制。
   * - ``omni_config.env.flatten_obs_space`` / ``flatten_action_space``
     - 保持 ``False`` 以保留结构化的观测 / 动作空间。
   * - ``omni_config.macro.use_gpu_dynamics``
     - ``False`` 通常更快；仅在需要粒子 / 流体时启用。
   * - ``omni_config.macro.enable_object_states``
     - 保持 ``True``——``BehaviorTask`` 依赖物体状态。
   * - ``omni_config.macro.enable_transition_rules``
     - ``True`` 启用基于转移规则的状态变化（如切割、烹饪）。
   * - ``omni_config.macro.use_numpy_controller_backend``
     - ``True`` 使用 numpy 控制器后端，在单进程 / 中等并行下通常更快。
   * - ``skip_intermediate_renders``
     - 为 ``True`` 时跳过动作块的中间渲染，返回块末观察。仿真器每步仍会读取传感器。
   * - ``num_env_subprocess``
     - 每个本地环境 worker 启动的仿真进程数，预设值为 ``1``。例如，4 个本地环境分给 2 个进程时，每个进程运行 2 个场景。同一 worker 的各流水线阶段共享这个池。由于一次物理步会推进进程内的所有场景，每个阶段至少需要一个专属仿真进程。各环境需要独立重置时，为每个环境分配一个进程。

生成缓存任务实例
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``rlinf/envs/sim/behavior/instance_generator.py`` 直接从 ``behavior_r1pro.yaml`` 生成
``*_template.json`` 与 ``*_template-tro_state.json`` 文件（它读取 ``scene_model``、
``activity_name``、``activity_definition_id``、机器人配置与房间加载设置，然后从任务的标准场景模板重新采样）。
若设置了 ``activity_instance_dir`` 则写入该目录，否则写入 ``OMNIGIBSON_DATA_PATH`` 默认的
``2026-challenge-task-instances`` 中所选场景和任务的子目录；可用 ``--output-dir`` 覆盖。

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

生成的文件名遵循
``<scene_model>_task_<activity_name>_<activity_definition_id>_<activity_instance_id>_template(.json|-tro_state.json)``，
因此 ``--start-idx`` / ``--end-idx`` 决定 ``activity_instance_id`` 范围。``tro_state`` 输出仅在任务
元数据提供时包含顶层 ``robot_poses``，否则省略该键，reset 回退到任务默认机器人位姿。BEHAVIOR-1K 上游的
``multiply_b1k_tasks.py`` 仍可使用，但推荐 RLinf 的生成器，因为它直接读取 RLinf YAML 并保留
``activity_definition_id``。


可视化与结果
----------------------------------------

启动 TensorBoard 实时观察训练：

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

最值得关注的指标是 **``env/success_once``** —— 任务成功率。每个日志指标的含义见
:doc:`训练指标 <../../reference/metrics>`。

如需保存评估视频，在配置中启用：

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval


对于 BEHAVIOR 实验，我们受到了
`Behavior-1K baselines <https://github.com/StanfordVL/b1k-baselines.git>`_ 的启发，
仅进行了少量修改。感谢作者发布开源代码。
