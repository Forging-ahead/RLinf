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

      50 个 BEHAVIOR-1K 任务

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 支持光追的 GPU

| **你将完成：** 安装 IsaacSim 依赖 → 下载资产与基座模型 → 运行 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · Isaac Sim 5.1 与 BEHAVIOR-1K 资产（>30 GB）· 基座检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 说明
   * - 任务
     - 来自 BEHAVIOR-1K 的 50 个家居操作任务（通过 ``omni_config.task.activity_name`` 和对应场景选择）。
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

BEHAVIOR 仿真端使用官方 ``v3.9.3-post2`` SDK，包含 OmniGibson 3.9.3、Isaac Sim 5.1 和 PyTorch 2.7.0。在 Linux 上安装前，请核对 `Isaac Sim 5.1 硬件与驱动要求 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_。相机渲染需要支持光线追踪的 GPU。

.. include:: _setup_common.rst

仿真端和模型需要不同版本的 LeRobot 和 Hugging Face Hub。在 RLinf 仓库根目录运行以下命令，安装器会分别将模型和仿真端安装到 ``.venv`` 和 ``.venv-behavior``。激活模型环境后，``BEHAVIOR_ENV_PYTHON`` 会指向仿真端解释器。

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env behavior --no-flash-attn
   source .venv/bin/activate

使用 OpenVLA-OFT 时，将模型参数改为 ``--model openvla-oft``，并省略 ``--no-flash-attn`` 以安装对应的注意力后端。两个环境使用相同的 Python 3.11 补丁版本和 Ray 版本，供 RLinf 跨解释器启动 worker。解释器路径应避免空格和 shell 特殊字符。Docker 构建目标 ``embodied-behavior-openpi`` 和 ``embodied-behavior-openvlaoft`` 使用相同的安装入口，运行时需要挂载权重和仿真资产。

.. code-block:: bash

   docker build -f docker/Dockerfile --build-arg BUILD_TARGET=embodied-behavior-openpi -t rlinf:behavior-openpi .

如果已有上述固定版本的 SDK，请先用 SDK 的解释器创建两个空环境，再运行上面的安装命令：

.. code-block:: bash

   export BEHAVIOR_PATH=/path/to/BEHAVIOR-1K
   uv venv --python /path/to/sdk/bin/python --system-site-packages .venv-behavior
   uv venv --python /path/to/sdk/bin/python .venv

只使用仿真端的应用可以运行 ``bash requirements/install.sh embodied --env behavior --venv .venv-behavior --no-flash-attn``。安装器使用单独的 SDK 源码副本，并将 LeRobot 固定在支持 Python 3.11 的 ``436812bd8ee39b768c645c248c91f1330834e687``。仅在系统依赖已经安装时使用 ``--no-root``。

运行 ``"$BEHAVIOR_ENV_PYTHON" -m pip check`` 可以检查仿真端的全部依赖，包括从已有 SDK 继承的软件包。使用上述固定版本时，该命令会报告四项元数据冲突，退出码为 1。Isaac Sim 固定的 ``packaging``、``websockets`` 和 ``wrapt`` 版本低于其他依赖的要求；OmniGibson 要求 Pillow 11.0.x，而 Isaac Sim 使用 11.3.0。

下载资产
----------------------------------------

使用仿真端解释器下载 SDK 配套的机器人资产、BEHAVIOR-1K 资产和 2026 年的任务实例。数据根目录还需包含数据集密钥。通过 pip 安装的 Isaac Sim 会自行定位运行时；单独安装匹配版本的 Isaac Sim 时才需要设置 ``ISAAC_PATH``。

.. code-block:: bash

   export OMNI_KIT_ACCEPT_EULA=YES
   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   "$BEHAVIOR_ENV_PYTHON" -m omnigibson.utils.asset_utils \
     --download_omnigibson_robot_assets --download_behavior_1k_assets \
     --download_2026_challenge_task_instances --accept_license

自托管 CI runner 默认从 ``/workspace/dataset/behavior-datasets`` 读取这些资产。也可以通过仓库变量 ``BEHAVIOR_DATA_PATH`` 指定实际的绝对路径。

下载模型
----------------------------------------

独立评测使用 PyTorch 格式的 π₀.₅ 权重及对应的 BEHAVIOR 归一化统计量。可以从 `RLinf 的 Hugging Face 仓库 <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_ 下载公开权重。

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 \
     --local-dir /path/to/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32

使用其他训练配置时，可按模型选择 ``RLinf/RLinf-OpenVLAOFT-Behavior`` 或 ``RLinf/RLinf-Pi0-Behavior``。

.. include:: _model_path.rst

运行
----------------------------------------

在训练配置中填入 actor 和 rollout 的模型路径，激活模型环境，并设置 ``OMNIGIBSON_DATA_PATH``。以下命令使用 π₀.₅ 进行 PPO 训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh behavior_ppo_openpi_pi05

另外两个训练配置是 ``behavior_ppo_openvlaoft`` 和 ``behavior_ppo_openpi``。env worker 使用仿真端解释器推进 OmniGibson，actor 和 rollout worker 使用模型环境。独立评测及 64 步收音机测试见 :doc:`BEHAVIOR 评测指南 <../../evaluations/guides/behavior>`。

之前的 BEHAVIOR 训练中，OpenVLA-OFT 和 π₀ 的 ``env/success_once`` 可能一直为 0。这些结果来自升级到 3.9.3 之前的实验。

配置参考
----------------------------------------

训练和评测共用 ``examples/embodiment/config/env/behavior_r1pro.yaml``。切换任务时，同时指定 ``omni_config.task.activity_name`` 和对应的 ``omni_config.scene.scene_model``。2026 年训练集的实例编号从 0 开始，公开测试集从 301 开始。显式设置 ``activity_instance_dir`` 时，优先从该目录读取。任务实例和场景资产应来自同一版本 SDK。

默认每个环境独占一个仿真进程，局部 reset 不会推进其他场景。整批环境同时 reset 时，可用正整数 ``num_env_subprocess`` 在进程内批处理。启用 ``auto_reset: True`` 时，每个环境需要独立进程；启用 offload 时需要显式 reset。``torch_num_threads`` 控制 PyTorch 的 CPU 线程数。``skip_intermediate_renders`` 可以省略动作块的中间渲染，同时保留每一步的物理推进、奖励和终止判断。

需要生成本地任务实例时，在仿真环境运行原有生成器，并按配置所用文件格式选择 ``tro_state`` 或 ``template``：

.. code-block:: bash

   "$BEHAVIOR_ENV_PYTHON" -m rlinf.envs.sim.behavior.instance_generator \
     --config examples/embodiment/config/env/behavior_r1pro.yaml \
     --output-dir /path/to/instances --output-format tro_state --start-idx 1 --end-idx 10

如果应用自己管理仿真进程，可以在独立进程的主线程直接从 RLinf 的 BEHAVIOR 包导入 ``BehaviorSimulator``。设置好数据根目录后运行以下示例：

.. code-block:: python

   from pathlib import Path
   from hydra import compose, initialize_config_dir
   from rlinf.envs.sim.behavior import BehaviorSimulator

   with initialize_config_dir(
       version_base=None, config_dir=str(Path('examples/embodiment/config/env').resolve())
   ):
       cfg = compose(config_name="behavior_r1pro")
   sim = BehaviorSimulator(cfg, num_envs=1, seed=0)
   obs, info = sim.reset()
   print(sim.get_metrics(), flush=True)
   sim.close()

``reset`` 返回每个环境的观测和信息。``step(actions)`` 接受形状为 ``[num_envs, 23]`` 的有限数值动作，返回观测、奖励、终止标志、截断标志和信息。``observe(render=True)`` 会刷新图像，物理状态保持在当前步。观测、状态、相机参数和指标均返回副本；``get_robot()`` 返回借用的机器人对象，可用至下一次 reset 或关闭。先保存结果，再将 ``close()`` 作为进程的最后一步，因为 Isaac 关闭时可能终止进程。


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
