RPent 多模态强化学习
====================

本示例用 RLinf 训练 Qwen3.5-4B 规划器，通过 RPent 在 LIBERO 中执行六种末端控制工具。规划器接收图像和文字，生成工具调用；RLinf 负责采样与 Actor/Critic 更新。训练更新语言参数，冻结视觉编码器和 projector。

概述
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型

      完整的多模态 Qwen3.5-4B checkpoint

   .. grid-item-card:: 算法

      同步 PPO 或异步 SAO

   .. grid-item-card:: 工具

      末端执行器控制与 SAM3 分割

   .. grid-item-card:: 硬件

      部署示例使用两台八卡模型节点和一台八卡环境节点

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 工作流
     - 范围
   * - 本页：``libero_rl``
     - 标准 LIBERO 的 40 个任务；seed 1–32 用于训练，seed 0 用于评测。划分的是同一组任务的初始状态。
   * - :doc:`原版 Planner PPO <rpent>`
     - LIBERO-PRO、Pi0.5、任务记忆和文本观测。使用该页面的配置与安装说明。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 项目
     - 行为
   * - 观测
     - 初始化与动作后返回实测状态；``view_env`` 返回相机图像。
   * - 动作
     - ``move_to``、``set_gripper``、``view_env``、``segment``、``back_project`` 和 ``finish``。
   * - 奖励
     - 以环境原生成功信号为准：成功为 1，失败为 0。规划器调用 ``finish`` 不代表任务成功。
   * - 任务提示
     - 任务指令与所选工具提示词，默认使用 ``simple``。

安装
----

使用 ``Sonorous281/RLinf`` 与 ``Sonorous281/RPent`` 配套的 ``dev/rpent-rlinf-integration`` 分支。先完成 :doc:`rpent_libero_rl_setup`，准备两个 Python 环境、完整模型、SAM3 权重、模拟器资源、任务数据和 Ray 服务。旧 baseline 分支 README 指定的是另一套 RPent 版本，不适用于本部署。

安装页以三台八卡机器为例。启动前按已分配资源调整 placement 与并发；模板不保证最低显存需求。RPent 独立采集器使用另一套模型服务环境，默认基准也为 LIBERO-PRO。

准备任务数据
------------

任务生成器输出 ``train.jsonl``，并记录任务 ID、初始状态 seed、数量和哈希。每行包含任务指令 ``prompt``，以及 JSON 字符串形式的 ``answer``：

.. code-block:: json

   {"prompt": "open the bottom drawer", "answer": "{\"suite\": \"libero_goal\", \"task\": 0, \"seed\": 1}"}

``suite`` 和 ``task`` 指定模拟器任务；实际指令由生成器读取。Rollout 保留采样 token ID、behavior logprob、图像、policy 版本与原生结果。Policy loss 只覆盖模型生成的回复 token；工具观测和强制闭合 token 的 loss mask 为 0。

运行训练
--------

安装完成后，在 head 节点运行下方 SAO 示例。六个覆盖参数填写 local 配置的必填路径：

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

末尾添加 ``--cfg job --resolve`` 可检查最终配置而不启动训练。先用 Hydra 覆盖参数缩小预算，验证 loss/梯度、参数更新、权重同步与基础设施错误。短 smoke 通过不代表最长 262144-token 工作负载能放入显存。

结果与验证
----------

.. list-table::
   :header-rows: 1

   * - 证据
     - 检查内容
   * - 输出
     - ``runner.output_dir`` 下的实验目录保存训练日志与 Actor/Critic checkpoint；``tools.rpent.output_dir`` 保存 episode 请求、图像和结果。
   * - 运行正确性
     - 在部署版本上确认 loss/梯度有限、参数确实更新、权重同步正常，以及 checkpoint 可以恢复。
   * - 任务结果
     - ``rpent/success_rate`` 反映训练 rollout 结果。分别报告原生成功数／完成数与基础设施失败；学习收益需用同条件独立评测衡量。
   * - 验证范围
     - CPU 组件测试覆盖轨迹、mask 和 packing。真实 SGLang/Megatron 数值一致性、完整更新与恢复仍需在目标环境验证。本分支尚无证明学习收益的独立评测。

同时查看 ``rpent/turns`` 和 ``rpent/infra_retry_rate``。模板设置 ``agentloop.max_infra_retry_rate=1.0``，允许已恢复的重试继续训练；单局重试耗尽仍报错。历史强制闭合数值探针未达到诊断容差，不能作为部署验收证据。

恢复与评测
----------

恢复训练
~~~~~~~~

使用 ``runner.resume_dir=/path/to/checkpoints/global_step_N`` 从完整 SAO checkpoint 恢复两套模型/优化器与 buffered/pending rollout 状态。只选择包含 actor/critic 状态及最后写入的 ``data/data.pt`` 标记的完整 checkpoint。未完成任务的 prompt 会重新入队，模拟器不会从中断位置继续运行；恢复时保持数据集、算法、world size 和引用图像兼容。同步 PPO checkpoint 不包含 SAO 所需的队列状态。

独立评测
~~~~~~~~

在另行分配的资源上运行下方命令。评测默认使用一台八卡模型节点和一台独立环境节点，须按实际资源调整 placement。

模板的任务、采样、episode 预算与历史窗口均与训练一致；训练若覆盖了这些参数，评测也须同步。RPent 独立 batch 工具用于模型服务与数据采集，原有 ``eval_qwen3.5_4b`` 则对应原版 LIBERO 工作流。

导出时 ``--vlm`` 必须指向训练初始化所用的完整模型，以保留冻结的视觉权重和 processor。从 SFT 开始训练时相应替换路径。导出目录必须是新目录；HF 导出只用于推理，不能替代包含优化器和队列的恢复 checkpoint。

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
     tools.rpent.libero_config_path="$LIBERO_CONFIG_PATH"

评测初始模型时，只改变模型与输出路径，复用同一份 manifest、任务/初态 ID、prompt、采样参数、轮数/动作预算及压缩设置。分别报告原生成功数/总数、轮数耗尽和基础设施失败。训练完成队列的成功率不是独立评测；固定 40 条 case 也不足以确认很小的学习增益。

选择训练配置
------------

.. list-table::
   :header-rows: 1

   * - 配置名（不含 ``.yaml``）
     - 用途
   * - ``train_qwen3.5_4b_libero_rl_ppo``
     - 同步 PPO
   * - ``train_qwen3.5_4b_libero_rl_sao``
     - 按 episode 完成顺序调度的 SAO
   * - ``train_qwen3.5_4b_libero_rl_sao_3node``
     - 三节点 SAO placement
   * - ``train_qwen3.5_4b_libero_rl_sao_local``
     - 需填写本地路径的三节点 SAO
   * - ``eval_qwen3.5_4b_libero_rl``
     - 独立多模态评测

启动器默认仍为原版 ``libero``，必须显式传入 ``libero_rl`` 配置名。PPO → SAO → 三节点 → local 配置依次继承公共设置。

.. list-table::
   :header-rows: 1

   * - 项目
     - 模板值
   * - 模型布局
     - 节点 0–1：Actor、Critic 与 rollout 分阶段共享 16 卡。Learner 使用 TP2/CP1/DP8、sequence parallel；rollout 为 16 个 TP1 副本。
   * - 环境布局
     - 节点 2：八个 worker，每个 32 个 episode 槽位；GPU 0 运行 SAM3。每个 worker 最多同时初始化八个环境，限制冷启动压力。
   * - 调度
     - Batch 128，在途 episode 和 buffer 容量均为 256。SAO 按完成顺序取样；共享 GPU 上更新模型时暂停生成，环境动作可继续。
   * - 训练
     - 10 个数据 epoch，每五次更新保存。1,280 条任务对应 100 次 learner 更新。Actor LR 1e-6、Critic LR 5e-6；每次 SAO 更新含一个 actor minibatch、两个 critic epoch。
   * - Episode
     - 100 turns、10,000 env steps、四小时预算；context 262144，output 32768。
   * - 采样
     - 开启 thinking；temperature 1、top-p 0.95、top-k 20、presence penalty 1.5、repetition penalty 1.0。
   * - 历史
     - 消息窗口 80→40，达到 context 的 70% 时压缩，不额外生成摘要。
   * - 推理容量
     - 每副本显存比例 0.7、最多 128 个请求是准入设置，不代表实测吞吐或显存容量。

SAO 使用 DIS actor loss（clip 边界 0.3/5.0）、``sao_gae``，alpha 为 1.5、SAO critic loss 和采样时记录的 behavior logprob。:doc:`完全异步 Search-R1 <searchr1/async>` 使用独立推理卡，更新调度不同。

更新耗时较长时，可设置 ``rollout.drain_before_training=true``，先完成在途 episode 再暂停生成。Buffer 容量必须覆盖在途上限；该选项不改变任务超时预算。

可选运行设置
------------

训练和评测默认设置 ``rollout.sglang.disable_overlap_schedule=true``，供 presence penalty 和可选 Thinking 限制读取当前 CPU token 历史。修改前需在目标 SGLang 版本验证采样 logprob 与强制 token 对齐。

``agentloop.chat_template_kwargs.enable_thinking=false`` 可关闭 thinking；``agentloop.prompt=simple_brief`` 则用提示词引导每次决策约 256 个推理 token，不强制改变解码。切换前固定模型和任务 seed，从新 episode 对比原生成功率与成本；不能删除已有轨迹中的 thinking token 或 behavior logprob。

若要限制 chat template 已开启的首段 Thinking，设置 ``agentloop.thinking_token_budget=512`` 和 ``rollout.sglang.enable_custom_logit_processor=true``，并保持 ``enable_thinking=true`` 和 ``rollout.sglang.disable_overlap_schedule=true``。启用预算时，配置校验会拒绝重叠调度，避免 CPU token 历史落后于实际解码。预算默认是 ``null``。引擎记录实际强制插入的闭合 Token，该 Token 的训练 mask 为零；自然生成的闭合 Token 仍参与训练。此选项不限制后续重新开启的 Thinking 段。组装轨迹时必须保持返回的 Token ID、behavior logprob、强制标记和权重版本的对应关系。

设置 ``actor.model.lm_head_chunk_size=256``，按块计算参与 loss 的回复 token 的输出层结果；默认 ``0`` 不分块。反向传播时会重新计算这些结果，支持 PP1/CP1 的 MBridge Qwen3.5，包括 TP 和 sequence parallel。使用安装页记录的 Megatron 版本。Entropy 只作为 detached 诊断指标，不支持 entropy loss bonus；该模式也不支持 FP8、MTP、延迟 embedding 梯度、muP 和 autocast。这些约束界定受支持的执行方式，不代表已有吞吐或学习收益。

这些配置默认使用进程内 SGLang 后端。在相同启动命令中添加 ``+rollout.sglang.backend_type=server``，可改用主干的 HTTP server 后端。两个后端都传递图像、Thinking 预算处理器和已安装的权重版本。server 作为本地子进程运行，仅绑定回环地址；Actor 权重仍由 scheduler 直接接收。

工具结果和图片保留在后续输入中。Policy loss 只作用于 policy 生成的 assistant token，user/tool/history token 不参与监督。逐 token 前缀兼容的连续 turn 合并，压缩或历史改写后分段。Actor 和 critic 使用相同的 response 位置映射；共享轨迹 packing 在任意一个计分 token 重叠时都会保留独立行，避免合并覆盖学习目标；这也适用于纯文本 SearchR1。回归测试比较 packing 前后的 loss 和梯度。response-only logits 要求 CP1。

启用 ``tools.rpent.store_images_on_disk=true`` 时，所有 rollout 和 learner 进程必须能读取 **相同绝对路径下的图像** 。``tools.rpent.output_dir`` 需要在模型节点与环境节点间共享，并在轨迹或保存的 checkpoint 仍引用它时保留；不同节点上同名的本地目录不够。没有共享图像存储时保留默认 ``false``，让轨迹携带图像字节，但内存成本更高。不要清理可恢复 checkpoint 所需的 Episode 产物。

为什么需要保留多轮图像的 Token
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

合并 Qwen3.5 图像 turn 时，在每台模型节点启动 Ray **之前** 启用保留 token 的 processor：

.. code-block:: bash

   export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rlinf.hybrid_engines.sglang.multimodal

adaptive 训练和评测模板也通过 ``rollout.env_vars`` 传入此变量，使 worker 启动不只依赖终端环境。

该配置使用 SGLang 的外部 processor 注册入口。图像请求携带已展开的 prompt IDs 和原始图片；适配器核对图片数量及各图 pad 段与处理后 grid 的对应关系，保留原始文本和动作 IDs，再由 SGLang 根据这一序列计算 offset 和 M-RoPE。对已采样 token 解码后重新编码，即使文本相同，也可能改变 token 边界。字符串请求保持上游行为；该图像适配器会拒绝携带视频、音频或预计算图像输入的 token-ID 请求。

使用包含 config、tokenizer 和 processor 文件的本地模型目录运行回归：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES='' SGLANG_USE_CUDA_IPC_TRANSPORT=0 \
     RLINF_TEST_QWEN_VL_MODEL=/path/to/Qwen3.5-4B \
     PYTHONPATH=.:tests/unit_tests \
     python -m pytest tests/unit_tests/test_sglang.py -k qwen_image_processor

这些 CPU 检查覆盖输入 IDs、图像 tensor、offset，以及 HF 与 SGLang 的 M-RoPE 一致性。部署新版本前，还需要验证真实引擎的生成和 logprob、完整 learner 更新及 checkpoint 恢复。

.. toctree::
   :hidden:

   rpent_libero_rl_setup
