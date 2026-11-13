RPent 强化学习
========================

本示例使用 RPent 执行 LIBERO 任务，并通过 RLinf 根据环境反馈训练多模态 Qwen3.5 policy。

选择工作流
------------------------

启动器默认仍为原版 ``libero``。``train_qwen3.5_4b_ppo`` 和 ``eval_qwen3.5_4b`` 保留原来的 Pi0.5、memory、纯语言模型配置；相关用法见 :doc:`原版 LIBERO 示例 </rst_source/examples/agentic/rpent>`。

独立的 ``libero_rl`` 使用 **完整多模态** Qwen checkpoint、adaptive 观测，以及六个工具：``move_to``、``set_gripper``、``view_env``、``segment``、``back_project``、``finish``。SAM3 提供分割，运动工具直接在 LIBERO 中执行。必须显式选择以下配置：

.. list-table::
   :header-rows: 1

   * - 配置名，不含 ``.yaml``
     - 用途
   * - ``train_qwen3.5_4b_libero_rl_ppo``
     - 同步 PPO 模板
   * - ``train_qwen3.5_4b_libero_rl_sao``
     - 异步 SAO 模板
   * - ``train_qwen3.5_4b_libero_rl_sao_local``
     - 需自行填写路径的启动配置
   * - ``train_qwen3.5_4b_libero_rl_sao_3node``
     - 两台模型节点共享 16 卡运行 SAO，另加一台环境节点
   * - ``eval_qwen3.5_4b_libero_rl``
     - 多模态评测模板

轨迹数据格式
------------------------

任务 JSONL 的每行包含 ``prompt`` 和 ``answer``；``answer`` 是 JSON 字符串，包含 ``suite``、``task``、``seed``。任务生成器同时写出 task/初始状态清单。每个 episode 按顺序记录模型请求、采样 token ID、behavior logprob、工具结果、图像和原生终局 reward：成功为 1，失败为 0。

RLinf 仅在 token 前缀一致时合并多轮回复。训练行保留 prompt/response 长度、图像输入、逐 token policy 版本和 response mask。工具观测、强制闭合 token 的 mask 为 0，不参与 policy loss 或 Critic target。Actor 和 Critic 推理携带相同的 rollout ID；即使文本完全相同、图像不同，也不会按文本内容匹配样本。

快速开始
----------------

首次使用先完成 :doc:`rpent_libero_rl_setup`。已有集群可直接激活 RLinf 环境，填写 ``train_qwen3.5_4b_libero_rl_sao_local.yaml`` 中的必填路径，并核对下方资源布局。

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_sao_local

评测时填写 ``eval_qwen3.5_4b_libero_rl.yaml`` 中的模型、数据和环境路径，然后运行：

.. code-block:: bash

   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_libero_rl

评测训练 Checkpoint 前需先导出完整多模态模型，命令与恢复说明见下方。

指标与验证边界
------------------------

主要查看 ``rpent/success_rate``、``rpent/turns``、``rpent/infra_retry_rate``，以及 Actor/Critic 的 loss 和梯度。成功率按原生成功数／完成数统计，基础设施故障单独报告。模板用 ``agentloop.max_infra_retry_rate=1.0`` 允许已恢复的重试继续训练；单局重试耗尽仍报错。同步 PPO 和 SAO 通过同一个任务 hook 应用配置的重试规则。

这些模板说明支持的配置；现有证据尚不足以证明学习收益：

.. list-table::
   :header-rows: 1

   * - 证据
     - 覆盖范围与结果
   * - CPU 回归
     - 组件测试覆盖轨迹身份、action mask、packet 元数据、packing 前后的 loss/梯度、陈旧数据准入及工具/上下文边界。
   * - 部署正确性
     - 仍需针对实际部署版本和硬件验证真实 SGLang 请求、learner 更新及 checkpoint 恢复。可选 Thinking 闭合和 LM Head 分块默认关闭；历史强制闭合数值探针未达到诊断容差。
   * - 学习结果
     - 本分支尚无证明涨点的已完成独立评测。历史训练完成队列曲线不能替代初始模型与 checkpoint 的同条件评测。

使用 16 张 GPU 训练 SAO
--------------------------

多模态训练使用四份依次继承的配置：``train_qwen3.5_4b_libero_rl_ppo`` → ``train_qwen3.5_4b_libero_rl_sao`` → ``train_qwen3.5_4b_libero_rl_sao_3node`` → ``train_qwen3.5_4b_libero_rl_sao_local``。第一份继承原版 ``train_qwen3.5_4b_ppo``，原版 LIBERO 集成仍可使用该基础配置。任务、采样和模型参数放在多模态 PPO 配置中；SAO 只覆盖更新规则，三节点配置只覆盖 placement，local 配置只填写部署路径。

根据自己的安装位置填写 ``examples/agent/rpent/config/train_qwen3.5_4b_libero_rl_sao_local.yaml``，或使用下方启动命令的路径覆盖项；未填写的 ``???`` 会报错，不会回退到原实验路径。需填写完整多模态模型、训练 JSONL、RPent 解释器、LIBERO 配置、episode 目录和训练输出目录。父配置将这些路径标记为必填，不提供特定机器的默认路径。``rollout.model.model_path`` 同时设置 actor/critic 的 tokenizer 和 HF 导入路径。

.. list-table::
   :header-rows: 1

   * - 资源或参数
     - 配置
   * - 节点 0–1，GPU 0–15
     - Rollout、critic、actor 分阶段共享全部 16 卡，阶段间 offload 权重、梯度和优化器
   * - Learner
     - TP2、CP1、DP8；开启 sequence parallel；BF16；activation recomputation；response-only logits
   * - Rollout
     - 16 个 TP1 SGLang 副本；``replica_affinity=false``；每副本 ``max_running_requests=128``；显存比例 0.7
   * - 节点 2，GPU 0–7
     - 八个 agent worker，每个 32 个 episode 槽位；GPU 0 上一个 SAM3 服务
   * - 完成顺序调度
     - Batch 128；``max_in_flight_rollouts=null`` 和 ``rollout_buffer_capacity=null`` 分别解析为 256
   * - 训练预算
     - 10 个数据 epoch，无额外 step 上限；1,280 条任务对应 100 次 learner 更新；每五次保存
   * - 优化
     - 每次 learner 更新使用一个 actor minibatch、两个 critic update epoch；microbatch 1；actor LR ``1e-6``、critic LR ``5e-6``；训练语言参数，冻结视觉编码器/projector
   * - Episode
     - Simple prompt、adaptive 观测、thinking、100 turns、10,000 env steps、四小时预算
   * - 生成
     - Context 262144，output 32768；temperature 1、top-p 0.95、top-k 20、presence penalty 1.5、repetition penalty 1.0
   * - Context management
     - 原生消息窗口 80→40，触发比例 0.7；不额外清除工具结果或生成摘要

训练和评测模板均设置 ``rollout.sglang.disable_overlap_schedule=true``，让 presence penalty 和可选的 Thinking 限制使用当前 CPU token 历史。修改此项前，须在所用 SGLang 版本上验证采样 logprob 和强制 token 对齐。

``agentloop.chat_template_kwargs.enable_thinking`` 支持 ``true`` 和 ``false``，本示例默认为 ``true``。改变训练模式前，应固定 checkpoint 和任务 seed，对比原生成功率与生成成本。对照必须从新的 episode 开始，不能删除已有轨迹中的 thinking token 或行为 logprob。

``agentloop.prompt=simple_brief`` 选择 RPent 的可选简短推理提示词，目标是每次决策约 256 个推理 token，不强制插入解码 token。RPent 版本需支持此变体。保留历史 thinking，并比较原生成功率与实际成本后再采用。

可选的 Thinking 与投影限制
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

若要限制 chat template 已开启的首段 Thinking，设置 ``agentloop.thinking_token_budget=512`` 和 ``rollout.sglang.enable_custom_logit_processor=true``，并保持 ``enable_thinking=true`` 和 ``rollout.sglang.disable_overlap_schedule=true``。启用预算时，配置校验会拒绝重叠调度，避免 CPU token 历史落后于实际解码。预算默认是 ``null``。引擎记录实际强制插入的闭合 Token，该 Token 的训练 mask 为零；自然生成的闭合 Token 仍参与训练。此选项不限制后续重新开启的 Thinking 段。组装轨迹时必须保持返回的 Token ID、behavior logprob、强制标记和权重版本的对应关系。

若要启用完整投影分块，设置 ``actor.model.lm_head_chunk_size=256``；默认 ``0`` 使用常规投影。分块路径按计分回复位置分批投影，并在反向时重算，支持 PP1/CP1 的 MBridge Qwen3.5，包括 TP 和 sequence parallel。使用安装页记录的 Megatron 版本；此路径会临时替换该模型的 postprocess。Entropy 只作为 detached 诊断指标，不支持 entropy loss bonus；该模式也不支持 FP8、MTP、延迟 embedding 梯度、muP 和 autocast。这些约束界定受支持的执行方式，不代表已有吞吐或学习收益。

这些配置默认使用进程内 SGLang 后端。在相同启动命令中添加 ``+rollout.sglang.backend_type=server``，可改用主干的 HTTP server 后端。两个后端都传递图像、Thinking 预算处理器和已安装的权重版本。server 作为本地子进程运行，仅绑定回环地址；Actor 权重仍由 scheduler 直接接收。

这里的 SAO 使用 Partial rollout 和 Critic，训练与推理共用 GPU。已完成轨迹保存在缓冲区中，输入游标随 checkpoint 保存。完全异步 Search-R1 使用独立推理 GPU 和主干的分组缓冲区，不能直接替代这里的 Critic 更新顺序。对应示例见 :doc:`searchr1/async`。

SAO 允许最多 256 条在途 episode，按完成顺序收集下一批 128 条轨迹，不再等待某个固定同步 batch 的全部轨迹。Episode 槽位包含等待模型与执行工具的时间；八个 worker 各 32 个槽位，足以同时展开全部 256 条在途轨迹。``max_env_initializations_per_worker=8`` 将八个 worker 的同时冷启动数量限制为 64；保存首次环境状态后立即释放冷启动槽位，不等 episode 结束，因此预热后仍可保持 256 个活跃环境。这只约束 CPU、import 和 CUDA context 初始化压力，不降低 rollout 容量。Buffer 保存已完成轨迹；额外排队的 prompt 不会提高在途上限。每个 episode 通常只有一个未完成模型请求，因此每副本 128 是准入上限，并不表示每张 GPU 始终都有 128 个活跃请求。

共享 GPU 仍需切换阶段：暂停并收回模型请求、卸载 rollout 显存、依次训练 critic 和 actor、同步权重，再恢复等待中的请求。训练时环境动作可以继续，但模型生成暂停。SAO 的学习规则也有变化：DIS actor loss、0.3/5.0 clip 边界、alpha 为 1.5 的 ``sao_gae`` 和 SAO critic 目标；它直接使用 rollout 保存的 behavior logprob，不重算旧 policy 的 logprob。任务、模型、prompt、采样和压缩参数均继承多模态 PPO 配置。

工具结果和图片保留在后续输入中。Policy loss 只作用于 policy 生成的 assistant token，user/tool/history token 不参与监督。逐 token 前缀兼容的连续 turn 合并，压缩或历史改写后分段。Actor 和 critic 使用相同的 response 位置映射；共享轨迹 packing 在任意一个计分 token 重叠时都会保留独立行，避免合并覆盖学习目标；这也适用于纯文本 SearchR1。回归测试比较 packing 前后的 loss 和梯度。response-only logits 要求 CP1。

启用 ``tools.rpent.store_images_on_disk=true`` 时，所有 rollout 和 learner 进程必须能读取 **相同绝对路径下的图像** 。``tools.rpent.output_dir`` 需要在模型节点与环境节点间共享，并在轨迹或保存的 checkpoint 仍引用它时保留；不同节点上同名的本地目录不够。没有共享图像存储时保留默认 ``false``，让轨迹携带图像字节，但内存成本更高。不要清理可恢复 checkpoint 所需的 Episode 产物。

为什么需要保留多轮图像的 Token
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

启动、检查与恢复
------------------------

填好 local 配置后，在 head 节点执行：

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

Hydra 覆盖参数放在配置名之后，例如 ``rollout.model.model_path="$STACK_ROOT/models/Qwen3.5-4B-SFT"``。末尾加 ``--cfg job --resolve`` 可以只检查最终配置而不启动训练。使用 ``runner.resume_dir=/path/to/checkpoints/global_step_N`` 从完整 SAO checkpoint 恢复两套模型/优化器与 buffered/pending rollout 状态。只选择包含 actor/critic 状态及最后写入的 ``data/data.pt`` 标记的完整 checkpoint。Pending prompt 会重新入队，并非持久化到一半的模拟器状态；恢复时保持数据集、算法、world size 和引用图像兼容。同步 PPO checkpoint 不包含 SAO 所需的队列状态。Actor/critic checkpoint 和日志位于 ``runner.output_dir`` 下的实验目录；episode 请求、工具图片和结果位于 ``tools.rpent.output_dir``。

使用 Hydra 覆盖参数运行独立短 smoke，无须单独维护 smoke YAML。检查 loss/梯度是否有限、参数是否实际更新、模型副本活动、任务成功率及基础设施错误。短 smoke 通过证明连通和优化器执行，不代表最长 262144-token 工作负载一定能放入显存。遇到显存限制先保留证据并诊断，再决定切分或预算调整。

独立评测使用下方命令，在另行分配的可用资源上运行。模板的任务、采样、episode 预算和历史窗口与训练一致，只需填写部署和数据路径。如果训练覆盖了这些参数，评测也须同步调整。默认 placement 是一台八卡模型节点和独立环境节点，请按实际资源修改。RPent 的独立 batch 工具用于模型服务与数据采集；原有 ``eval_qwen3.5_4b`` 对应原始 LIBERO 工作流。

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

原版不传配置名的启动行为及原版 LIBERO YAML 继续保持不变。


.. toctree::
   :hidden:

   rpent_libero_rl_setup
