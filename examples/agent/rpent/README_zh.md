<!-- sphinx-body -->

# RPent 强化学习

本示例使用 RPent 执行 LIBERO 任务，并通过 RLinf 根据环境反馈训练多模态 Qwen3.5 policy。

## 选择工作流

启动器默认仍为原版 `libero`。`train_qwen3.5_4b_ppo` 和 `eval_qwen3.5_4b` 保留原来的 Pi0.5、memory、纯语言模型配置；相关用法见{doc}`原版 LIBERO 示例 </rst_source/examples/agentic/rpent>`。

独立的 `libero_rl` 使用**完整多模态** Qwen checkpoint、adaptive 观测，以及六个工具：`move_to`、`set_gripper`、`view_env`、`segment`、`back_project`、`finish`。它需要 SAM3，无须 Pi0.5 或 memory 服务。必须显式选择以下配置：

| 配置名，不含 `.yaml` | 用途 |
|---|---|
| `train_qwen3.5_4b_libero_rl_ppo` | 同步 PPO 模板 |
| `train_qwen3.5_4b_libero_rl_sao` | 异步 SAO 模板 |
| `train_qwen3.5_4b_libero_rl_sao_local` | 需自行填写路径的启动配置 |
| `train_qwen3.5_4b_libero_rl_sao_3node` | 两台模型节点共享 16 卡运行 SAO，另加一台环境节点 |
| `eval_qwen3.5_4b_libero_rl` | 多模态评测模板 |

## 开发基线与实验边界

本分支基于 zhexuan 的 `rpent-ppo` commit `f351c57e62818b18315ed4b2194312a44cafb697`。新增内容按职责拆成带 sign-off 的提交：图像 Token 保真及延迟加载、按目标交换 rollout 行、分块重算 logprob、rollout 准入背压与生产顺序恢复、端口范围分配，以及请求和评测诊断。用 `git log --reverse f351c57e..HEAD` 查看改动归属与依赖顺序。

历史 Batch64/Env128 实验还使用了启动覆盖项和部署代码。这份可移植模板默认 Batch128/Env256、一个 SAM3 服务、100 轮，Thinking 只受总回复预算约束。本分支提供可选的 Thinking 强制闭合和完整 LM Head 分块，默认均关闭；不包含 H/GD 远程传输、四 SAM 布局、CP3 专用恢复包装或维护脚本。比较结果时应保留各实验的配置和源码清单，不能仅凭此 YAML 复现历史实验。

LIBERO 训练模板设置 `agentloop.max_infra_retry_rate=1.0`：基础设施故障经重试恢复后，完整且有效的批次可继续更新。重试指标和单局重试次数限制仍保留；重试耗尽或原生结果无效仍会失败。其他模板保留各自的重试策略。

下方软件版本记录历史运行兼容性。接收方仍需在自己的机器验证干净安装、SGLang 图像请求与 logprob、完整 learner update 和 checkpoint 恢复；CPU 测试不能证明 GPU 容量或学习效果。

这是开发交接版本，尚未完成全部 infra 与学习效果验收。同步多模态 PPO 请使用 Actor/Critic 共卡布局：独立推理路径仍按 token ID 匹配结果，相同 token、不同图片的样本可能串配。跨后端数值检查仍未结束，一组人工强制闭合探针超出诊断容差。可选 Thinking 强制闭合和 LM Head 分块默认保持关闭。

## 安装代码与环境

使用 Linux NVIDIA 主机，准备兼容驱动、CUDA 12.8 toolkit、EGL/OpenGL 库、Git 和 `uv`，各节点保持相同路径。下方已发布分支包含配套源码，无需额外 patch。安装前记录 RLinf commit，并保留固定的 RPent commit。

```bash
export STACK_ROOT="$HOME/rpent-stack"
mkdir -p "$STACK_ROOT/repos" "$STACK_ROOT/envs" "$STACK_ROOT/models" "$STACK_ROOT/data"
git clone --branch fix/rpent-infra-cleanup https://github.com/Sonorous281/RLinf.git "$STACK_ROOT/repos/RLinf"
git clone --branch fix/rpent-infra-cleanup https://github.com/Sonorous281/RPent.git "$STACK_ROOT/repos/RPent"
git -C "$STACK_ROOT/repos/RPent" checkout --detach dce07b9717cab06eb32d0b80c91c0117c13a57f1
git -C "$STACK_ROOT/repos/RLinf" rev-parse HEAD
git -C "$STACK_ROOT/repos/RPent" rev-parse HEAD
git clone https://github.com/NVIDIA/Megatron-LM.git "$STACK_ROOT/repos/Megatron-LM"
git -C "$STACK_ROOT/repos/Megatron-LM" checkout dc0ee41ada177d4c74ba46bb501bba5967264815
export MEGATRON_PATH="$STACK_ROOT/repos/Megatron-LM"
cd "$STACK_ROOT/repos/RLinf"
UV_TORCH_BACKEND=cu128 UNINSTALL_FA4=1 bash requirements/install.sh agentic \
  --venv "$STACK_ROOT/envs/rlinf" --python 3.12.12 --torch 2.11.0 \
  --engine sglang --sglang 0.5.12.post1 --transformers 5.6.1 \
  --use-mirror --no-root --install-rlinf
```

安装器复用已有的 `MEGATRON_PATH`。它在解析依赖时会临时修改依赖元数据，因此请使用空闲 checkout 和新环境。`--no-root` 要求原生系统依赖已准备好；`UNINSTALL_FA4=1` 选择经过验证的 FA2 路径。其他平台参数见 `bash requirements/install.sh --help`。

从固定的 RPent checkout 安装独立 simulator/SAM3 环境。它的 requirements 引用了移动中的 RLinf 分支，因此下方 override 将依赖指向交付 checkout。启动前核对输出的模块路径：

```bash
cd "$STACK_ROOT/repos/RPent"
uv venv --python 3.12 "$STACK_ROOT/envs/rpent"
source "$STACK_ROOT/envs/rpent/bin/activate"
python - <<'PYTHON' > "$STACK_ROOT/envs/rpent-source-overrides.txt"
import os
from pathlib import Path
print("rlinf @ " + (Path(os.environ["STACK_ROOT"]) / "repos/RLinf").as_uri())
PYTHON
uv pip install --prerelease=allow --torch-backend cu128 \
  -r requirements/libero-rl.txt --override "$STACK_ROOT/envs/rpent-source-overrides.txt"
python -c 'import rlinf, robots.libero_rl; print(rlinf.__file__, robots.libero_rl.__file__)'
```

以下记录已验证的兼容组合，不保证未来任意一次依赖解析都会得到完全相同的环境。

| 组件 | 已验证版本 |
|---|---|
| RLinf | Python 3.12.12；Torch 2.11.0+cu128；Transformers 5.6.1；SGLang 0.5.12.post1 |
| 训练内核 | Megatron Core 0.18.3，源码 commit 如上；Bridge 0.5.0；Transformer Engine 2.17.0；适配 Torch 2.11 的 FlashAttention 2.8.3 |
| Qwen 线性注意力 | flash-linear-attention/fla-core 0.6.0；TileLang 0.1.8；Triton 3.6.0 |
| 环境模拟器 | Torch 2.11.0+cu128；Transformers 4.57.6；NumPy 1.26.4；MuJoCo 3.3.0；robosuite 1.5.2；rpent-libero/rpent-liberopro 0.2.0；SAM3 0.1.0 |

使用 `uv pip freeze`、`python -c 'import torch; print(torch.__version__, torch.version.cuda)'` 和 `git -C "$MEGATRON_PATH" rev-parse HEAD` 核对并在验证后保留版本清单。源码编译需要匹配的 CUDA 编译器；可联网的纯 CPU 节点适合下载 wheels/源码，但不一定能编译 CUDA 扩展。离线计算节点应先在兼容 Linux/Python/CUDA 构建机准备完整依赖，再传输 wheels 和 Git 源码，通过 `uv pip install --no-index --find-links /path/to/wheelhouse ...` 安装。部分 wheel 缓存不等于完整离线安装包。

2026-11-13 的有限 infra 验证复用了独立模拟器环境，实际为 Python 3.11.14、Torch 2.7.1。这些结果验证了该环境中的原生接口，不代表上方 Python 3.12 / Torch 2.11 模拟器方案已完成全新安装验收；验证没有启动效果训练。

所有 Ray 节点必须有相同的 RLinf 环境、源码和模型文件。RPent 保留独立环境，不要为了统一版本把 RLinf 的 Transformers 升级为 RPent 版本。若使用自定义激活脚本，保留安装器设置的 Megatron、CUDA、NCCL 路径。本方案使用 FA2 并关闭 Torch 编译，在启动 Ray 前设置 `NVTE_FLASH_ATTN=1`、`NVTE_FUSED_ATTN=0`、`TORCH_COMPILE_DISABLE=1`。环境节点还需要可用的 EGL。

## 准备模型和标准 LIBERO 资产

将完整 Qwen3.5-4B 或兼容的全参 SFT 导出放到 `$STACK_ROOT/models/`。其中必须包括视觉权重、tokenizer、processor 和 chat template；纯语言抽取模型不能处理 adaptive 带图回合。例如在联网机器下载基础模型：

```bash
hf download Qwen/Qwen3.5-4B --local-dir "$STACK_ROOT/models/Qwen3.5-4B"
```

按照 [SAM3 仓库](https://github.com/facebookresearch/sam3)说明获取权重，并将 `SAM3_CHECKPOINT_PATH` 指向本地 `sam3.pt`。Python 包不包含模拟器资产。在 RPent 环境下载标准资产并生成独立 benchmark 配置：

```bash
source "$STACK_ROOT/envs/rpent/bin/activate"
export LIBERO_ASSETS="$STACK_ROOT/data/libero/assets"
export LIBERO_CONFIG_PATH="$STACK_ROOT/data/libero-standard-config"
env -u LIBERO_ASSET_PATH libero-download-assets \
  --assets-dir "$LIBERO_ASSETS" --skip-existing --no-cache
mkdir -p "$LIBERO_CONFIG_PATH"
python - <<'PY'
import os
from pathlib import Path
import yaml
import libero.libero as libero
assets = Path(os.environ['LIBERO_ASSETS']).resolve()
package_assets = Path(libero.__file__).resolve().parent / 'assets'
# LIBERO also constructs scene/mesh paths relative to its installed package.
if not package_assets.exists():
    if package_assets.is_symlink():
        raise RuntimeError(f'Broken LIBERO assets link: {package_assets}')
    package_assets.symlink_to(assets, target_is_directory=True)
for scene in ['libero_tabletop_base_style.xml', 'libero_floor_base_style.xml',
              'libero_kitchen_tabletop_base_style.xml',
              'libero_living_room_tabletop_base_style.xml']:
    assert (package_assets / 'scenes' / scene).is_file(), scene
config = libero.get_default_path_dict()
config['assets'] = str(assets)
Path(os.environ['LIBERO_CONFIG_PATH'], 'config.yaml').write_text(yaml.safe_dump(config))
PY
```

LIBERO 的部分 XML/mesh 路径直接相对安装包构造，独立于 config.yaml，所以还必须补齐包内 assets 链接。已有完整资产目录时保留；遇到断链应修复路径，不替换场景几何。保留 `get_default_path_dict()` 返回的包内 benchmark、BDDL 和初始状态路径。`sao_3node` 配置把此路径显式传给 episode 子进程，避免父环境中的 LIBERO-PRO 配置悄悄替换标准 benchmark。

继续使用 RPent 解释器生成标准 40-task 训练清单：

```bash
cd "$STACK_ROOT/repos/RLinf"
python examples/agent/rpent/prepare_tasks.py --workflow libero_rl \
  --libero-type standard --libero-config-path "$LIBERO_CONFIG_PATH" \
  --seeds 1-32 --output-dir "$STACK_ROOT/data/libero-rl-tasks"
```

这里按同一组 40 个任务的初始状态划分，不是未见过的任务族；称为独立 holdout 前还需检查与 SFT 训练清单是否重叠。生成 1,280 行及记录 task/初始状态数量、seeds、哈希的 manifest。seed 0 保留评测。生成器拒绝越过各任务初始状态范围的 seed，避免取模后重复同一初始状态。可使用 `--seeds 33-39 --split dev` 或 `--seeds 40-49 --split test`，为各 split 指定独立输出目录。原来的 memory 清单生成器仍为默认 `--workflow libero`，仍要求 `--memory-dir`。

## 配置共享 16 卡 SAO

Clean workflow 只保留四份依次继承的训练配置：`libero_rl_ppo` → `libero_rl_sao` → `libero_rl_sao_3node` → `libero_rl_sao_local`，完整文件名均带 `train_qwen3.5_4b_` 前缀。第一份继承原版 `train_qwen3.5_4b_ppo`，原版 LIBERO 集成仍可使用该基础配置。任务、采样和模型参数放在 clean PPO 中；SAO 只覆盖更新规则，三节点配置只覆盖 placement，local 配置只填写部署路径。

根据自己的安装位置填写 `examples/agent/rpent/config/train_qwen3.5_4b_libero_rl_sao_local.yaml`，或使用下方启动命令的路径覆盖项；未填写的 `???` 会报错，不会回退到原实验路径。需填写完整多模态模型、训练 JSONL、RPent 解释器、LIBERO 配置、episode 目录和训练输出目录。父配置将这些路径标记为必填，不提供特定机器的默认路径。`rollout.model.model_path` 同时设置 actor/critic 的 tokenizer 和 HF 导入路径。

| 资源或参数 | 配置 |
|---|---|
| 节点 0–1，GPU 0–15 | Rollout、critic、actor 分阶段共享全部 16 卡，阶段间 offload 权重、梯度和优化器 |
| Learner | TP2、CP1、DP8；开启 sequence parallel；BF16；activation recomputation；response-only logits |
| Rollout | 16 个 TP1 SGLang 副本；`replica_affinity=false`；每副本 `max_running_requests=128`；显存比例 0.7 |
| 节点 2，GPU 0–7 | 八个 agent worker，每个 32 个 episode 槽位；GPU 0 上一个 SAM3 服务 |
| 完成顺序调度 | Batch 128；`max_in_flight_rollouts=null` 和 `rollout_buffer_capacity=null` 分别解析为 256 |
| 训练预算 | 10 个数据 epoch，无额外 step 上限；1,280 条任务对应 100 次 learner 更新；每五次保存 |
| 优化 | 每次 learner 更新使用一个 actor minibatch、两个 critic update epoch；microbatch 1；actor LR `1e-6`、critic LR `5e-6`；训练语言参数，冻结视觉编码器/projector |
| Episode | Simple prompt、adaptive 观测、thinking、100 turns、10,000 env steps、四小时预算 |
| 生成 | Context 262144，output 32768；temperature 1、top-p 0.95、top-k 20、presence penalty 1.5、repetition penalty 1.0 |
| Context management | 原生消息窗口 80→40，触发比例 0.7；不额外清除工具结果或生成摘要 |

`agentloop.chat_template_kwargs.enable_thinking` 支持 `true` 和 `false`，本配方默认为 `true`。改变训练模式前，应固定 checkpoint 和任务 seed，对比原生成功率与生成成本。对照必须从新的 episode 开始，不能删除已有轨迹中的 thinking token 或行为 logprob。

`agentloop.prompt=simple_brief` 选择 RPent 的可选简短推理提示词，目标是每次决策约 256 个推理 token，不强制插入解码 token。RPent 版本需支持此变体。保留历史 thinking，并比较原生成功率与实际成本后再采用。

### 可选的 Thinking 与投影限制

若要限制 chat template 已开启的首段 Thinking，设置 `agentloop.thinking_token_budget=512` 和 `rollout.sglang.enable_custom_logit_processor=true`，并保持 `enable_thinking=true`。预算默认是 `null`。引擎记录实际强制插入的闭合 Token，该 Token 的训练 mask 为零；自然生成的闭合 Token 仍参与训练。此选项不限制后续重新开启的 Thinking 段。组装轨迹时必须保持返回的 Token ID、behavior logprob、强制标记和权重版本的对应关系。

若要启用完整投影分块，设置 `actor.model.lm_head_chunk_size=256`；默认 `0` 使用常规投影。分块路径按计分回复位置分批投影，并在反向时重算，支持 PP1/CP1 的 MBridge Qwen3.5，包括 TP 和 sequence parallel。使用上文记录的 Megatron 版本；此路径会临时替换该模型的 postprocess。Entropy 只作为 detached 诊断指标，不支持 entropy loss bonus；该模式也不支持 FP8、MTP、延迟 embedding 梯度、muP 和 autocast。这些约束界定受支持的执行方式，不代表已有吞吐或学习收益。

SAO 允许最多 256 条在途 episode，按完成顺序收集下一批 128 条轨迹，不再等待某个固定同步 batch 的全部轨迹。Episode 槽位包含等待模型与执行工具的时间；八个 worker 各 32 个槽位，足以同时展开全部 256 条在途轨迹。`max_env_initializations_per_worker=8` 将八个 worker 的同时冷启动数量限制为 64；保存首次环境状态后立即释放冷启动槽位，不等 episode 结束，因此预热后仍可保持 256 个活跃环境。这只约束 CPU、import 和 CUDA context 初始化压力，不降低 rollout 容量。Buffer 保存已完成轨迹；额外排队的 prompt 不会提高在途上限。每个 episode 通常只有一个未完成模型请求，因此每副本 128 是准入上限，并不表示每张 GPU 始终都有 128 个活跃请求。

共享 GPU 仍需切换阶段：暂停并收回模型请求、卸载 rollout 显存、依次训练 critic 和 actor、同步权重，再恢复等待中的请求。训练时环境动作可以继续，但模型生成暂停。SAO 的学习规则也有变化：DIS actor loss、0.3/5.0 clip 边界、alpha 为 1.5 的 `sao_gae` 和 SAO critic 目标；它直接使用 rollout 保存的 behavior logprob，不重算旧 policy 的 logprob。任务、模型、prompt、采样和压缩参数均继承 clean PPO。

工具结果和图片保留在后续输入中。Policy loss 只作用于 policy 生成的 assistant token，user/tool/history token 不参与监督。逐 token 前缀兼容的连续 turn 合并，压缩或历史改写后分段。Actor 和 critic 使用相同的 response 位置映射；response-only logits 要求 CP1。

启用 `tools.rpent.store_images_on_disk=true` 时，所有 rollout 和 learner 进程必须能读取**相同绝对路径下的图像**。`tools.rpent.output_dir` 需要在模型节点与环境节点间共享，并在轨迹或保存的 checkpoint 仍引用它时保留；不同节点上同名的本地目录不够。没有共享图像存储时保留默认 `false`，让轨迹携带图像字节，但内存成本更高。不要清理可恢复 checkpoint 所需的 Episode 产物。

### 保留图像 Turn 的 Token 边界

合并 Qwen3.5 图像 turn 时，在每台模型节点启动 Ray **之前**启用保留 token 的 processor：

```bash
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rlinf.hybrid_engines.sglang.multimodal
```

adaptive 训练和评测模板也通过 `rollout.env_vars` 传入此变量，使 worker 启动不只依赖终端环境。

该配置使用 SGLang 的外部 processor 注册入口。图像请求携带已展开的 prompt IDs 和原始图片；适配器核对图片数量及各图 pad 段与处理后 grid 的对应关系，保留原始文本和动作 IDs，再由 SGLang 根据这一序列计算 offset 和 M-RoPE。对已采样 token 解码后重新编码，即使文本相同，也可能改变 token 边界。字符串请求保持上游行为；该图像适配器会拒绝携带视频、音频或预计算图像输入的 token-ID 请求。

使用包含 config、tokenizer 和 processor 文件的本地模型目录运行回归：

```bash
CUDA_VISIBLE_DEVICES='' SGLANG_USE_CUDA_IPC_TRANSPORT=0 \
  RLINF_TEST_QWEN_VL_MODEL=/path/to/Qwen3.5-4B \
  PYTHONPATH=.:tests/unit_tests \
  python -m pytest tests/unit_tests/test_agent_ppo.py -k qwen_image_processor
```

这些 CPU 检查覆盖输入 IDs、图像 tensor、offset，以及 HF 与 SGLang 的 M-RoPE 一致性。部署新版本前，还需要验证真实引擎的生成和 logprob、完整 learner 更新及 checkpoint 恢复。

## 启动三节点 Ray 和 SAM3

**每个节点**启动 Ray 前都先 source RLinf 环境，导出 `MEGATRON_PATH`、`NVTE_FLASH_ATTN=1`、`NVTE_FUSED_ATTN=0`、`TORCH_COMPILE_DISABLE=1`，并设置足够的 `RLINF_TIMEOUT`，例如 `10800`。按实际网络选择网卡，`RLINF_COMM_NET_DEVICES=eth0` 仅为示例。Ray worker 在启动时继承这些环境变量。

```bash
source "$STACK_ROOT/envs/rlinf/bin/activate"
export PYTHONPATH="$STACK_ROOT/repos/RPent:$STACK_ROOT/repos/RLinf:$MEGATRON_PATH${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rlinf.hybrid_engines.sglang.multimodal
export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 TORCH_COMPILE_DISABLE=1
export RLINF_TIMEOUT=10800
export HEAD_IP=10.0.0.10  # 各节点可达的 head 地址
export RLINF_NODE_RANK=0
ray start --head --port 6387 --node-ip-address="$HEAD_IP" --num-gpus 8
```

模型节点 1 和环境节点 2 使用同样的源码/环境，各自填写 rank 和本地地址：

```bash
export RLINF_NODE_RANK=1  # 环境节点使用 2
export NODE_IP=10.0.0.11 # 环境节点填写自己的地址
ray start --address="$HEAD_IP:6387" --node-ip-address="$NODE_IP" --num-gpus 8
```

通过 `ray status --address="$HEAD_IP:6387"` 确认三节点、24 GPU。不要在已占用节点上启动冲突的第二个 Ray 集群。长训练前验证多节点 NCCL 通信；防火墙、网卡和 RDMA 设置取决于集群。

在节点 2 的独立终端使用 RPent 环境启动 SAM3：

```bash
source "$STACK_ROOT/envs/rpent/bin/activate"
cd "$STACK_ROOT/repos/RPent"
export SAM3_CHECKPOINT_PATH="$STACK_ROOT/models/sam3/sam3.pt"
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8101
```

Agent worker 和 episode 子进程都位于节点 2，因此 `http://127.0.0.1:8101` 连接本地 SAM3。策略的本地 HTTP endpoint 通过 RLinf channel 将模型请求发往两台模型节点，无须跨节点暴露 SAM3。

## 启动、检查与恢复

填好 local 配置后，在 head 节点执行：

```bash
source "$STACK_ROOT/envs/rlinf/bin/activate"
cd "$STACK_ROOT/repos/RLinf"
bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_sao_local \
  runner.output_dir="$STACK_ROOT/logs/training" \
  data.train_data_paths="[$STACK_ROOT/data/libero-rl-tasks/train.jsonl]" \
  tools.rpent.python="$STACK_ROOT/envs/rpent/bin/python" \
  tools.rpent.libero_config_path="$LIBERO_CONFIG_PATH" \
  tools.rpent.output_dir="$STACK_ROOT/data/episodes" \
  rollout.model.model_path="$STACK_ROOT/models/Qwen3.5-4B"
```

Hydra 覆盖参数放在配置名之后，例如 `rollout.model.model_path="$STACK_ROOT/models/Qwen3.5-4B-SFT"`。末尾加 `--cfg job --resolve` 可以只检查最终配置而不启动训练。使用 `runner.resume_dir=/path/to/checkpoints/global_step_N` 从完整 SAO checkpoint 恢复两套模型/优化器与 buffered/pending rollout 状态。只选择包含 actor/critic 状态及最后写入的 `data/data.pt` 标记的完整 checkpoint。Pending prompt 会重新入队，并非持久化到一半的模拟器状态；恢复时保持数据集、算法、world size 和引用图像兼容。同步 PPO checkpoint 不包含 SAO 所需的队列状态。Actor/critic checkpoint 和日志位于 `runner.output_dir` 下的实验目录；episode 请求、工具图片和结果位于 `tools.rpent.output_dir`。

使用 Hydra 覆盖参数运行独立短 smoke，无须单独维护 smoke YAML。检查 loss/梯度是否有限、参数是否实际更新、模型副本活动、任务成功率及基础设施错误。短 smoke 通过证明连通和优化器执行，不代表最长 262144-token 工作负载一定能放入显存。遇到显存限制先保留证据并诊断，再决定切分或预算调整。

独立评测需要另行分配可用资源；评测默认 placement 是一台八卡模型节点与独立环境节点，不要与训练共用已占用 GPU。原始评测模板使用 LIBERO-PRO、1200 步预算和更短历史窗口，**不能直接与标准训练配方比较**。下方覆盖这些参数以对齐训练；placement 仍需按实际分配资源设置。

导出时 `--vlm` 必须指向训练初始化所用的完整模型，以保留冻结的视觉权重和 processor。从 SFT 开始训练时相应替换路径。导出目录必须是新目录；HF 导出只用于推理，不能替代包含优化器和队列的恢复 checkpoint。

```bash
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
  +tools.rpent.libero_config_path="$LIBERO_CONFIG_PATH" \
  tools.rpent.libero_type=standard tools.rpent.max_episode_steps=10000 \
  agentloop.episode_timeout_s=14400 tools.rpent.process_timeout_s=15000 \
  agentloop.history_compaction.max_messages=80 \
  agentloop.history_compaction.keep_messages=40 \
  +algorithm.sampling_params.presence_penalty=1.5
```

评测初始模型时，只改变模型与输出路径，复用同一份 manifest、任务/初态 ID、prompt、采样参数、轮数/动作预算及压缩设置。分别报告原生成功数/总数、轮数耗尽和基础设施失败。训练完成队列的成功率不是独立评测；固定 40 条 case 也不足以确认很小的学习增益。

原版不传配置名的启动行为及原版 LIBERO YAML 继续保持不变。
