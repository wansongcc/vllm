# Cost-Aware Chunked Prefill 项目记忆

更新日期：2026-10-31（Asia/Shanghai）。这是项目上下文与决策记录，不是替代 AGENTS.md 的贡献规则。

## 1. 用户明确的目标与合作方式

- 项目用于求职经历：LLM 推理系统、AI Infra、高性能后端方向。重点是理解、实现、调试、测试、性能分析与工程交付，不以论文、原创性或创新性为验收条件。
- 用户希望持续围绕这个 idea 商讨、提问、学习并逐步实现。解释代码时帮助用户理解设计与运行逻辑，不只交付代码。
- 用户此前要求将核心计划压缩至一周，工作量与内容不减。七天是目标安排，不是完成保证；硬件、依赖、实验耗时尚未确认。
- 本轮要求是阅读、梳理与建立记忆，没有要求立即修改调度器或启动大规模实验。
- 附件中的科研差异化、论文比较、异构集群等讨论，不自动升级为项目验收条件。

## 2. 来源与当前事实

- 本对话此前的理论推导、工程计划与七天安排。
- 用户附件：/home/cc/.codex/attachments/ad08959c-e54e-4fe3-8931-4131e1563841/Pasted text.txt。
- 本地仓库：/home/cc/vllm；2026-10-31 检查的 HEAD 为 6e448d0ea9。
- 此前已经阅读 V1 Scheduler 的 RUNNING/WAITING 分支、EngineCore 入口和现有 Attention benchmark 说明；尚未完成调度器修改、GPU 成本采集、模型拟合或性能验证。
- 不将助手曾经建议的硬件、模型或实验阈值视作用户确认。附件提到 RTX 3060、Qwen3-0.6B，但实际可用 GPU、显存、模型、环境仍需检查。

## 3. 项目定义与价值

名称：Cost-Aware Chunked Prefill：面向长短上下文混合负载的计算成本感知调度。

目标：根据当前 Chunk 的新 Token 数 q、有效历史上下文 c 和实测 GPU 成本，动态限制 Prefill 工作量，降低长 Prefill 对 Decode 尾延迟的干扰，同时控制吞吐损失、长请求 TTFT 和公平性代价。

工程闭环：问题复现 → 源码与运行链路理解 → 数据采集 → 成本模型 → 调度接入 → 正确性与性能验证 → 可复现交付。

不预设一定有收益，不编造提升百分比。收益应与调优后的原生策略比较；没有收益的场景也要记录。

## 4. 理论共识与附件修正

标准因果全注意力的有效 Query-Key 配对数：

    W(q,c) = q*c + q*(q+1)/2

- q=1024、c=0：W=524800。
- q=1024、c=32768：W=34079232，约为前者的 64.94 倍。
- 附件中的 34030592 是算术错误，不沿用。
- W 是有效配对数，不是 GPU 毫秒，也不代表实际 Kernel 完全只做这些运算。
- 完整 Prompt 在无重算、无缓存跳过时，无论如何切块，总配对数仍为 L*(L+1)/2。优化改变执行时序、批处理效率与延迟权衡，不消除理论 Attention 总量。
- Full Attention GQA 仍可使用该 Token 配对特征，实际成本依赖 Head 配置；滑窗、稀疏、线性与混合注意力另行建模。
- KV 缓存避免重算历史 K/V，但新 Query 仍需访问允许范围内的历史 KV。

## 5. 方案分层，避免混淆版本号

A. 观测与基线：先复现上下文长度造成的成本差异和 Prefill/Decode 干扰。
B. 理论配对预算：保留 Token Budget，加 W 预算，作为简单原型和消融对照；单位是配对数。
C. 离线标定时间预算：独立 PrefillCostModel；起始形式 t_hat(q,c)=a*q²+b*q*c+d*q。
D. 批次与 Decode 感知：用实际混合 Batch 数据预测整体成本，考虑 Decode 负载和安全余量。
E. 在线反馈：采样实际耗时修正预测偏差，必要时更新参数；属于进阶范围，不默认是七天核心验收项。

附件中的 V1/V2 命名与此前对话不同，后续优先使用上面的功能名称。

主要约束：
- 原有 Token、KV、模型长度等硬约束全部保留。
- 新增成本预算作为可配置软目标；最小进度执行可能超成本目标，但不能越过硬约束。
- 选择满足剩余预算的最大合法 Chunk；二分搜索要求成本函数在搜索范围内单调。
- 非负回归、单调性检查、固定开销处理、超出标定范围的回退都要考虑。
- 单请求成本相加只是近似；批次模型需要留出混合负载验证，固定开销不能逐请求重复计算。
- 对非可加批次模型，撤销请求后需重新计算批次预测或一致地更新增量，不能机械退还一个固定单请求成本。
- 无 Decode 压力时避免无意义限流，可在后续比较条件预算与固定预算。

## 6. 源码运行链路与接入要点

运行链路：请求输入 → EngineCore → Scheduler → 请求内 KV 分配/复用 → SchedulerOutput → Executor / GPU Model Runner → 采样和状态更新 → 下一轮。

重点文件：
- vllm/v1/request.py
- vllm/v1/engine/core.py
- vllm/v1/core/sched/scheduler.py
- vllm/v1/core/sched/output.py
- vllm/v1/core/kv_cache_manager.py
- vllm/v1/worker/gpu_model_runner.py
- vllm/config/scheduler.py
- tests/v1/core/test_scheduler.py
- tests/v1/core/test_scheduler_e2e.py
- benchmarks/attention_benchmarks/README.md

关键实现条件：
- 同时覆盖 RUNNING 中续跑的 Prefill 和 WAITING 中新接入/恢复请求。
- RUNNING 不等于 Decode；不能把现有队列遍历描述成天然全局 Decode 优先。
- Prefix Cache 查询后使用有效上下文；缓存命中不意味着历史 Attention 免费。
- num_computed_tokens 在异步等模式下不能一概当成 GPU 已完成位置。
- 应用全部裁剪和对齐后，以实际安排的 q 记账；分配失败、跳过、抢占撤销时保持预算一致。
- 处理 q=0、尾块、极小预算、无法推进、取消和恢复等边界。
- 成本模型独立封装，配置提供开关、预算和标定参数；关闭时保持原行为。
- 第一版尽量保持原队列顺序和 SchedulerOutput 接口，减少归因困难。
- CPU 调度开销、CUDA Graph、异步重叠都可能影响净收益。

## 7. 范围与实验设计

起步范围建议：单 GPU、一个 Dense Decoder-only 全注意力文本模型（允许 GQA）。先固定/关闭 Prefix Cache、推测解码、异步调度等干扰因素，再做兼容性验证。此范围是此前建议，实际配置需按环境确认。

测试分层：Attention 微基准 → 整模型 Forward → 在线服务端到端。

基线：原版配置；原版扫描 Token Budget 后的较优配置；已有长 Prefill Threshold；理论配对预算；离线实测成本版本。

P-PAS 等可作阅读或可选对照，不要求为求职项目复刻所有论文方法。已有策略叠加需要消融，不假设收益可叠加。

负载：纯 Prefill、短请求、长短混合、持续 Decode 时注入长 Prompt、动态突发、容量扫描、长时间稳定性与公平性。

指标：TTFT、ITL、TPOT 的分布，输出吞吐、Goodput、端到端延迟、GPU/Step 时间、Scheduler CPU 开销、KV 使用、抢占、最长等待与完成率。

ITL 衡量相邻 Token 停顿；TPOT 是请求生成阶段平均每 Token 时间，不能替代 ITL 尾部指标。预测 Step 预算不是端到端 SLA 保证。

数据规范：固定版本/硬件/模型/精度/backend/请求轨迹；预热、重复测量；保存原始数据和完整命令；按执行形状/负载留出验证，不把相同配置的重复测量随机拆分后宣称未见形状泛化。

GPU 计时：CUDA Event 或 Profiler；明确计时边界；不在在线 Scheduler 每轮 synchronize；后续可异步回传测量与 Step 标识。EWMA 应修正形状条件化预测的偏差或比例，不能把不同负载的耗时直接平滑当成完整成本模型。

正确性：资源约束、预算回退、多轮完成、最小进度、关闭回归、Prefix Cache、异步兼容；生成质量评测；批形变化可能产生数值差异，需要合理分析。

## 8. 七天目标安排（沿用用户此前要求）

1. 源码链路、环境与原版基线。
2. q/c 成本采集、观测与混合负载问题复现。
3. 理论预算实现、状态与资源记账、单元测试。
4. 离线拟合、批次预测接入、误差与开销报告。
5. 正确性、模型评测、缓存/异步兼容验证。
6. 调优基线对照、负载扫描、公平性、稳定性。
7. 必要重测、复现脚本、图表、设计文档、演示和面试材料。

这是待执行计划，不代表任何一天已完成。在线自校正、异构设备、混合注意力、分布式调度保留为后续扩展，不因附件存在而默认纳入一周。

## 9. 下一个具体里程碑

先阅读 EngineCore.step → Request → Scheduler.schedule → update_from_output。

手推 2048 Prompt、1024 Token Budget、无缓存/推测/异步且 KV 充足时的两轮 Prefill 与随后 Decode，确认计数更新位置、WAITING→RUNNING 转换与预算约束。

随后取得第一张实测性能表：固定 q=1024，c 取 0/2048/4096/8192/16384（按显存与模型长度调整），记录整模型 GPU Forward 时间；复现持续 Decode 受到长 Prefill 干扰。再决定拟合模型和预算范围。

## 10. 待验证信息与不可直接沿用的内容

- 附件宣称的 long_prefill_token_threshold_adaptive：本次在本地 vllm/config/scheduler.py 搜索未命中；prefill_schedule_interval 命中。不可把附件所述最新功能当成本地已有功能。
- 附件的 vllm bench forward 及 --case 语法尚未验证。本次 vllm 路径下未发现匹配 benchmarks/.*forward 的文件；这不是全局不存在的证明。执行前检查本地 CLI/源码，必要时用已有 Attention benchmark 与合适的整模型测量工具。
- DeepInfer 的设计细节、P-PAS/RFC #52906 的内容与公开状态尚未独立验证；本轮只整理用户提供的材料，不据此宣称已复现或已核实。
- 实际硬件、模型上下文上限、环境是否就绪、可用 GPU 时间、性能目标仍待确认。
- 附件中代码块粘贴格式受损，不能直接运行；示例毫秒预算和系数不是实测参数。

## 11. 交付与贡献规则

求职交付：可回退实现、测试与模型评测、成本参数与误差、调优基线比较、原始数据、可复现脚本、设计说明、限制和实测简历表述。

遵守仓库 AGENTS.md：Python 通过 uv 与 .venv/bin/python；扩展附近测试，模型相关改动需评测；不把一次性 Kernel benchmark 放入 tests。

当前不要求向上游提交 PR。以后若要求提交，仍需完成重复工作检查、人类逐行审阅与相关测试，说明 AI 辅助；求职定位不免除贡献规范。

## 12. 后续记录方式

后续讨论区分用户确认、实现建议、待验证假设、实际测量四类信息。重要决策、源码变更、实验结果与下一步写回此文件；不要把计划记成已完成。

此文件是可持久读取的本地项目记录，不代表所有新聊天都会自动加载。新聊天可显式要求先阅读本文件恢复上下文。

## 13. 当前学习阶段（2026-10-31）

用户要求先说明核心源码与完整业务逻辑，由用户理解后主动告知，再推进下一步。当前不启动调度实现或性能采集。

已核对：EngineCore.step 串联 schedule、execute_model、必要的 sample_tokens、update_from_output；Scheduler._update_after_schedule 在 GPU 输出返回前推进 num_computed_tokens 与 num_in_flight_tokens，输出处理再减少 in-flight 并追加生成 Token。未完成的 Prefill Chunk 不产生有效用户输出。get_computed_blocks 为获取 logits 保留尾部重算，不能假设完整缓存命中后总是零计算。

本次讲解重点：Request 计数、RUNNING/WAITING、Token/KV 双重资源约束、SchedulerOutput、GPU 输入元数据、采样/停止、抢占重算；用 2048 Prompt / 1024 Budget 手推轨迹。用户尚未确认理解。

理解后的下一步：检查实际硬件与环境，固定配置，跑通原版最小服务/压测，随后采集固定 q 改变 c 的执行时间。

## 14. 环境与模型检查（2026-11-01）

用户已确认理解核心业务链路，进入环境与基线阶段。本次只读检查，无依赖安装、无模型权重加载、无服务停启。

- GPU：4×Tesla T4，每卡 15360 MiB，compute capability 7.5。检查时 GPU 1 为 0 MiB 占用；GPU 0/2/3 分别约 13639/13933/11717 MiB，被已有 llama-server 占用。不得未经授权停止这些服务；启动实验前重新确认空闲状态。
- 驱动 570.211.01；系统 CUDA Toolkit 12.8.93（/usr/local/cuda/bin/nvcc，不在当前 PATH）。nvidia-smi 显示 CUDA 12.8；PyTorch 构建 CUDA 12.9，三者不是同一个版本概念。已验证 torch CUDA 可见且支持 sm_75，尚未执行模型或算子兼容性测试。
- .venv Python 3.12.9；uv 0.13.6（.local-bin/uv）；torch 2.13.0+cu129；vllm wheel metadata 0.27.1+cu129，import __version__ 0.27.1；transformers 5.19.0；triton 3.7.1；flashinfer-python 0.7.0.post1。
- pytest、pre-commit 未安装；.git/hooks/pre-commit 不存在。flash-attn 独立包未安装，不据此推断所有 Attention backend 不可用。
- vllm wheel 来自与仓库 HEAD 6e448d0ea9 对应的 wheel URL，不是 editable 安装。
- 在仓库根目录导入会命中源码包，缺少 vllm._version 且无法找到源码侧的 vllm._C_stable_libtorch。必须在开发前解决源码与编译扩展的连接问题，不能假设修改仓库会影响 wheel。
- 在 /tmp 用 /home/cc/vllm/.venv/bin/python 成功导入已安装 vllm 与实际扩展 vllm._C_stable_libtorch。旧模块名 vllm._C 不存在，不能据此说该 wheel 扩展坏了。
- 内存约 187 GiB，总 available 约 116 GiB；Swap 接近满。/home 所在磁盘可用约 2.9 TiB。

本地模型（只验证配置与文件存在、大小、分片索引；不代表已经加载运行或校验全部权重内容）：
1. /home/cc/yhbian/B02/models/Qwen2.5-0.5B-Instruct：单 safetensors 约 0.920 GiB，tokenizer.json 与 tokenizer_config.json 存在；Qwen2ForCausalLM，24 层，14 Q heads/2 KV heads，use_sliding_window=false，max_position_embeddings=32768，配置 BF16，未量化。建议首个候选，用空闲 T4 和显式 FP16 验证。模型在其他项目目录中，只读复用，不修改其文件。
2. /home/cc/model/Qwen2.5-VL-3B-Instruct-AWQ 与 /home/cc/model/model/Qwen2.5-VL-3B-Instruct-AWQ：每个目录有单权重约 3.168 GiB 和 tokenizer，AWQ 多模态，暂不作为首选。未确认两目录是否相同实体或副本。
3. /home/cc/vllm/models/Qwen3.5-35B-A3B-GPTQ-Int4：14/14 索引分片存在，总约 22.743 GiB，Tokenizer 存在。MoE + 混合线性/全注意力，40 层其中 10 层全注意力；不适合作为单张 T4 的初版全注意力基线。
4. 还发现 Qwen2.5-1.5B、32B、72B，Llama-3.2-3B 和 Qwen3.5-35B 的 GGUF 文件/分片。本次未检查全部 GGUF 完整性与 vLLM 兼容性，不列作已验证可运行模型；ggml-vocab 文件不是模型权重。

下一步建议：在 GPU 1 上用已安装 wheel 和本地 Qwen2.5-0.5B-Instruct 做短上下文 FP16 最小生成验证，再跑在线基线；开发前安装必要测试/检查依赖并配置可编辑源码。当前阶段没有模型运行结果，不宣称环境完全就绪。

## 15. 开发主机迁移（2026-11-01）

用户指定改用 cc@192.168.2.126（主机名 cloud26）继续开发，并明确要求先将本仓库 push 到 Git 远端，再在 126 拉取。认证口令不写入文件或 Git。

远程只读检查：8×RTX 3090，每卡 24576 MiB，检查时均空闲；驱动 580.126.09；约 483 GiB 可用内存；/home 约 1.2 TiB 可用。远端已有 /home/cc/vllm，HEAD 为 1edfd09ff，工作区干净，未发现其中的 .venv；当前 PATH 未找到 uv。/home/cc/models 有若干 GGUF 模型目录，尚未找到适合本项目的 HF config.json。

迁移方式：推送 origin（https://github.com/wansongcc/vllm.git）的 codex/v0.27.1-study 分支，用户随后确认可删除 126 的旧 /home/cc/vllm；先在临时目录从 Git 远端克隆并验证，再替换为 /home/cc/vllm。模型、缓存、虚拟环境和无关本地脚本不纳入本次提交。后续以 126 为执行主机，通过 SSH 运行；当前聊天工作目录仍在原主机。
