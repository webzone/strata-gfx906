# strata-gfx906 — 开发约定

这是 `Niko1221/Strata@30ec18ec7094550fcc594fd948220d511d80464e` 的实验性 MI50 移植。
工作分支 `gfx906`；`origin` 是 `webzone/strata-gfx906`，`upstream` 只作上游对照。

1. **先读设备文档**。T5810 权威档案在设备仓 `t5810/README.md`（本机路径
   `/Users/chris/dev/local_devices/t5810/README.md`）。使用文档给出的 `chris@t5810` 登录，
   不猜账户、扫描端口或搜索其他项目的凭据。
2. **网络和既有服务优先**。编译/测试之前检查 GPU、RAM、磁盘、Docker/监听状态。
   不为测试停止工作负载；不改驱动、ROCm、网络、温控/调校服务，不自动恢复历史容器。
   模型/打包工具依赖只放本项目 `.venv`；不安装系统包。
3. **兼容性显式 opt-in**。`STRATA_EXPERIMENTAL_GFX906` 默认 OFF；安装器需要
   `--experimental-gfx906`。必须编译真实 gfx906、保留运行时架构检查；禁用 GPU 冒充/
   `HSA_OVERRIDE_GFX_VERSION`。没有 gfx906 的 TheRock wheel index，不使用 RDNA wheels 顶替。
4. **wave64 中保留逻辑 wave32**。不能把所有 `32` 机械换成 `64`。ballot 半波以线性 block
   线程索引选择；二维/三维 block、INT8 dot4 符号/溢出、shuffle 都要做数值回归。
   gfx906 时钟换算依赖实测 25 MHz，需要在每张测试卡上校验。
5. **多卡是 layer/pipeline split，不是 tensor parallel**。每卡管理自己的层和缓存。
   verify 交接使用各设备单独取得的 mapped-host alias；先同步生产者再启动消费者。
   不依赖 P2P，不在 GPU 忙时强跑双卡测试。`STRATA_HIP_MULTIGPU_TESTS=ON` 的测试必须有两张可见卡，
   不把缺卡跳过当通过。
6. **固定依赖和模型身份**。ggml/llama.cpp 固定 `3cf03257f219afbe7334045ff7c6a06ac68c627d`。
   当前验收模型是 GSQ-RCO IQ2_XS，按工具中的 HF revision / LFS SHA256 取文件。
   文件名不是每个张量的量化类型：本模型 gate/up 混合 IQ2_S、IQ2_XXS、IQ1_M，down 是 Q2_0。
   不强制所有专家走 MMQ；不重复写约 35.5GB 的 `experts.bin`，改用现有 resident-from-GGUF 路径。
7. **空间和原始数据不能随便删**。下载/准备阶段保留 4 GiB 系统盘安全余量；先核算 GGUF、
   float pack、MTP 源张量及量化/runtime 输出的峰值。允许断点续传，但不能覆盖损坏文件假装完成，
   也不删除别的模型/镜像腾空间。
8. **证据口径**。独立 CPU dequantizer/double accumulation 是合成算子 oracle；双卡交接探针
   不是模型 parity。跳过、缺失 fixtures、部分采样验证都必须明示。模型 token/s 只能来自真实生成，
   不用 dot4 微基准或 warm-cache 算子时间换算。
9. **改动实测后 commit + push**。按事项提交到 `origin/gfx906`，不强推、不丢弃他人改动；
   同步更新本项目 `docs/GFX906.md`，涉及 T5810 状态的还要回写设备仓 README 和变更记录。
   权重、二进制、venv、完整编译 warning 日志不入 Git；小型可审计验收结果可以入仓。
