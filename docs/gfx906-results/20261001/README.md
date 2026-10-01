# 2026-10-01 T5810 / MI50 审计证据

对应 [gfx906 开发指南](../../GFX906.md)。固定模型/ggml版本见指南，权重和二进制不入仓。

- `ctest-final.log`：41项、40通过/1 hipBLASLt跳过/0失败；不包含被preset排除的两项。
- `python-tests-final.log`：18项CPU-only入口/下载安全回归。
- `hip-layer-handoff.log`：两卡25MHz主动时钟校准、正反向和同设备交接，137,980,416项精确检查。
- `smoke-{dual,single,dual-threshold1,reverse}.json`：真实模型配置、prompt/generated IDs、文本、退出状态和原始引擎计数。
  4模式×4提示（每模式153个token）完全一致；`min-p=1`仍提出22个draft，不是MTP-off。
- `cli-no-mtp-v2.json`：无MTP单卡原生pack，既有 `--spec 2 --prefill 128` verifier路径，算术token一致。
- `reference.*`：pinned llama.cpp的CPU-only `llama-completion`，相同23-token模板、GGUF、greedy；文本`12`，退出0。
  这不是全logits/全层oracle，也不把一个短答案称作完整模型CPU parity。
- `model-plan-ready.json` / `model-sha256-verified.log`：模型身份、字节、LFS SHA256及下载结束证据；4GiB底线。
- `environment-final.json`：03:31 UTC GPU/服务/磁盘/内存/监听/内核日志/UMC；`edac-readonly.json`：03:35只读DRAM计数。
- `summary.json`：从原始计数计算吞吐、检查token一致性、明确硬件风险与验收范围。

**不能忽略硬件告警**：03:25主机DRAM CE/MCE、CMCI storm、两页soft-offline；MC0累计CE173/UE0，
不是本次新增173。GPU UMC全0。软件测试通过但主机稳定性/生产验收不通过，停止继续加压、等待硬件维护授权。

初轮错误原样留档：oracle注册成不存在的`gemm` mode、跨空闲CPU sleep时钟校准不适用、native-pack拒绝`--spec 0`。
修正之后重新运行，不把原失败或hipBLASLt跳过伪装为通过。完整编译/打包/引擎加载日志保留在机器
`/home/chris/dev/strata-gfx906/logs`（含首次编译的graph API调用错误），大型warning输出未上传。

## 复核

```bash
python3 tools/gfx906_summary.py docs/gfx906-results/20261001
```

校验使用显式异常、不依赖assert，`python3 -O`也不能绕过。原始首次`smoke-dual.json`的派生
`decode_tok_s`曾用`generated-1`；**不要用它**。汇总根据原始`engine.generated/decode_ms`重算，
后续工具也已修正，128/7195.3ms约17.8tok/s与引擎日志一致。

烟测时间包括特定decoder窗口，但不包括模型加载/cache预填；这是少量冒烟样例，不是隔离性能A/B。
部分对照轮次与CPU参考程序编译重叠，不能把观测表格转换成普遍双卡/投机加速倍率。
