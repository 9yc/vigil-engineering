# Vigil Engineering

本仓库公开 Vigil 的一个独立研究切面：**同容量下变长对象的分配与碎片**，
并提供进程通信、二进制编解码、结构化输出验证及测量工具。
公开源码可独立构建运行，无需原项目代码或数据。

| 交付 | 可直接核验的内容 |
|---|---|
| [容量分配实现](capacity_study/pool.py) | 固定槽位、分块分配、连续 first-fit、释放合并与失败归因 |
| [同预算请求回放](capacity_study/replay.py) | 相同容量和请求生存期下的接纳数、占用与碎片 |
| [容量分配测试](tests/test_capacity_pool.py) | 独立位图对照、边界检查、句柄误用与合成案例 |
| [采样工具](tools/sample_nodes.cpp) | 固定周期采样、双时钟时间戳、输入检查、退出与文件保护 |
| [CSV 汇总](tools/summarize_csv.py) | 指定数值列、分类汇总、中位数和范围、异常数据检查 |
| [测量完整性检查](tools/validate_measurements.py) | 样本总数、迭代顺序、缺失与重复记录、有效耗时 |
| [时间窗口积分](tools/integrate_power.py) | 采样间隔、窗口覆盖、功率积分及平均功率 |
| [进程通信](runtime_support/process_link.py) | 后台读取、请求与响应、超时、诊断记录、进程退出与回收 |
| [二进制编解码](runtime_support/binary_codec.py) | 小端字段、UTF-8 字符串、完整性检查与边界验证 |
| [结构化输出验证](runtime_support/json_output.py) | JSON 解析、格式回退、参数类型、必填字段与枚举 |
| [自动化测试](tests/test_tools.py) | 合成文件采样到 CSV 统计的运行链路，以及错误处理 |

## 容量管理研究切面

对象大小不一致时，固定槽位可能在仍有空余容量的情况下拒绝请求。
这里比较三种常见分配方式，研究请求大小、分配粒度和空闲区间如何共同影响接纳结果：

- **固定槽位**：每个对象占一个槽，超过单槽上限的对象被拒绝。
- **分块分配**：对象按需占用多个块，允许块不连续，并记录取整产生的内部浪费。
- **连续 first-fit**：选择第一个足够大的区间，释放时合并相邻空闲区间，
  并区分容量不足和外部碎片。

三者使用相同总预算和请求生存期，不预知未来请求。分配单位由调用方定义；
实现只记录占用，不存储负载、不搬移数据，也不估计元数据及读写开销。
这些是经典分配方式在同一接口下的对照，不把其中任一种算法单独作为新算法主张。

```python
from capacity_study import CapacityPool

pool = CapacityPool(24, mode="blocks", block_size=2)
allocation = pool.allocate(3)
assert allocation is not None
assert allocation.reserved_units == 4
pool.check_invariants()
pool.free(allocation)
```

运行仓库中的合成请求回放：

```sh
python3 -B -m capacity_study.replay
```

| 合成案例 | 分配请求数 | 固定槽接纳数 | 分块接纳数 | 连续 first-fit 接纳数 |
|---|---:|---:|---:|---:|
| 多个小对象 | 6 | 4 | 6 | 6 |
| 对象超过单槽上限 | 2 | 0 | 2 | 2 |
| 空闲区间被分隔 | 4 | 3 | 4 | 3 |
| 块内取整负面对照 | 4 | 4 | 2 | 4 |

每个案例的容量和粒度见 [合成输入](examples/capacity_events.json)。
最后一项展示分块粒度不合适时的损失。结果只描述这些合成请求的分配行为，
不代表实际应用的加速或整体收益。

## 最近可公开进展

- **2026-11-10，已有代码记录**：实现按需占用、分块分配、连续区间释放合并，
  并记录有效负载占用、预留占用和峰值。
- **2026-11-13，本次公开整理**：抽取独立容量接口，补充固定槽位对照，
  加入严格参数检查、重复释放与外来句柄检查。
- **2026-11-13，验证**：增加四组同预算合成回放和独立位图对照，
  本地全部 47 个测试通过。分块和连续两条路径另与已有实现逐操作对照，
  共 600 次分配/释放操作的结果一致。

## 构建与测试

依赖 Linux、C++17 编译器、GNU Make 和 Python 3.8 及以上版本。Python 只使用标准库。
先把 `ARTIFACT_DIR` 设置为你选择的、位于源码目录以外的临时产物绝对路径，然后运行：

```sh
make test ARTIFACT_DIR="$ARTIFACT_DIR"
```

构建产物、测试输入和测试输出均写入该目录。测试使用本地合成文件和子进程，
GitHub Actions 执行相同入口。

## 使用采样工具

```sh
"$ARTIFACT_DIR/sample_nodes" \
  "$ARTIFACT_DIR/samples.csv" "$NODE_A" "$NODE_B" 10
```

`NODE_A`、`NODE_B` 是两个可读取、可回到文件起始位置的文本节点路径。
每个节点提供一个有符号 32 位整数。工具按 100 ms 的绝对单调时钟计划采样，
记录实时时钟、单调时钟及两个原始整数；错过的时间点会跳过，不会补发密集样本。
它不是硬实时采样器，也不保证两个节点的读取具有原子性。

最后一个参数是样本数；省略或设为 `0` 则持续运行，收到 `SIGINT` 或 `SIGTERM`
后关闭输出。输出文件必须不存在。工具不推断单位，不换算功率或能量，
节点权限、单位和采样适用性由调用方确认。

## 使用 CSV 汇总

```sh
python3 -B tools/summarize_csv.py examples/synthetic_measurements.csv \
  --column value --group-by case
```

示例中 `sample-a` 的中位数为 `4.0`，`sample-b` 为 `3.0`，每组有三个样本。
这些数值只用于演示，不代表实际运行性能。省略 `--group-by` 可汇总整列；
`--output` 可指定一个尚不存在的输出文件。空输入、缺失字段、重复列名、
非数值及非有限数值会导致非零退出，不会被静默剔除。

## 检查测量完整性

```sh
python3 -B tools/validate_measurements.py examples/synthetic_durations.csv \
  --expected 3
```

检查记录数恰好符合预期、迭代编号按顺序覆盖 `0` 至 `expected-1`，
且耗时均为有限正数；遗漏、重复、乱序或无效记录会导致失败。
示例中三个合成样本的中位数为 `4.0`。列名可以通过命令行参数指定。

## 计算时间窗口积分

```sh
python3 -B tools/integrate_power.py examples/synthetic_power.csv \
  --start 0.25 --end 1.25 --max-gap 1
```

输入列为 `time_s,power_w`，单位分别为秒和瓦。相邻样本的中点划分保持区间，
工具在指定窗口内积分，不向首末时间戳之外外推。示例窗口的合成能量为
`3.5 J`，完整示例区间为 `6.0 J`；两者均为合成数值。
时间戳必须严格递增，功率必须非负且有限，窗口须被样本完整覆盖。
`--max-gap` 可检查最大采样间隔。计算结果取决于调用方的单位、校准和采样质量。

## 使用项目支撑模块

这些模块从项目已有支撑代码中抽取，公开版本可以单独导入。

```python
from runtime_support import FrameReader, FrameWriter, parse_output, validate_calls

writer = FrameWriter(b"EXAMPLE")
writer.put("I", 7)
writer.text("hello")
reader = FrameReader(writer.done(), b"EXAMPLE")
assert reader.get("I") == 7
assert reader.text() == "hello"
reader.finish()

specs = {"echo": {"text": {"type": "string", "required": True}}}
value, error, used_fallback = parse_output(
    '[{"name": "echo", "parameters": {"text": "hello"}}]'
)
assert error is None
assert validate_calls(value, specs) == []
```

`LineProcess(argv)` 通过标准输入输出与子进程逐行通信，调用方提供命令和响应前缀。
可用 `request("PING", ("PONG",), timeout=1)` 发送并等待匹配响应；
超时或输出结束返回 `None`。使用 `with LineProcess(...)` 会在退出时关闭输入、
等待进程退出，必要时终止并回收直接子进程。通信方法由单一调用者顺序使用。
调用方可以显式指定诊断文件路径，默认不创建文件。

编解码器的标记和字段布局由调用者提供，字符串长度以 UTF-8 字节计算。
输出验证器只检查提供的参数规范；调用方可依据 `used_fallback` 决定是否接受
从说明文字中抽出的 JSON。结构通过不等同于函数已执行。

## 工程核验

`engineering-work` 分支以源码、提交差异及可复现测试展示工程交付。
支撑模块与工具覆盖进程通信、输入输出处理及测量流程；单元和集成测试
验证正常链路与错误输入。当前共 47 个测试，可通过上述 `make test` 入口运行，
持续集成结果见仓库 Actions 页面。

本分支展示容量管理这一项独立切面及通用支撑代码。其余研究机制、
专用集成与原始数据保留在本地；
提交日期表示公开整理或后续维护的时间，不用于还原原项目的历史开发日期。
使用说明见 [NOTICE.md](NOTICE.md)。
