# Vigil Engineering

Vigil 项目通用工程代码的公开子集，包含定时采样、CSV 汇总、测量完整性检查、
时间窗口积分及自动化测试。公开源码可独立构建运行，无需原项目代码或数据。

| 交付 | 可直接核验的内容 |
|---|---|
| [采样工具](tools/sample_nodes.cpp) | 固定周期采样、双时钟时间戳、输入检查、退出与文件保护 |
| [CSV 汇总](tools/summarize_csv.py) | 指定数值列、分类汇总、中位数和范围、异常数据检查 |
| [测量完整性检查](tools/validate_measurements.py) | 样本总数、迭代顺序、缺失与重复记录、有效耗时 |
| [时间窗口积分](tools/integrate_power.py) | 采样间隔、窗口覆盖、功率积分及平均功率 |
| [自动化测试](tests/test_tools.py) | 合成文件采样到 CSV 统计的运行链路，以及错误处理 |

## 构建与测试

依赖 Linux、C++17 编译器、GNU Make 和 Python 3.8 及以上版本。Python 只使用标准库。
先把 `ARTIFACT_DIR` 设置为你选择的、位于源码目录以外的临时产物绝对路径，然后运行：

```sh
make test ARTIFACT_DIR="$ARTIFACT_DIR"
```

构建产物、测试输入和测试输出均写入该目录。测试使用本地合成文件，
无需设备、SDK、模型权重、账户凭据或网络连接。GitHub Actions 执行相同入口。

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
这些数值只用于演示，不代表实际设备性能。省略 `--group-by` 可汇总整列；
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
`3.5 J`，完整示例区间为 `6.0 J`；两者均不是实际设备测量结果。
时间戳必须严格递增，功率必须非负且有限，窗口须被样本完整覆盖。
`--max-gap` 可检查最大采样间隔。计算结果取决于调用方的单位、校准和采样质量。

## 工程核验

`engineering-work` 分支以源码、提交差异及可复现测试展示工程交付。
四项工具覆盖数据采集、样本完整性、统计与时间窗口计算；单元和集成测试
验证正常链路与错误输入。本分支共 23 个测试，可通过上述 `make test` 入口运行，
持续集成结果见仓库 Actions 页面。

本分支展示经过整理的通用代码子集。研究核心实现和研究资料保留在本地；
提交日期表示公开整理或后续维护的时间，不用于还原原项目的历史开发日期。
使用说明见 [NOTICE.md](NOTICE.md)。
