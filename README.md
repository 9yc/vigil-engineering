# Vigil Engineering

用于展示 Vigil 项目通用工程工作的公开仓库，包含定时采样工具、CSV 数据汇总、
自动化测试和工程工作记录。公开源码可独立构建运行，无需原项目代码或数据。

| 交付 | 可直接核验的内容 |
|---|---|
| [采样工具](tools/sample_nodes.cpp) | 固定周期采样、双时钟时间戳、输入检查、退出与文件保护 |
| [CSV 汇总](tools/summarize_csv.py) | 指定数值列、分类汇总、中位数和范围、异常数据检查 |
| [自动化测试](tests/test_tools.py) | 合成文件采样到 CSV 统计的运行链路，以及错误处理 |
| [工程记录](reports/work_record.md) | 已有工程积累与本次公开整理的范围 |
| [项目规模快照](reports/project_inventory.json) | 完整本地项目的汇总文件数量和非空行数 |
| [验证记录](reports/validation.md) | 公开工具实际完成的验证及复现方法 |

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

## 工作量核验

公开源码、测试和 CI 可以直接检查。完整项目的规模快照只提供汇总数量，
不包含文件路径、实现结构或实验结果；它是项目维护方提供的静态盘点，
不等同于第三方审计、个人净新增代码量或功能验收结果。

完整项目中未公开部分的工作量可通过内部材料核验。研究核心实现及研究资料
未包含在本仓库。公开文件采用独立提交历史，后续更新记录实际发布和维护工作。
发布范围与使用说明见 [NOTICE.md](NOTICE.md)。
