# -*- coding: utf-8 -*-
"""Create a Chinese result guide for the already completed frozen-model test."""
import calendar
import json
from pathlib import Path
R=Path(__file__).resolve().parent
ROOT=R.parent
s=json.loads((R/'summary.json').read_text(encoding='utf8'))
labels={'neural_network':'神经网络','decision_tree':'决策树','linear_regression':'线性回归','time_series':'时间序列 AR(1)','historical_average_three_week':'三周历史均值'}
folders={'neural_network':'annual_neural','decision_tree':'annual_tree','linear_regression':'linear_regression','time_series':'time_series_forecast','historical_average_three_week':'historical_mean'}
text=['# JFK 2025 年下半年：五种方法最终测试','', '## 数据与评估口径','',f"- 测试期：2025 年 7 月 1 日至 12 月 31 日，共 {s['window_count']:,} 个预测时点，纽约当地时间。",'- 原始数据：TLC 黄色出租车记录中上客地点 PULocationID=132，已排除不属于源文件月份的时间异常记录。',f"- 下半年 24 小时原始上客行程共 {s['raw_pickups_total']:,} 趟。与每天 06:00–23:20 的每分钟预测窗口是不同统计口径；窗口高度重叠，不能把它们相加算总上客量。",'- 在 t 预测 (t+10 分钟, t+40 分钟] 的已记录上客趟数；若右边界为次日 00:00，排除次日午夜。模型输出截为非负，按最近整数取整，恰好半整数取偶。','- 固定已有 1–5 月模型参数和 6 月模型选择结果，不在 7–12 月重新拟合。滚动回放允许使用预测时刻前已发生的测试期上客信息；三周历史均值也随时间更新。不同方法使用的历史信息并非完全相同。','', '## 原始上客记录（每天 24 小时）','','| 月份 | 上客行程数 | 平均每天上客行程数 |','|---|---:|---:|']
for month,count in s['raw_pickups_by_month'].items():
 days=calendar.monthrange(2025,int(month[-2:]))[1]
 text.append(f'| {month} | {count:,} | {count/days:,.0f} |')
text+=['','## 整体测试表现','','MAE 是平均绝对误差；RMSE 对较大误差更敏感；Bias 是预测值减实际值的平均数，负数表示整体低估。单位均为趟/预测窗口。','','| 方法 | MAE | RMSE | Bias |','|---|---:|---:|---:|']
for k,v in s['methods'].items():
 a=v['overall'];text.append(f"| {labels[k]} | {a['mae_trips']:.2f} | {a['rmse_trips']:.2f} | {a['bias_trips']:.2f} |")
text+=['','## 按月 MAE','','| 方法 | 7月 | 8月 | 9月 | 10月 | 11月 | 12月 |','|---|---:|---:|---:|---:|---:|---:|']
for k,v in s['methods'].items():
 text.append('| '+labels[k]+' | '+' | '.join(f"{v['by_month'][f'2025-{m:02d}']['mae_trips']:.2f}" for m in range(7,13))+' |')
text+=['','## 工作日和周末 MAE','','| 方法 | 工作日 | 周末 |','|---|---:|---:|']
for k,v in s['methods'].items():
 a=v['by_day_type'];text.append(f"| {labels[k]} | {a['workday']['mae_trips']:.2f} | {a['weekend']['mae_trips']:.2f} |")
text+=['','工作日 137,412 个窗口，周末 54,132 个窗口；每个日期包含 1,041 个时点。','','## 按预测时刻分组的 MAE','','| 方法 | 06:00–11:59 | 12:00–17:59 | 18:00–23:20 |','|---|---:|---:|---:|']
for k,v in s['methods'].items():
 a=v['by_prediction_time'];text.append(f"| {labels[k]} | {a['06:00-11:59']['mae_trips']:.2f} | {a['12:00-17:59']['mae_trips']:.2f} | {a['18:00-23:20']['mae_trips']:.2f} |")
text+=['', '## 文件','', '- `monthly_test_mae.png`：逐月 MAE 对比。','- `daily_means_all_methods.png`：每日每个预测窗口平均实际值与平均预测值；这不是全天上客总量。','- `all_test_predictions.csv`：五种方法在全部测试时点的预测。','- `summary.json`：全部总体、按月、工作日与周末、时段的 MAE、RMSE、Bias、样本量及数据与模型哈希。','- 各方法文件夹中的 `test.py` 运行该方法测试；`results/test_metrics.json`、`results/test_predictions.csv`、`results/test_comparison.png` 是对应结果，图为 7 月 10 日示例，每 10 分钟抽样。','', '## 运行与核对','','在项目根目录运行：','','```bash','.venv/bin/python -B final_test_2025/test_all.py','```','','单独运行某个方法：`.venv/bin/python -B annual_neural/test.py`；其他方法在对应文件夹下同样使用 `test.py`。这些入口依赖本项目内的统一脚本、四个模型、统一数据及 JFK 月度 Parquet，并非独立压缩包。单独运行写回该方法的测试预测和指标；运行统一脚本还会更新全部比较图。','', '核对：先逐时点复现 6 月保存的五种预测；核对测试源文件哈希；逐时点从原始记录验证测试目标、最近 10/30/60 分钟计数、三周历史均值及日历编码。所有方法使用相同测试行。','','## 解读限制','','- 这次下半年结果已被查看，若依据它继续选择或调参，需另保留一个从未用于选择的新测试期。','- 多个连续预测窗口高度重叠，不能把 191,544 个时点当作独立实验；本报告未计算独立样本置信区间。','- 结果只覆盖 JFK 已记录的黄色出租车上客，不是旅客总数、未满足需求或等候时间；其他机场和其他年份尚未检验。','- 模拟假设测试时刻能实时获得上客记录，TLC 历史月度数据本身不能证明部署时可做到。']
(R/'README_下半年测试总结.md').write_text('\n'.join(text)+'\n',encoding='utf8')
for k,v in s['methods'].items():
 a=v['overall'];monthly=v['by_month']; day=v['by_day_type'];title=labels[k]
 lines=[f'# {title}：2025 年 7–12 月测试','',f"共 {v['test_rows']:,} 个预测时点；MAE **{a['mae_trips']:.2f}** 趟、RMSE **{a['rmse_trips']:.2f}** 趟、Bias **{a['bias_trips']:.2f}** 趟/窗口。",'', '| 月份 | 窗口数 | MAE | RMSE | Bias |','|---|---:|---:|---:|---:|']
 for m,e in monthly.items():lines.append(f"| {m} | {e['rows']:,} | {e['mae_trips']:.2f} | {e['rmse_trips']:.2f} | {e['bias_trips']:.2f} |")
 lines+=['',f"工作日 MAE {day['workday']['mae_trips']:.2f}，周末 MAE {day['weekend']['mae_trips']:.2f}。",'', '- `test_metrics.json`：总体、分月及工作日/周末指标。','- `test_predictions.csv`：逐时点实际值和预测值。','- `test_comparison.png`：7 月 10 日预测与实际值，每 10 分钟展示一个点；评分使用全部每分钟窗口。','- 本文件夹上一级 `test.py` 为入口，依赖项目根目录 `final_test_2025/test_all.py` 和原始数据。','','训练好的模型参数保持不变。历史均值每次使用过去三周可用的实际记录；其他使用最近数据的方法也滚动使用截至当时已发生的记录。预测目标是已记录 JFK 黄色出租车上客趟数；窗口高度重叠，不是独立实验。','', '请同时阅读项目 `final_test_2025/README_下半年测试总结.md`。']
 (ROOT/folders[k]/'results'/'README_测试结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print('Wrote summary and five method-specific result guides')
