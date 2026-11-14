"""Summarize all June validation windows by day, for five JFK forecasts."""
import csv
import math
import os
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/aae5103-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
SOURCE = RESULTS / 'validation_predictions.csv'
FIGURE = RESULTS / 'all_methods_full_june_validation.png'
DAILY = RESULTS / 'all_methods_full_june_daily_metrics.csv'
METHODS = [
    ('neural_forecast', 'Neural network', '#a13370', '-'),
    ('tree_forecast', 'Decision tree', '#66853a', '--'),
    ('linear_forecast', 'Linear regression', '#a84e20', '-.'),
    ('time_series_forecast', 'Seasonal 10-min AR(1)', '#128caf', '-'),
    ('history_three_week', 'Three-week historical mean', '#eda039', '--'),
]

by_date = defaultdict(list)
with SOURCE.open(newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        t = datetime.fromisoformat(row['as_of_local'])
        by_date[t.date()].append(row)
days = sorted(by_date)
assert days == [date(2025, 6, d) for d in range(1, 31)]
assert all(len(by_date[day]) == 1041 for day in days)
results = []
for day in days:
    rows = by_date[day]
    actual = [int(row['actual_pickups']) for row in rows]
    entry = {'date': day.isoformat(), 'window_count': len(rows), 'mean_recorded': sum(actual) / len(actual)}
    for field, _, _, _ in METHODS:
        forecasts = [int(row[field]) for row in rows]
        errors = [p - a for p, a in zip(forecasts, actual)]
        entry[field + '_mean'] = sum(forecasts) / len(forecasts)
        entry[field + '_mae'] = sum(abs(e) for e in errors) / len(errors)
        entry[field + '_rmse'] = math.sqrt(sum(e * e for e in errors) / len(errors))
    results.append(entry)
with DAILY.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)

fig, (top, bottom) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={'hspace': .18})
top.plot(days, [r['mean_recorded'] for r in results], label='Recorded pickups', color='#243e60', linewidth=2.5)
for field, label, color, linestyle in METHODS:
    top.plot(days, [r[field + '_mean'] for r in results], label=label, color=color, linestyle=linestyle, linewidth=1.8)
    bottom.plot(days, [r[field + '_mae'] for r in results], label=label, color=color, linestyle=linestyle, linewidth=1.8)
fig.suptitle('JFK pickup forecasts: June 2025 validation', fontsize=18, y=.985)
top.set_ylabel('Mean pickups per window (trips)', fontsize=12)
top.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=3, fontsize=9, framealpha=.95)
bottom.set_title('Daily mean absolute error (MAE)', fontsize=14, loc='left')
bottom.set_ylabel('MAE per window (trips)', fontsize=12)
bottom.set_xlabel('Date (New York local)', fontsize=12)
bottom.set_ylim(bottom=0)
bottom.set_xticks([date(2025, 6, day) for day in (1, 5, 10, 15, 20, 25, 30)])
bottom.xaxis.set_major_formatter(mdates.DateFormatter('%d Jun'))
bottom.set_xlim(days[0] - timedelta(days=0.5), days[-1] + timedelta(days=0.5))
for ax in (top, bottom):
    ax.grid(True, alpha=.25)
    ax.tick_params(labelsize=10)
fig.text(.5, .015, 'Each day: 1,041 prediction times, 06:00–23:20; each predicts pickups in (t+10, t+40].', ha='center', fontsize=10)
fig.subplots_adjust(left=.09, right=.98, top=.87, bottom=.09)
fig.savefig(FIGURE, dpi=190)
plt.close(fig)
print('Saved:', FIGURE)
print('Daily table:', DAILY)
for field,label,_,_ in METHODS:
    overall = sum(row[field + '_mae'] * row['window_count'] for row in results) / sum(row['window_count'] for row in results)
    print(f'{label}: full-June MAE {overall:.4f}')
