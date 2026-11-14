"""Plot all five forecasts against recorded JFK pickups for a validation day."""
import csv
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/aae5103-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'results' / 'validation_predictions.csv'
OUTPUT = HERE / 'results' / 'all_five_methods_validation_2025-06-07.png'

with SOURCE.open(newline='', encoding='utf-8') as f:
    all_rows = list(csv.DictReader(f))
rows = []
for row in all_rows:
    time = datetime.fromisoformat(row['as_of_local'])
    if time.date().isoformat() == '2025-06-07' and time.minute % 10 == 0:
        rows.append((time, row))
if len(rows) != 105 or rows[0][0].strftime('%H:%M') != '06:00' or rows[-1][0].strftime('%H:%M') != '23:20':
    raise ValueError('Expected every 10 minutes from 06:00 to 23:20 on 7 June 2025')

fig, ax = plt.subplots(figsize=(16, 6))
series = [
    ('actual_pickups', 'Recorded pickups', '#243e60', '-', 2.5),
    ('neural_forecast', 'Neural network', '#a13370', '-', 1.8),
    ('tree_forecast', 'Decision tree', '#66853a', '--', 1.8),
    ('linear_forecast', 'Linear regression', '#a84e20', '-.', 1.8),
    ('time_series_forecast', 'Seasonal 10-min AR(1)', '#128caf', '-', 1.8),
    ('history_three_week', 'Three-week historical mean', '#eda039', '--', 1.9),
]
for field, label, color, style, width in series:
    ax.plot([t for t, _ in rows], [int(row[field]) for _, row in rows],
            color=color, linestyle=style, linewidth=width, label=label)
ax.set_title('JFK pickup forecasts: 7 June 2025 validation', fontsize=17)
ax.set_xlabel('Prediction time (New York local)', fontsize=13)
ax.set_ylabel('Pickups in (t+10, t+40] (trips)', fontsize=13)
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(6, 24, 2)))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.tick_params(labelsize=11)
ax.grid(True, alpha=0.25)
ax.legend(loc='upper left', ncol=2, fontsize=10, framealpha=0.94)
fig.tight_layout()
fig.savefig(OUTPUT, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'Saved {OUTPUT}; {len(rows)} shared prediction times; sampling interval 10 min.')
