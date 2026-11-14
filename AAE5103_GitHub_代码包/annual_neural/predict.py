"""Predict one 30-minute window from already observed JFK pickup counts."""
import argparse
import math
from datetime import datetime, timedelta
import numpy as np
import torch
from common import FEATURES, RESULTS, final_counts
from train import PickupNet


def predict(as_of, last_10, last_30, last_60, model_path=RESULTS / 'annual_model.pt'):
    if as_of.tzinfo is not None or as_of.second or as_of.microsecond:
        raise ValueError('Use a minute-aligned, timezone-naive New York local time')
    minute = as_of.hour * 60 + as_of.minute
    if not 360 <= minute <= 1400:
        raise ValueError('Prediction time must be within 06:00–23:20 local time')
    counts = (last_10, last_30, last_60)
    if any(not isinstance(x, (int, np.integer)) or x < 0 for x in counts) or not last_10 <= last_30 <= last_60:
        raise ValueError('Pickup counts must be nonnegative integers with 10 min ≤ 30 min ≤ 60 min')
    checkpoint = torch.load(model_path, map_location='cpu')  # Only load a trusted checkpoint.
    if checkpoint['features'] != FEATURES:
        raise ValueError('Feature schema mismatch')
    angle = 2 * math.pi * minute / 1440
    inputs = [*counts, math.sin(angle), math.cos(angle)] + [int(as_of.weekday() == d) for d in range(7)]
    x = np.asarray(inputs, dtype=np.float32)
    mean = np.asarray(checkpoint['x_mean'], dtype=np.float32)
    std = np.asarray(checkpoint['x_std'], dtype=np.float32)
    model = PickupNet()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    with torch.no_grad():
        raw = float(model(torch.from_numpy(((x - mean) / std).reshape(1, -1))).item() * checkpoint['y_std'] + checkpoint['y_mean'])
    end = as_of + timedelta(minutes=40)
    return {'as_of_local': as_of.isoformat(sep=' '), 'window_start_exclusive': (as_of + timedelta(minutes=10)).isoformat(sep=' '), 'window_end': end.isoformat(sep=' '), 'window_end_inclusive': end.date() == as_of.date(), 'raw_prediction': raw, 'forecast_pickups': int(final_counts(np.array([raw]))[0])}


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--time', required=True, help='New York local time, e.g. "2025-06-10 10:00"')
    p.add_argument('--last-10', type=int, required=True)
    p.add_argument('--last-30', type=int, required=True)
    p.add_argument('--last-60', type=int, required=True)
    args = p.parse_args()
    print(json.dumps(predict(datetime.fromisoformat(args.time), args.last_10, args.last_30, args.last_60), ensure_ascii=False, indent=2))
