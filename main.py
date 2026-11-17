"""Forecast launch-day Twitch viewers hour by hour.

Reads one JSON case per .in file in MAKEFUN_INPUT_DIR and writes one
.out forecast per case into MAKEFUN_OUTPUT_DIR. The forecast is a
ridge regression over past launch curves: each comparable launch
contributes the log-ratio of its hour-h viewers to its hour-1 viewers,
regressed on three standardised launch features (plus an optional
diurnal term), in four weighted/unweighted variants whose log-quantiles
are averaged. Standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

ZS = (-1.6448536269514722, -0.6744897501960817, 0.0,
      0.6744897501960817, 1.6448536269514722)
TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
LOW = 0.01
HIGH = 1000000000.0


def parse_time(text):
    """Parse an ISO 8601 timestamp (with Z or an offset) to an aware datetime."""
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def safe_log(value):
    """Natural log, floored at 1.0 so it is always defined."""
    return math.log(max(float(value), 1.0))


def percentile(sorted_values, p):
    """Linear-interpolation percentile of an already sorted list."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = (n - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def solve_linear(matrix, rhs):
    """Solve a small dense system by Gauss-Jordan with partial pivoting.

    Columns whose best pivot is smaller than 1e-12 in absolute value are
    skipped, and a coefficient whose final diagonal is near zero is 0.0.
    """
    n = len(rhs)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = -1
        best = 0.0
        for row in range(col, n):
            if abs(aug[row][col]) > best:
                best = abs(aug[row][col])
                pivot = row
        if pivot < 0 or best < 1e-12:
            continue
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    out = []
    for i in range(n):
        if abs(aug[i][i]) < 1e-12:
            out.append(0.0)
        else:
            out.append(aug[i][n] / aug[i][i])
    return out


def hourly_points_from_minutes(minute_rows):
    """Aggregate minute reads into whole-hour means, sorted by hour."""
    buckets = {}
    for row in minute_rows or []:
        try:
            when = parse_time(row["at"])
            viewers = float(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        bucket = when.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(bucket, []).append(viewers)
    points = []
    for bucket in sorted(buckets):
        mean_viewers = sum(buckets[bucket]) / len(buckets[bucket])
        points.append((bucket, mean_viewers))
    return points


def target_features(points, day_start, hour_one):
    """Three launch features: log hour-one, pre-release drift, weekly level."""
    log_one = safe_log(hour_one)
    ordered = sorted(points)
    last = None
    for when, viewers in ordered:
        if day_start - timedelta(hours=48) <= when < day_start:
            last = viewers
    if last is None:
        second = 0.0
    else:
        second = safe_log(last) - log_one
    week = [v for when, v in ordered
            if day_start - timedelta(days=7) <= when < day_start]
    if week:
        third = safe_log(sum(week) / len(week)) - log_one
    else:
        third = safe_log(1.0) - log_one
    return (log_one, second, third)


def diurnal_profile(points, day_start):
    """Centred mean log viewers per UTC hour over the week before launch."""
    window = [(when, v) for when, v in points
              if day_start - timedelta(days=7) <= when < day_start]
    if not window:
        return [0.0] * 24
    logs = [safe_log(v) for _, v in window]
    overall = sum(logs) / len(logs)
    per_hour = [[] for _ in range(24)]
    for when, v in window:
        per_hour[when.hour].append(safe_log(v))
    profile = []
    for h in range(24):
        if per_hour[h]:
            profile.append(sum(per_hour[h]) / len(per_hour[h]) - overall)
        else:
            profile.append(0.0)
    return profile


def build_pool(case, target_name, day_start, hour_one):
    """Filter comparables and keep each one's curve, features and profile."""
    pool = []
    comparables = sorted(case.get("comparables") or [],
                         key=lambda c: c.get("name") or "")
    target_lower = (target_name or "").lower()
    for comp in comparables:
        unlock = str(comp.get("unlock") or "").lower()
        if "target-history" in unlock or "tier: weak" in unlock:
            continue
        comp_name = str(comp.get("name") or "")
        comp_lower = comp_name.lower()
        if target_lower and (target_lower == comp_lower
                             or target_lower in comp_lower):
            continue
        try:
            comp_start = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        by_time = {}
        rows = []
        for row in comp.get("hourly") or []:
            try:
                when = parse_time(row["from"])
                viewers = float(row["viewers"])
            except (KeyError, TypeError, ValueError):
                continue
            by_time[when] = viewers
            rows.append((when, viewers))
        curve = []
        for hour in range(1, 25):
            curve.append(by_time.get(comp_start + timedelta(hours=hour - 1)))
        hour1 = curve[0]
        if hour1 is None or hour1 <= 0:
            continue
        present = [v for v in curve if v is not None]
        if present and max(present) / hour1 > 20.0:
            continue
        features = target_features(rows, comp_start, hour1)
        profile = diurnal_profile(rows, comp_start)
        pool.append({"curve": curve, "features": features,
                     "profile": profile, "hour_one": hour1})
    return pool


def fallback_quantiles(hour_one, hour):
    """Baseline forecast: hour-one print with a widening log spread."""
    spread = 0.15 * math.sqrt(hour - 1)
    return [hour_one * math.exp(z * spread) for z in ZS]


def fit_quantiles(pool, target_row, weights, hour_one, hour, diurnal=False):
    """One ridge fit for a single hour; returns five quantiles or None."""
    rows = []
    ys = []
    row_weights = []
    for idx, entry in enumerate(pool):
        value = entry["curve"][hour - 1]
        if value is None:
            continue
        rows.append(entry["row_diurnal"] if diurnal else entry["row"])
        ys.append(safe_log(value) - safe_log(entry["curve"][0]))
        row_weights.append(weights[idx])
    if not rows:
        return None
    width = len(rows[0])
    normal = [[0.0] * width for _ in range(width)]
    rhs = [0.0] * width
    for i, row in enumerate(rows):
        w = row_weights[i]
        for a in range(width):
            rhs[a] += w * row[a] * ys[i]
            for b in range(width):
                normal[a][b] += w * row[a] * row[b]
    for a in range(width):
        normal[a][a] += 1.0
    coef = solve_linear(normal, rhs)
    centre = sum(c * x for c, x in zip(coef, target_row))
    residuals = sorted(ys[i] - sum(c * x for c, x in zip(coef, rows[i]))
                       for i in range(len(rows)))
    spread = max((percentile(residuals, 0.75)
                  - percentile(residuals, 0.25)) / 1.349, 0.05)
    return [hour_one * math.exp(centre + z * spread) for z in ZS]


def combine_fits_and_shrink(fit_values, hour_one, hour):
    """Average the four fits in log space, then shrink the late-hour spread."""
    combined = []
    for k in range(5):
        log_mean = sum(math.log(max(fit_values[f][k], 1e-9))
                       for f in range(4)) / 4.0
        combined.append(math.exp(log_mean))
    frac = (safe_log(hour_one) - 10.5) / (12.5 - 10.5)
    frac = min(1.0, max(0.0, frac))
    factor = 1.0 + (-0.5) * frac * (hour - 2) / 22.0
    median = combined[2]
    log_median = math.log(max(median, 1e-9))
    shrunk = []
    for q in combined:
        shrunk.append(math.exp(log_median
                               + (math.log(max(q, 1e-9)) - log_median)
                               * factor))
    for i in range(1, 5):
        if shrunk[i] < shrunk[i - 1]:
            shrunk[i] = shrunk[i - 1]
    return shrunk


def regression_quantiles(pool, features, profile, hour_one, hour):
    """Four ridge fits (plain/size-weighted x plain/diurnal) for one hour.

    Standardises the three features on the pool, builds the four design
    variants, fits each by ridge regression, and combines their quantiles
    in log space with the late-hour spread shrink. Returns None if any
    fit has no usable rows for this hour.
    """
    means = []
    stds = []
    for f in range(3):
        values = [entry["features"][f] for entry in pool]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var)
        means.append(mean)
        stds.append(std if std != 0.0 else 1.0)
    target_std = [1.0] + [(features[f] - means[f]) / stds[f]
                          for f in range(3)]
    log_one = safe_log(hour_one)
    for entry in pool:
        std_feats = [(entry["features"][f] - means[f]) / stds[f]
                     for f in range(3)]
        entry["row"] = [1.0] + std_feats
        entry["row_diurnal"] = [1.0] + std_feats + \
            [entry["profile"][hour - 1]]
    weights_plain = [1.0] * len(pool)
    weights_size = [math.exp(-0.5 * ((safe_log(entry["hour_one"])
                                       - log_one) / 1.0) ** 2)
                    for entry in pool]
    target_diurnal = [1.0] + target_std[1:] + [profile[hour - 1]]
    fits = [
        fit_quantiles(pool, target_std, weights_plain, hour_one, hour),
        fit_quantiles(pool, target_std, weights_size, hour_one, hour),
        fit_quantiles(pool, target_diurnal, weights_plain, hour_one, hour,
                      diurnal=True),
        fit_quantiles(pool, target_diurnal, weights_size, hour_one, hour,
                      diurnal=True),
    ]
    if any(f is None for f in fits):
        return None
    return combine_fits_and_shrink(fits, hour_one, hour)


def forecast_case(case):
    """Forecast hours 2..24 for one case dict."""
    known = case.get("known") or []
    if known:
        hour_one = max(float(known[0].get("viewers") or 0), 1.0)
    else:
        hour_one = 1.0
    day_start = parse_time(case["day_start"])
    hourly_rows = case.get("history_hourly") or []
    if hourly_rows:
        points = []
        for row in hourly_rows:
            try:
                points.append((parse_time(row["from"]),
                               float(row["viewers"])))
            except (KeyError, TypeError, ValueError):
                continue
    else:
        points = hourly_points_from_minutes(case.get("history") or [])
    game = case.get("game") or {}
    target_name = game.get("name") or ""
    features = target_features(points, day_start, hour_one)
    profile = diurnal_profile(points, day_start)
    pool = build_pool(case, target_name, day_start, hour_one)
    hours_out = []
    for hour in range(2, 25):
        if len(pool) < 3:
            quantiles = fallback_quantiles(hour_one, hour)
        else:
            quantiles = regression_quantiles(pool, features, profile,
                                             hour_one, hour)
            if quantiles is None:
                quantiles = fallback_quantiles(hour_one, hour)
        cleaned = []
        for q in quantiles:
            value = round(float(q), 2)
            value = min(HIGH, max(LOW, value))
            cleaned.append(value)
        for i in range(1, 5):
            if cleaned[i] < cleaned[i - 1]:
                cleaned[i] = cleaned[i - 1]
        hours_out.append({"hour": hour, "q": cleaned})
    return {"hours": hours_out}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    names = sorted(name for name in os.listdir(input_dir)
                   if name.endswith(".in"))
    if not names:
        print("no .in files in input directory", file=sys.stderr)
        return 1
    os.makedirs(output_dir, exist_ok=True)
    for name in names:
        with open(os.path.join(input_dir, name), "r",
                  encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(output_dir, out_name), "w",
                  encoding="utf-8") as handle:
            json.dump(result, handle)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
