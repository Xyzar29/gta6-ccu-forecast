#!/usr/bin/env python3
"""Forecast a game launch day's Twitch viewers, hours 2-24.

For every hour after the first, four ridge regressions are fitted over a
pool of comparable past launches.  Each fit predicts the log ratio of
that hour's viewers to the launch's own hour-1 viewers from a small set
of features describing the launch's size and its pre-release audience.
Two fits weight every comparable equally and two weight nearby-sized
launches more heavily; two fits in each pair also receive a diurnal
profile value describing the target category's usual shape at that
hour of day.  The four fits' quantiles are combined geometrically,
and the spread of late hours is shrunk for the largest launches.
"""

import glob
import json
import math
import os
from datetime import datetime, timedelta, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_VALUES = (
    -1.6448536269514722,
    -0.6744897501960817,
    0.0,
    0.6744897501960817,
    1.6448536269514722,
)
HOURS = list(range(2, 25))


def parse_time(text):
    """Parse an ISO 8601 timestamp into an aware datetime."""
    if text is None:
        return None
    value = str(text).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def percentile(sorted_values, fraction):
    """Linear-interpolation percentile of an already sorted list."""
    count = len(sorted_values)
    if count == 0:
        return 0.0
    if count == 1:
        return float(sorted_values[0])
    position = fraction * (count - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(
        sorted_values[upper]
    ) * weight


def clip(value, low, high):
    return max(low, min(high, value))


def safe_viewers(value):
    """Return a viewer count as a float, treating junk as zero."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


def build_features(h1, pre_points):
    """Features shared by the target and by every comparable.

    pre_points is a chronological list of (moment, viewers) pairs that
    all lie before the release day starts.  The returned mapping holds
    the raw feature values plus the diurnal profile for release hours.
    """
    base = h1 if h1 > 0 else 1.0
    logh1 = math.log(max(base, 1.0))
    if pre_points:
        last1 = pre_points[-1][1]
        tail = pre_points[-3:]
        last3 = sum(item[1] for item in tail) / len(tail)
    else:
        last1 = base
        last3 = base
    # last3 is part of the prepared record even though the design
    # below works with last1 and the seven-day mean.
    _ = last3
    log_last1_ratio = math.log(max(last1, 1.0) / max(base, 1.0))
    return {
        "h1": base,
        "logh1": logh1,
        "log_last1_ratio": log_last1_ratio,
        "pre_points": pre_points,
    }


def add_window_features(record, day_start):
    """Add the seven-day mean ratio and the diurnal profile."""
    base = record["h1"]
    window_start = day_start - timedelta(days=7)
    window = [
        (moment, viewers)
        for moment, viewers in record["pre_points"]
        if window_start <= moment < day_start
    ]
    if window:
        pre7mean = sum(viewers for _, viewers in window) / len(window)
    else:
        pre7mean = base
    record["log_pre7mean_ratio"] = math.log(
        max(pre7mean, 1.0) / max(base, 1.0)
    )
    diurnal = [0.0] * 25
    if window:
        logs = [math.log(max(viewers, 1.0)) for _, viewers in window]
        overall = sum(logs) / len(logs)
        by_hour = {}
        for (moment, _), log_value in zip(window, logs):
            by_hour.setdefault(moment.hour, []).append(log_value)
        means = {
            hour: sum(values) / len(values)
            for hour, values in by_hour.items()
        }
        for hour in range(2, 25):
            utc_hour = hour - 1
            if utc_hour in means:
                diurnal[hour] = means[utc_hour] - overall
    record["diurnal"] = diurnal
    return record


def prepare_target(case):
    """Build the feature record for the case being forecast."""
    day_start = parse_time(case.get("day_start"))
    known = case.get("known") or []
    h1 = 0.0
    if known:
        h1 = safe_viewers(known[0].get("viewers"))
    pre_points = []
    for row in case.get("history_hourly") or []:
        moment = parse_time(row.get("from"))
        if moment is None or day_start is None:
            continue
        if moment < day_start:
            pre_points.append((moment, safe_viewers(row.get("viewers"))))
    pre_points.sort(key=lambda item: item[0])
    record = build_features(h1, pre_points)
    if day_start is not None:
        add_window_features(record, day_start)
    else:
        record["log_pre7mean_ratio"] = record["log_last1_ratio"]
        record["diurnal"] = [0.0] * 25
    return record


def prepare_comparable(comp):
    """Build a comparable's curve and features, or None if unusable."""
    unlock = comp.get("unlock") or ""
    if "Tier: weak" in unlock:
        return None
    if "Tier: target-history" in unlock:
        return None
    day_start = parse_time(comp.get("day_start"))
    if day_start is None:
        return None
    hourly = []
    for row in comp.get("hourly") or []:
        moment = parse_time(row.get("from"))
        if moment is None:
            continue
        hourly.append((moment, safe_viewers(row.get("viewers"))))
    hourly.sort(key=lambda item: item[0])
    curve = {}
    for moment, viewers in hourly:
        delta = (moment - day_start).total_seconds()
        if delta % 3600 == 0 and 0 <= delta < 24 * 3600:
            curve[int(delta // 3600) + 1] = viewers
    if 1 not in curve:
        return None
    h1 = curve[1]
    base = h1 if h1 > 0 else 1.0
    if curve:
        peak_ratio = max(curve.values()) / base
        if peak_ratio > 20:
            return None
    pre_points = [
        (moment, viewers) for moment, viewers in hourly if moment < day_start
    ]
    record = build_features(h1, pre_points)
    record["curve"] = curve
    add_window_features(record, day_start)
    return record


def build_pool(case):
    pool = []
    for comp in case.get("comparables") or []:
        record = prepare_comparable(comp)
        if record is not None:
            pool.append(record)
    return pool


def solve_linear(matrix, vector):
    """Solve a small dense system by Gaussian elimination.

    Partial pivoting keeps the elimination stable; a singular pivot
    falls back to a tiny nudge so a forecast is always produced.
    """
    size = len(vector)
    rows = [list(matrix[i]) + [vector[i]] for i in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(rows[r][column]))
        if abs(rows[pivot][column]) < 1e-12:
            rows[pivot][column] = 1e-12
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
        pivot_value = rows[column][column]
        for row in range(column + 1, size):
            factor = rows[row][column] / pivot_value
            if factor == 0.0:
                continue
            for col in range(column, size + 1):
                rows[row][col] -= factor * rows[column][col]
    solution = [0.0] * size
    for index in range(size - 1, -1, -1):
        total = rows[index][size]
        for col in range(index + 1, size):
            total -= rows[index][col] * solution[col]
        solution[index] = total / rows[index][index]
    return solution


def ridge_fit(members, target, stats, hour, use_diurnal):
    """Fit one ridge model for one hour and return its quantiles.

    members is a list of (record, design, y, weight) tuples whose
    design vectors are already standardised for this hour.
    """
    if not members:
        return None
    width = len(members[0][1])
    gram = [[0.0] * width for _ in range(width)]
    rhs = [0.0] * width
    for _, design, y_value, weight in members:
        for i in range(width):
            rhs[i] += weight * design[i] * y_value
            for j in range(width):
                gram[i][j] += weight * design[i] * design[j]
    for i in range(width):
        gram[i][i] += 1.0
    beta = solve_linear(gram, rhs)
    target_design = make_design(target, stats, hour, use_diurnal)
    median_log_ratio = sum(b * x for b, x in zip(beta, target_design))
    residuals = sorted(
        y_value - sum(b * x for b, x in zip(beta, design))
        for _, design, y_value, _ in members
    )
    spread = (percentile(residuals, 0.75) - percentile(residuals, 0.25)) / 1.349
    sigma = max(spread, 0.05)
    quantiles = []
    for z_value in Z_VALUES:
        value = target["h1"] * math.exp(median_log_ratio + z_value * sigma)
        quantiles.append(value)
    return quantiles


def make_design(record, stats, hour, use_diurnal):
    means, stds = stats
    raw = (record["logh1"], record["log_last1_ratio"], record["log_pre7mean_ratio"])
    design = [1.0]
    for index, value in enumerate(raw):
        design.append((value - means[index]) / stds[index])
    if use_diurnal:
        design.append(record["diurnal"][hour])
    return design


def hour_members(pool, target, hour, use_diurnal, use_local):
    """Collect the fitted members for one hour with their designs.

    Returns a (members, stats) pair; members is empty when no pool
    record has this hour.
    """
    chosen = [record for record in pool if hour in record["curve"]]
    if not chosen:
        return [], None
    raw_rows = [
        (
            record["logh1"],
            record["log_last1_ratio"],
            record["log_pre7mean_ratio"],
        )
        for record in chosen
    ]
    means = []
    stds = []
    for index in range(3):
        column = [row[index] for row in raw_rows]
        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        std = math.sqrt(variance)
        means.append(mean)
        stds.append(std if std != 0 else 1.0)
    stats = (means, stds)
    members = []
    for record in chosen:
        design = make_design(record, stats, hour, use_diurnal)
        y_value = math.log(max(record["curve"][hour], 1.0) / record["h1"])
        if use_local:
            gap = (record["logh1"] - target["logh1"]) / 1.0
            weight = math.exp(-0.5 * gap * gap)
        else:
            weight = 1.0
        members.append((record, design, y_value, weight))
    return members, stats


def enforce_order(values):
    ordered = list(values)
    for index in range(1, len(ordered)):
        if ordered[index] < ordered[index - 1]:
            ordered[index] = ordered[index - 1]
    return ordered


def fallback_row(h1, hour):
    base = h1 if h1 > 0 else 1.0
    width = 0.15 * math.sqrt(hour - 1)
    return [base * math.exp(z_value * width) for z_value in Z_VALUES]


def base_forecast(target, pool):
    """Ensemble forecast with the size-conditioned spread shrink."""
    rows = {}
    if not pool:
        for hour in HOURS:
            rows[hour] = fallback_row(target["h1"], hour)
        return rows
    frac_s = clip((target["logh1"] - 10.5) / 2.0, 0.0, 1.0)
    for hour in HOURS:
        fits = []
        for use_diurnal in (False, True):
            for use_local in (False, True):
                members, stats = hour_members(
                    pool, target, hour, use_diurnal, use_local
                )
                if not members:
                    fits = []
                    break
                quantiles = ridge_fit(members, target, stats, hour, use_diurnal)
                if quantiles is None:
                    fits = []
                    break
                fits.append(quantiles)
            if not fits:
                break
        if len(fits) != 4:
            rows[hour] = fallback_row(target["h1"], hour)
            continue
        combined = []
        for level_index in range(5):
            mean_log = sum(math.log(fit[level_index]) for fit in fits) / len(fits)
            combined.append(math.exp(mean_log))
        combined = enforce_order(combined)
        median = combined[2]
        factor = 1.0 - 0.5 * frac_s * (hour - 2) / 22.0
        log_median = math.log(median)
        shrunk = [
            math.exp(log_median + (math.log(value) - log_median) * factor)
            for value in combined
        ]
        rows[hour] = enforce_order(shrunk)
    return rows


def band_for_hour(hour):
    if hour <= 6:
        return "early"
    if hour <= 12:
        return "mid"
    return "late"


def calibrate(target, pool, rows):
    """Leave-one-out calibration of the ensemble forecast.

    Every pool member is forecast from the rest of the pool, and its
    log residuals against its real curve are pooled into three hour
    bands.  Only large launches (log hour-1 at least 12.4) contribute.
    The target forecast is then shifted by half of each band's median
    residual, gated by how large the target itself is, and the mid
    band's top quantile is widened around its median.
    """
    band_values = {"early": [], "mid": [], "late": []}
    for member in pool:
        rest = [record for record in pool if record is not member]
        member_rows = base_forecast(member, rest)
        for hour in HOURS:
            if hour not in member["curve"]:
                continue
            residual = math.log(max(member["curve"][hour], 1.0)) - math.log(
                member_rows[hour][2]
            )
            if member["logh1"] >= 12.4:
                band_values[band_for_hour(hour)].append(residual)
    shifts = {}
    for band, values in band_values.items():
        if len(values) >= 8:
            shifts[band] = percentile(sorted(values), 0.5)
        else:
            shifts[band] = 0.0
    frac_t = clip((target["logh1"] - 11.5) / 1.5, 0.0, 1.0)
    adjusted = {}
    for hour in HOURS:
        band = band_for_hour(hour)
        factor = math.exp(0.5 * frac_t * shifts[band])
        values = [value * factor for value in rows[hour]]
        if band == "mid":
            log_median = math.log(values[2])
            log_top = math.log(values[4])
            widened = log_median + (log_top - log_median) * (1.0 + 0.6 * frac_t)
            values[4] = math.exp(widened)
        values = enforce_order(values)
        adjusted[hour] = [clip(value, 1e-6, 1e9) for value in values]
    return adjusted


def forecast_case(case):
    target = prepare_target(case)
    pool = build_pool(case)
    rows = base_forecast(target, pool)
    rows = calibrate(target, pool, rows)
    hours = []
    for hour in HOURS:
        values = []
        for value in rows[hour]:
            values.append(clip(float(value), 1e-6, 1e9))
        hours.append({"hour": hour, "q": enforce_order(values)})
    return {"hours": hours}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(input_dir, "*.in")))
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        case_id = case.get("id") or os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, str(case_id) + ".out")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
