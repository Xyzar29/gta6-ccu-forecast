#!/usr/bin/env python3
"""Launch-day Twitch CCU forecaster for Grand Theft Auto VI.

Reads one makefun/twitch-ccu-case/v1 case (a release day cut at the end of
its first hour) and writes, for hours 2..24, the 5/25/50/75/95% quantiles
of that hour's concurrent viewers in the game's Twitch category.

How it forecasts
----------------
Every past big launch in the case's comparables is reduced to one curve:
the log ratio of each release-day hour's viewers to that launch's first
hour.  Four ridge regressions are fitted per hour on those curves and
their forecasts are averaged in log space with equal weights:

1. a global ridge regression of the log ratio on three features of the
   launch that are known at the checkpoint -- the first hour's level, how
   far the last pre-release hour sits above that first hour, and how the
   week's pre-release mean sits against it;
2. the same regression re-weighted by a Gaussian kernel on the first
   hour's level, so launches of a similar size dominate;
3. the global regression with one more per-hour feature: the category's
   own pre-release diurnal profile at that hour (mean log viewers by UTC
   hour over the seven days before release, minus the week's mean), which
   carries the shape of the day -- when the audience is awake and when
   it sleeps -- and which the first two fits cannot see;
4. the same diurnal regression kernel-weighted as in (2).

The first hour's level is the dominant signal: launches that start small
ramp several-fold over the day as regions unlock, launches that start
huge stay almost flat.  The pre-release features measure how depressed
the first hour is against the category's own baseline just before
midnight, which is what separates a quiet first hour from a small launch.

Only rolling-midnight launches enter the fit: a comparable is dropped
when its peak is more than twenty times its first hour (a simultaneous
global unlock, whose first hour is pre-unlock trickle) or when it is
marked a weak analogue.  Launches marked early-access contaminated are
kept: their release-day curves are flat because play began days earlier,
which is the same direction the scale effect points for a huge launch,
and they anchor the high first-hour end of the fit where this target
sits.

Spread
------
Each fit's spread is the interquartile range of its own residuals,
floored at 0.05 in log space.  On top of that, the late-hour spread is
shrunk for big launches: big launches' late-day curves are tight (the
day saturates), while small launches stay wild late into the day, so the
shrink scales with the target's first-hour level -- zero below a first
hour of about 36 thousand viewers, full above about 268 thousand, linear
in log space between the two.  The shrink factor at hour h is
1 - 0.5 * frac * (h - 2) / 22.

Standard library only, no network, no randomness: the same case always
produces the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
HOURS = range(2, 25)

# Peak-to-first-hour ratio above which a launch is treated as a
# simultaneous global unlock (its first hour is pre-unlock trickle).
SIMULTANEOUS_CUT = 20.0
KERNEL_BW = 1.0
RIDGE = 1.0

# Size gate for the late-hour spread shrink, in log first-hour viewers.
SHRINK_LO = 10.5
SHRINK_HI = 12.5
SHRINK_SLOPE = -0.5


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def safe_log(value):
    return math.log(max(float(value), 1.0))


def percentile(sorted_values, prob):
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = (n - 1) * prob
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def solve_linear(matrix, rhs, ridge):
    """Solve a small dense linear system by Gauss-Jordan elimination."""
    size = len(matrix)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for i in range(size):
        aug[i][i] += ridge
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        if abs(piv) < 1e-12:
            continue
        for row in range(size):
            if row != col and aug[row][col] != 0.0:
                factor = aug[row][col] / piv
                for c in range(col, size + 1):
                    aug[row][c] -= factor * aug[col][c]
    out = []
    for i in range(size):
        out.append(aug[i][size] / aug[i][i] if abs(aug[i][i]) > 1e-12 else 0.0)
    return out


def launch_features(hour_one, history_points, day_start):
    """Features of one launch, all knowable at its checkpoint.

    history_points: hourly rows {from, viewers} strictly before day_start.
    Returns (log hour-one viewers, log last-hour ratio, log week-mean ratio).
    """
    before = sorted(
        (p for p in history_points if parse_time(p["from"]) < day_start),
        key=lambda p: p["from"],
    )
    week_ago = day_start - timedelta(days=7)
    week = [p["viewers"] for p in before if parse_time(p["from"]) >= week_ago]
    week_mean = sum(week) / len(week) if week else 1.0
    two_days_ago = day_start - timedelta(hours=48)
    recent = [p for p in before if parse_time(p["from"]) >= two_days_ago]
    last = recent[-1]["viewers"] if recent else hour_one
    return (
        safe_log(hour_one),
        safe_log(last) - safe_log(hour_one),
        safe_log(week_mean) - safe_log(hour_one),
    )


def diurnal_profile(history_points, day_start):
    """The category's pre-release daily rhythm, per UTC hour.

    For each UTC hour, the mean log viewers over the seven days before
    the release day, minus the mean over all those hours.  Hours with no
    reads take 0.0 (the week's mean).  This is the shape of the audience's
    day: a launch's release-day curve follows it, because viewers watch
    when they are awake and the regions unlock through the day.
    """
    week_ago = day_start - timedelta(days=7)
    by_hour = {hour: [] for hour in range(24)}
    for point in history_points:
        stamp = parse_time(point["from"])
        if week_ago <= stamp < day_start:
            by_hour[stamp.hour].append(safe_log(point["viewers"]))
    all_values = [v for hour in range(24) for v in by_hour[hour]]
    mean = sum(all_values) / len(all_values) if all_values else 0.0
    profile = []
    for hour in range(24):
        values = by_hour[hour]
        profile.append(sum(values) / len(values) - mean if values else 0.0)
    return profile


def release_curve(hourly_rows, day_start):
    """Hourly viewers for hours 1..24 of a launch day, None where missing."""
    by_time = {}
    for row in hourly_rows:
        by_time[parse_time(row["from"])] = row["viewers"]
    curve = []
    for hour in range(1, 25):
        curve.append(by_time.get(day_start + timedelta(hours=hour - 1)))
    return curve


def comparable_pool(case):
    """Rolling-midnight past launches usable as analogues for the target."""
    target_name = (case.get("game") or {}).get("name", "").lower()
    pool = []
    for comp in sorted(case.get("comparables") or [], key=lambda c: c.get("name", "")):
        notes = (comp.get("unlock") or "").lower()
        if "target-history" in notes:
            continue
        if "tier: weak" in notes:
            continue
        name = (comp.get("name") or "").lower()
        if target_name and (name == target_name or target_name in name):
            continue
        comp_day = parse_time(comp["day_start"])
        hourly = comp.get("hourly") or []
        curve = release_curve(hourly, comp_day)
        first = curve[0]
        if first is None or first <= 0:
            continue
        peak = max((v for v in curve if v is not None), default=None)
        if peak is None or peak / first > SIMULTANEOUS_CUT:
            continue
        feats = launch_features(first, hourly, comp_day)
        pool.append({
            "curve": curve,
            "features": feats,
            "diurnal": diurnal_profile(hourly, comp_day),
        })
    return pool


def fit_forecast(pool, target_features, target_diurnal, hour_one, weighted,
                 use_diurnal):
    """One forecaster: per-hour ridge regression of log ratios.

    weighted: when True, rows are kernel-weighted by closeness of the
    launch's first-hour level to the target's (local-linear fit).
    use_diurnal: when True, each hour's fit also takes the category's
    diurnal profile at that hour as a fourth, unstandardised feature.
    Returns {hour: [q05, q25, q50, q75, q95]} in viewer counts.
    """
    target_log_first = target_features[0]
    # Standardise the three features on the pool, as the fits expect.
    means = []
    scales = []
    for k in range(3):
        xs = [entry["features"][k] for entry in pool]
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        means.append(mean)
        scales.append(math.sqrt(var) or 1.0)

    def design(entry, hour):
        row = [1.0] + [
            (entry["features"][k] - means[k]) / scales[k] for k in range(3)
        ]
        if use_diurnal:
            row.append(entry["diurnal"][hour - 1])
        return row

    def weight(entry):
        if not weighted:
            return 1.0
        diff = entry["features"][0] - target_log_first
        return math.exp(-0.5 * (diff / KERNEL_BW) ** 2)

    target_row = [1.0] + [
        (target_features[k] - means[k]) / scales[k] for k in range(3)
    ]
    forecast = {}
    for hour in HOURS:
        target_x = target_row + ([target_diurnal[hour - 1]] if use_diurnal else [])
        rows = []
        for entry in pool:
            value = entry["curve"][hour - 1]
            if value is None:
                continue
            ratio = safe_log(value) - safe_log(entry["curve"][0])
            rows.append((design(entry, hour), ratio, weight(entry)))
        dim = len(target_x)
        normal = [[0.0] * dim for _ in range(dim)]
        rhs = [0.0] * dim
        for x, y, w in rows:
            for a in range(dim):
                rhs[a] += w * x[a] * y
                for b in range(dim):
                    normal[a][b] += w * x[a] * x[b]
        beta = solve_linear(normal, rhs, RIDGE)
        centre = sum(beta[t] * target_x[t] for t in range(dim))
        residuals = sorted(
            y - sum(beta[t] * x[t] for t in range(dim)) for x, y, _ in rows
        )
        spread = max(
            (percentile(residuals, 0.75) - percentile(residuals, 0.25)) / 1.349,
            0.05,
        )
        forecast[hour] = [
            hour_one * math.exp(centre + z * spread) for z in Z
        ]
    return forecast


def house_fallback(hour_one):
    """Persistence with a widening spread, used when the pool is too small."""
    out = {}
    for hour in HOURS:
        spread = 0.15 * math.sqrt(hour - 1)
        out[hour] = [hour_one * math.exp(z * spread) for z in Z]
    return out


def shrink_fraction(hour_one):
    """How much of the late-hour spread shrink a launch of this size gets.

    Zero for small launches, full for huge ones, linear in log space in
    between.  Big launches' late-day curves are tight; small launches
    stay wild, so the shrink must not apply to them.
    """
    log_first = safe_log(hour_one)
    frac = (log_first - SHRINK_LO) / (SHRINK_HI - SHRINK_LO)
    return min(max(frac, 0.0), 1.0)


def forecast_case(case):
    known = case.get("known") or [{}]
    hour_one = max(float(known[0].get("viewers") or 0), 1.0)
    day_start = parse_time(case["day_start"])
    history = case.get("history_hourly") or []
    if not history:
        # Fall back to aggregating minute reads into whole-hour means.
        buckets = {}
        for point in case.get("history") or []:
            stamp = parse_time(point["at"])
            key = stamp.replace(minute=0, second=0, microsecond=0)
            buckets.setdefault(key, []).append(point["viewers"])
        history = [
            {"from": key.isoformat().replace("+00:00", "Z"),
             "viewers": sum(v) / len(v)}
            for key, v in sorted(buckets.items())
        ]
    target_features = launch_features(hour_one, history, day_start)
    target_diurnal = diurnal_profile(history, day_start)
    pool = comparable_pool(case)
    if len(pool) < 3:
        combined = house_fallback(hour_one)
    else:
        fits = [
            fit_forecast(pool, target_features, target_diurnal, hour_one,
                         weighted=False, use_diurnal=False),
            fit_forecast(pool, target_features, target_diurnal, hour_one,
                         weighted=True, use_diurnal=False),
            fit_forecast(pool, target_features, target_diurnal, hour_one,
                         weighted=False, use_diurnal=True),
            fit_forecast(pool, target_features, target_diurnal, hour_one,
                         weighted=True, use_diurnal=True),
        ]
        frac = shrink_fraction(hour_one)
        combined = {}
        for hour in HOURS:
            quantiles = []
            for k in range(5):
                log_mean = sum(
                    safe_log(fit[hour][k]) for fit in fits
                ) / len(fits)
                quantiles.append(math.exp(log_mean))
            # Shrink the late-hour spread for big launches, around the
            # median: the median itself does not move.
            if frac > 0.0:
                factor = 1.0 + SHRINK_SLOPE * frac * (hour - 2) / 22.0
                median_log = safe_log(quantiles[2])
                quantiles = [
                    math.exp(median_log + (safe_log(q) - median_log) * factor)
                    for q in quantiles
                ]
            for k in range(1, 5):
                quantiles[k] = max(quantiles[k], quantiles[k - 1])
            combined[hour] = quantiles
    rows = []
    for hour in HOURS:
        quantiles = [
            min(max(round(q, 2), 0.01), 1_000_000_000.0)
            for q in combined[hour]
        ]
        for k in range(1, 5):
            quantiles[k] = max(quantiles[k], quantiles[k - 1])
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(
        name for name in os.listdir(input_dir) if name.endswith(".in")
    )
    if not names:
        print("no .in files in %s" % input_dir, file=sys.stderr)
        return 1
    for name in names:
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
