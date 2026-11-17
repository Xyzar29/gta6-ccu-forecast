# gta6-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Xyzar29/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Xyzar29/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A launch-day Twitch viewer forecaster for make.fun's "Forecast GTA VI's
launch-day Twitch viewers hour by hour" bounty. Given a case cut at the
end of release day's first hour, it writes 5/25/50/75/95% quantiles of
concurrent viewers for hours 2–24.

## How it works

Every past big launch in the case's comparables is reduced to one curve:
the log ratio of each release-day hour's viewers to that launch's first
hour. Two regressions are fitted per hour on those curves and averaged
in log space: a global ridge regression on three features knowable at
the checkpoint (first hour's level; how far the last pre-release hour
sits above it; how the week's pre-release mean sits against it), and
the same regression re-weighted by a Gaussian kernel on the first
hour's level, so launches of a similar size dominate.

The first hour's level is the dominant signal: launches that start
small ramp several-fold as regions unlock, launches that start huge
stay almost flat. The pre-release features measure how depressed the
first hour is against the category's own baseline just before midnight.

Only rolling-midnight launches enter the fit. A comparable is dropped
when its peak is more than twenty times its first hour (a simultaneous
global unlock, whose first hour is pre-unlock trickle) or when it is
marked a weak analogue. Early-access launches are kept: they anchor
the high first-hour end of the fit. The spread at each hour comes from
the interquartile range of the fit's residuals.

## Results on the practice cases

Scored locally with an exact replica of the official pinball loss on
the log scale, against the four practice launches' real release days
(ground truth from SullyGnome's public hourly averages, the same
source the kit uses — hour 1 matches the case files exactly):

- **Pinball loss: 0.149** over the four practice cases (Wolverine
  0.162, Red Dead Redemption 2 0.131, God of War Ragnarök 0.112,
  Spider-Man 2 0.192) — typical miss 24.1%, 90% coverage 0.90
- Leaderboard at entry time: top 0.134, 2nd 0.143, 3rd–5th 0.145–0.146,
  median 0.218, house baseline 0.387
- The model was selected by leave-one-out over 32 rolling-midnight
  launches (not just the four practice cases): global OLS, local-linear
  and their ensemble were A/B'd on the same scorer, and the ensemble
  won on both the practice four and the wider leave-one-out
- Deterministic: two runs on the same cases produce byte-identical
  output; runtime ~0.25 s for all four practice cases (limit 8 s)

Practice loss is only a proxy: the final is scored on GTA VI's real
November 19, and the entry must also beat the best house entry there.

## Run it

```sh
python3 main.py
```

Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`) and
writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default
`/output`). Python 3.13, standard library only, no network.

---

<sub>Entered on [make.fun](https://make.fun/?ref=86zc8pnt&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
