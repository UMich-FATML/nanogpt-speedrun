#!/usr/bin/env python
"""Summarize Track 3 repro logs.

The training scripts write one log file per torchrun invocation and may contain
many trials. Each trial starts with a step:0 validation line.
"""

from __future__ import annotations

import argparse
import glob
import math
import re
from collections import defaultdict
from pathlib import Path


VAL_RE = re.compile(r"step:(?P<step>\d+)/(?P<total>\d+) val_loss:(?P<loss>\d+\.\d+)")


def iter_trials(paths: list[str]):
    for pattern in paths:
        for path_str in sorted(glob.glob(pattern)):
            path = Path(path_str)
            trial: dict[int, float] = {}
            for line in path.read_text().splitlines():
                match = VAL_RE.search(line)
                if not match:
                    continue
                step = int(match.group("step"))
                loss = float(match.group("loss"))
                if step == 0 and trial:
                    yield path, trial
                    trial = {}
                trial[step] = loss
            if trial:
                yield path, trial


def summarize(trials: list[tuple[Path, dict[int, float]]], steps: list[int]) -> None:
    by_step: dict[int, list[float]] = defaultdict(list)
    for _, trial in trials:
        for step in steps:
            if step in trial:
                by_step[step].append(trial[step])

    print(f"trials_detected: {len(trials)}")
    print("step,n,mean,std,stderr,precision_margin")
    for step in steps:
        losses = by_step.get(step, [])
        if not losses:
            print(f"{step},0,,,,")
            continue
        mean = sum(losses) / len(losses)
        if len(losses) > 1:
            var = sum((loss - mean) ** 2 for loss in losses) / (len(losses) - 1)
            std = math.sqrt(var)
        else:
            std = float("nan")
        stderr = std / math.sqrt(len(losses)) if len(losses) > 1 else float("nan")
        precision_margin = (3.28 - mean) * math.sqrt(len(losses))
        print(
            f"{step},{len(losses)},{mean:.5f},{std:.5f},"
            f"{stderr:.5f},{precision_margin:.5f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="Log files or glob patterns")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[3125, 3175, 3200, 3225, 3250, 3300],
        help="Validation steps to summarize",
    )
    args = parser.parse_args()
    summarize(list(iter_trials(args.logs)), args.steps)


if __name__ == "__main__":
    main()
