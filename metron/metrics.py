# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict
from dateutil import parser


def get_timestamp(trace: Dict) -> str:
    generation_start_ts = None

    for point in trace:
        if point["metric"] == "total_requests_sent" and "time" in point["data"]:
            current_ts = parser.parse(point["data"]["time"])

            if generation_start_ts is None or current_ts < generation_start_ts:
                generation_start_ts = current_ts

    return str(generation_start_ts)


def get_total_duration(summary: Dict) -> float:
    tokens = summary["metrics"].get("total_generated_tokens", {})
    rate = tokens.get("rate", 0)
    count = tokens.get("count", 0)

    # A run where every request failed generates zero tokens, so the rate is
    # zero. Returning 0 keeps postprocessing alive to report the failures (via
    # run_health) instead of dividing by zero and crashing on them.
    return count / rate if rate else 0.0


def calculate_common_throughput(summary: Dict) -> float:
    return summary["metrics"].get("total_generated_tokens", {}).get("rate", 0.0)


def calculate_common_requests(summary: Dict) -> float:
    passes_count = summary["root_group"]["checks"]["is status 200"]["passes"]
    total_generation_time = get_total_duration(summary)
    if not total_generation_time:
        return 0.0
    return (passes_count * 60) / total_generation_time


def run_health(summary: Dict) -> list:
    """Reasons the percentiles may not mean what they look like: dropped
    iterations (the schedule slipped) and non-200s (excluded from the tail)."""
    warnings = []
    metrics = summary.get("metrics", {})

    dropped = metrics.get("dropped_iterations", {}).get("count", 0)
    if dropped:
        warnings.append(
            f"{int(dropped)} iterations dropped: the arrival rate could not "
            f"be met, so the client stopped sending on schedule and these "
            f"percentiles understate the tail (coordinated omission). Raise "
            f"--max-vus or lower the rate."
        )

    empty = metrics.get("responses_without_content", {}).get("count", 0)
    if empty:
        warnings.append(
            f"{int(empty)} requests returned a success status with no content: "
            f"they generated nothing and contribute zero tokens, so throughput "
            f"is computed over the requests that produced output."
        )

    estimated = metrics.get("token_counts_estimated", {}).get("count", 0)
    if estimated:
        warnings.append(
            f"tokens for {int(estimated)} requests were counted from response "
            f"chunks because the server sent no usage block. An engine that "
            f"packs several tokens into one chunk undercounts here; see "
            f"itl_per_token for the corrected interval."
        )

    incomplete = metrics.get("incomplete_streams", {}).get("count", 0)
    if incomplete:
        warnings.append(
            f"{int(incomplete)} streams ended without [DONE] or a finish "
            f"reason: those responses may be truncated, and a truncated "
            f"response looks faster than a complete one."
        )

    checks = summary.get("root_group", {}).get("checks", {})
    status = checks.get("is status 200", {})
    fails = status.get("fails", 0)
    if fails:
        passes = status.get("passes", 0)
        warnings.append(
            f"{int(fails)} of {int(fails) + int(passes)} requests were not 200: "
            f"failures do not appear in the latency percentiles, which are "
            f"therefore computed over the survivors only."
        )
    return warnings
