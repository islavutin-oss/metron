# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import argparse
import hashlib
import os
import shutil
import sys
import time
import json
import urllib.request
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
import socket

from .artifact import create_artifact, json_lines_to_common_json
from .metrics import run_health

__version__ = "0.1.0"

REPO = "https://github.com/islavutin-oss/metron"


def prepare_workload(args: argparse.ArgumentParser, path_to_artifact: str) -> None:
    from .workload import arg2generator

    prompt_generator = arg2generator(args.input_tokens_distribution)
    prompt_generator.prepare_json(path_to_artifact, sequence_type="prompts")

    output_lengths_generator = arg2generator(args.output_tokens_distribution)
    output_lengths_generator.prepare_json(path_to_artifact, sequence_type="generation")


def postprocess_results(path_to_trace: str, path_to_summary: str) -> None:
    from .metrics import (
        calculate_common_throughput,
        get_timestamp,
        get_total_duration,
        calculate_common_requests,
    )

    json_lines_to_common_json(path_to_trace)

    with open(path_to_trace, "r") as trace:
        trace_dict = json.load(trace)

    with open(path_to_summary, "r") as summary:
        summary_dict = json.load(summary)

    common_throughput = calculate_common_throughput(summary_dict)
    common_throughput_requests = calculate_common_requests(summary_dict)

    uuid = path_to_trace.split("/")[-2]
    summary_dict["experiment_id"] = uuid
    summary_dict["timestamp"] = get_timestamp(trace_dict)
    summary_dict["metrics"]["duration"] = get_total_duration(summary_dict)
    summary_dict["metrics"]["throughput"] = common_throughput

    with open(path_to_summary, "w") as summary:
        json.dump(summary_dict, summary, indent=4)

    print(f"Throughput out tokens = {common_throughput:.2f} tokens / s")
    print(f"Throughput requests = {common_throughput_requests:.2f} req / min")


def postprocess_args(args: argparse.ArgumentParser) -> List[argparse.ArgumentParser]:
    if args.run_id is None:
        args.run_id = hashlib.md5(os.urandom(16)).hexdigest()

    if args.proxy_server_port is None:
        args.proxy_server_port = find_free_port()
    else:
        args.proxy_server_port = find_free_port(args.proxy_server_port)

    for arg in ["vus", "rps"]:
        arg_val = getattr(args, arg)

        if arg_val is None:
            continue

        if ":" in arg_val:
            raw_steps = [int(param) for param in arg_val.split(":")]
            if len(raw_steps) == 2:
                start, end = raw_steps
                step = 1
            elif len(raw_steps) == 3:
                start, step, end = raw_steps
            else:
                raise RuntimeError(
                    "Measurements with fixed step should be configured as `<start>:<step>:<end>`"
                )
        else:
            end = int(arg_val)
            start = end
            step = 1

        default_extra_args = {
            "quantization": "fp16",
            "sharding": 1,
            "server_solution": "vllm",
            "hardware": "A100",
        }

        if args.extra_args != "":
            specified_extra_args = {
                arg.split("=")[0]: arg.split("=")[1]
                for arg in args.extra_args.split(",")
            }
        else:
            specified_extra_args = {}

        # Fill extra args with default values if field does not exist
        for key in default_extra_args:
            if key not in specified_extra_args:
                specified_extra_args[key] = default_extra_args[key]

        # Check that there are no invalid keys specified
        for key in specified_extra_args:
            if key not in default_extra_args:
                raise KeyError(
                    "--extra-args accepts only "
                    + ", ".join(sorted(default_extra_args))
                    + f". Found `{key}`"
                )
            else:
                # If ok, then put as a new arg
                setattr(args, key, specified_extra_args[key])

        all_args = []
        arg_vals = list(range(end, start - 1, -step))
        if start not in arg_vals:
            arg_vals.append(start)
        arg_vals = arg_vals[::-1]

        for arg_val in arg_vals:
            current_args = argparse.Namespace(**vars(args))

            setattr(current_args, arg, arg_val)
            setattr(current_args, "load", f"{arg}={arg_val}")

            all_args.append(current_args)

        return all_args


def save_workload_configuration(
    args: argparse.ArgumentParser, path_to_artifact: str
) -> None:
    with open(Path(path_to_artifact) / "config.json", "w") as config:
        json.dump(
            {
                "run_id": args.run_id,
                "server": {
                    "model": os.environ.get("MODEL_NAME", "unknown"),
                    "quantization": args.quantization,
                    "sharding": args.sharding,
                    "server_solution": args.server_solution,
                    "hardware": args.hardware,
                },
                "workload": {
                    "prompt": args.input_tokens_distribution,
                    "response": args.output_tokens_distribution,
                    "load": args.load,
                },
            },
            config,
            indent=4,
        )


def die(message: str) -> None:
    """Exit with an actionable message rather than a traceback."""
    print(f"metron: {message}", file=sys.stderr)
    raise SystemExit(1)


def preflight() -> None:
    """Check the external tools we shell out to, before doing any work."""
    if shutil.which("k6") is None:
        die(
            "k6 not found on PATH.\n"
            "  metron drives load with k6. Install it from "
            "https://github.com/grafana/k6/releases\n"
            "  and make sure the binary is on your PATH."
        )
    if shutil.which("node") is None:
        die(
            "node not found on PATH.\n"
            "  The streaming proxy runs on Node.js 20+. See "
            "https://nodejs.org/en/download/package-manager"
        )


def wait_for_port(port: int, timeout: float = 30.0) -> bool:
    """Poll until something accepts connections on ``port``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False


def detect_model(url: str) -> Optional[str]:
    """Ask the endpoint which model it serves, so users need not repeat it."""
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as resp:
            return json.load(resp)["data"][0]["id"]
    except Exception:
        return None


# A percentile needs roughly ten observations beyond it before it means
# anything. Below that it is one sample wearing a percentile's name: at 15
# seconds and 15 requests a second, p99 rests on two requests and p99.9 is
# simply the maximum. Printing them anyway invites the reader to believe a
# tail measurement that was never taken.
_MIN_SAMPLES = {"p(90)": 100, "p(99)": 1000, "p(99.9)": 10000}


def supported_percentiles(n: int) -> dict:
    """Which percentiles this many samples can carry, and which cannot."""
    return {k: n >= need for k, need in _MIN_SAMPLES.items()}


def print_summary(path_to_summary: str) -> None:
    """Print the percentiles the sample size supports, and say what it was."""
    try:
        with open(path_to_summary) as fh:
            summary = json.load(fh)
        metrics = summary["metrics"]
    except (OSError, KeyError, json.JSONDecodeError):
        return

    # k6's own http_reqs is stripped from the summary along with the rest of the
    # http* metrics; total_requests_sent is metron's counter, incremented once
    # per request, and is the sample size behind every percentile below.
    reqs = metrics.get("total_requests_sent") or metrics.get("http_reqs") or {}
    n = int(reqs.get("count", 0)) if isinstance(reqs, dict) else 0

    rows = [
        ("TTFT", "ttft"),
        ("ITL", "itl"),
        ("ITL/token", "itl_per_token"),
        ("TPOT", "tpot"),
        ("E2E", "end-to-end_latency"),
    ]
    cols = ["med", "p(90)", "p(99)", "max"]
    have = [(label, key) for label, key in rows if key in metrics]
    if not have:
        return

    ok = supported_percentiles(n)
    header = "ms" if not n else f"ms  (n={n})"
    print(f"{header:<14}" + "".join(f"{c:>12}" for c in ["p50", "p90", "p99", "max"]))
    for label, key in have:
        cells = ""
        for c in cols:
            if c in ok and not ok[c]:
                cells += f"{'—':>12}"
                continue
            v = metrics[key].get(c)
            cells += f"{(v if v is not None else float('nan')):>12.2f}"
        print(f"{label:<14}{cells}")

    thin = [c for c, good in ok.items() if not good and c in cols]
    if thin and n:
        need = ", ".join(f"{c} needs {_MIN_SAMPLES[c]}" for c in thin)
        print(
            f"\n{', '.join(thin)} not shown: {n} requests is too few ({need}). "
            f"Raise --duration or the rate."
        )

    for warning in run_health(summary):
        print(f"warning: {warning}")


def find_free_port(start_port: int = 3000) -> int:
    """Find first available port for proxy server"""
    current_port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", current_port))
                return current_port
            except OSError:
                current_port += 1


def main():
    parser = argparse.ArgumentParser(
        prog="metron",
        description="Arrival-rate load tester for OpenAI-compatible inference endpoints.",
        epilog=f"Full documentation: {REPO}",
    )
    parser.add_argument("--version", action="version", version=f"metron {__version__}")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Where to write run artifacts. Default: ./artifacts",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Endpoint URL to run benchmark against",
    )
    parser.add_argument(
        "--token", type=str, default="dummy_token", help="Access token for endpoint"
    )
    parser.add_argument(
        "--vus",
        type=str,
        help="Number of virtual users. To do a sequence of runs - specify as `<start>:<step>:<end>`. "
        "The values will be taken from the upper bound with fixed step until the lower bound is reached. "
        "If the lower bound itself wasn't taken (upper bound is not divisible by step) then it will be force appended.",
    )
    parser.add_argument(
        "--rps",
        type=str,
        help="Request per second. To do a sequence of runs - specify as `<start>:<step>:<end>`. "
        "The values will be taken from the upper bound with fixed step until the lower bound is reached. "
        "If the lower bound itself wasn't taken (upper bound is not divisible by step) then it will be force appended.",
    )
    parser.add_argument(
        "--max-vus",
        type=int,
        default=500,
        help="Ceiling on concurrent requests for rate-based runs. If the rate "
        "needs more than this, k6 drops iterations rather than exceed it; the "
        "summary says so. Default: 500",
    )
    parser.add_argument(
        "--duration",
        type=str,
        help="Test duration as a k6 duration string, e.g. 30s or 5m. Default: 30s",
    )
    parser.add_argument(
        "--iterations", type=int, help="How many requests to send to server under test"
    )
    parser.add_argument(
        "--input-tokens-distribution",
        type=str,
        default="normal(1000, 128)",
        help="""
        Generated prompt tokens distribution (choose 'normal', 'uniform' or 'const' and specify mean and std of number of tokens).
        To use with a given set of prompts specify `dataset(<path to .jsonl file with prompts>, <length of model's context>)`
        (the second arg is optional).
        """,
    )
    parser.add_argument(
        "--output-tokens-distribution",
        type=str,
        default="normal(256, 8)",
        help="""
        Output tokens distribution (choose 'normal', 'uniform' or 'const' and specify mean and std of number of tokens).
        To use sampling parameters from a given set of requests specify `dataset(<path to .jsonl file with requests>)`.
        """,
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="HF tokenizer id or path. Without it, token counts are approximated "
        "at 1 token per 4 characters.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="openai/gsm8k",
        help="Text the prompts are built from: a HuggingFace dataset id, or a "
        "path to a local .jsonl. Default openai/gsm8k (MIT). Real text matters: "
        "a small synthetic vocabulary repeats, which an engine with prefix "
        "caching answers without working, and it collapses expert routing on "
        "mixture-of-experts models.",
    )
    parser.add_argument(
        "--corpus-field",
        type=str,
        default="question",
        help="Field to read from each corpus record. Default 'question', which is GSM8K's schema — change it with --corpus.",
    )
    parser.add_argument(
        "--corpus-config",
        type=str,
        default="main",
        help="HuggingFace dataset config name. Default 'main' is GSM8K's; ignored for a local .jsonl corpus.",
    )
    parser.add_argument(
        "--corpus-split",
        type=str,
        default="train",
        help="HuggingFace dataset split.",
    )
    parser.add_argument(
        "--proxy-server-port",
        type=int,
        default=None,
        help="Port for the streaming proxy. Default: first free port from 3000",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Free-form run metadata recorded in config.json, as a comma-separated "
        "list. Accepts quantization, sharding, server_solution and hardware. "
        "For example: --extra-args server_solution=sglang,hardware=H200",
    )
    parser.add_argument(
        "--executor",
        type=str,
        choices=[
            "constant-vus",
            "constant-arrival-rate",
            "shared-iterations",
            "per-vu-iterations",
            "ramping-vus",
            "ramping-arrival-rate",
        ],
        help="Choice of k6 executor. For more information please see"
        "https://grafana.com/docs/k6/latest/using-k6/scenarios/executors/#all-executors",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="String identificator for measurement. If not set then random hash will be used",
    )
    parser.add_argument(
        "--additional-api-args",
        type=str,
        default="",
        help="If you need to pass additional arguments to the API, specify them here as a comma-separated list. "
        "For example, --additional-api-args TEMPERATURE=0.1.",
    )
    parser.add_argument(
        "--route",
        type=str,
        default="/v1/chat/completions",
        help="API path appended to --url. Default: /v1/chat/completions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for prompt/length selection. The same seed reproduces the "
        "same request stream; change it to vary the sample. Default: 42",
    )

    args = parser.parse_args()

    if bool(args.vus) == bool(args.rps):
        parser.error(
            "specify exactly one of --vus (N users, each waiting for its reply "
            "before sending again) or --rps (a fixed arrival rate, sending on "
            "schedule regardless). A rate exposes tail latency; waiting users "
            "throttle themselves exactly when the server slows."
        )

    if args.iterations and args.duration:
        parser.error("--duration and --iterations are mutually exclusive")

    if args.vus and args.executor is None:
        args.executor = "shared-iterations" if args.iterations else "constant-vus"

    if args.rps and args.executor is None:
        if args.iterations:
            parser.error("--iterations applies only to --vus runs; use --duration")
        args.executor = "constant-arrival-rate"

    vus_executors = [
        "constant-vus",
        "shared-iterations",
        "per-vu-iterations",
        "ramping-vus",
    ]
    rps_executors = ["constant-arrival-rate", "ramping-arrival-rate"]
    if args.vus and args.executor not in vus_executors:
        parser.error(
            f"--executor {args.executor} is rate-based; use --rps, or pick one of: "
            + ", ".join(vus_executors)
        )
    if args.rps and args.executor not in rps_executors:
        parser.error(
            f"--executor {args.executor} is user-based; use --vus, or pick one of: "
            + ", ".join(rps_executors)
        )

    if not args.duration and not args.iterations:
        args.duration = "30s"

    preflight()

    env_path = os.path.join(Path(__file__).parent, "configs", ".env.openai")
    load_dotenv(env_path, override=True)
    if args.additional_api_args:
        additional_args = {
            arg.split("=")[0]: arg.split("=")[1]
            for arg in args.additional_api_args.split(",")
        }
        for key, value in additional_args.items():
            os.environ[key] = value

    if not os.environ.get("MODEL_NAME"):
        detected = detect_model(args.url)
        if detected:
            os.environ["MODEL_NAME"] = detected
            print(f"Model auto-detected from {args.url}: {detected}")

    os.environ["URL"] = args.url + args.route
    os.environ["TOKEN"] = args.token
    os.environ["EXECUTOR"] = args.executor
    os.environ["SEED"] = str(args.seed)
    os.environ["MAX_VUS"] = str(args.max_vus)
    os.environ["API_ROUTE"] = args.route
    os.environ["TOKENIZER"] = args.tokenizer
    os.environ["METRON_CORPUS"] = args.corpus
    os.environ["METRON_CORPUS_FIELD"] = args.corpus_field
    os.environ["METRON_CORPUS_CONFIG"] = args.corpus_config
    os.environ["METRON_CORPUS_SPLIT"] = args.corpus_split

    if args.duration:
        os.environ["DURATION"] = args.duration

    if args.iterations:
        os.environ["ITERATIONS"] = str(args.iterations)

    if (
        os.environ["API_ROUTE"] == "/v1/completions"
        and os.environ.get("LOGPROBS", "false") == "false"
    ):
        os.environ["LOGPROBS"] = "0"

    list_of_args = postprocess_args(args)

    path_to_artifact = str(create_artifact(args.output_dir))
    os.environ["ARTIFACT"] = path_to_artifact
    prepare_workload(args, path_to_artifact)

    for run_id, args in enumerate(list_of_args):
        os.environ["PROXY_PORT"] = str(args.proxy_server_port)

        os.environ["VUS"] = str(args.vus)
        os.environ["RPS"] = str(args.rps)

        current_load = f"vus_{args.vus}" if args.vus else f"rps_{args.rps}"
        current_path = f"{path_to_artifact}/{current_load}"
        os.makedirs(current_path, exist_ok=True)
        path_to_trace = f"{current_path}/trace.json"
        path_to_summary = f"{current_path}/summary.json"
        save_workload_configuration(args, current_path)

        proxy_log_path = f"{current_path}/proxy.log"
        print(f"Starting proxy on port {args.proxy_server_port}...")
        with open(proxy_log_path, "w") as proxy_log:
            # Never a PIPE nobody reads: a chatty proxy would fill it and hang.
            proxy = subprocess.Popen(
                ["node", Path(__file__).parent / "proxy_server.js"],
                stdout=proxy_log,
                stderr=subprocess.STDOUT,
            )
            try:
                if not wait_for_port(args.proxy_server_port):
                    proxy.terminate()
                    die(
                        f"proxy did not come up on port "
                        f"{args.proxy_server_port}; see {proxy_log_path}"
                    )

                print("Starting k6...")
                result = subprocess.run(
                    [
                        "k6",
                        "run",
                        Path(__file__).parent / "metron.js",
                        "--out",
                        f"json={path_to_trace}",
                        "--summary-export",
                        f"{path_to_summary}",
                        "--summary-trend-stats",
                        "avg,min,med,max,p(50),p(75),p(90),p(95),p(99)",
                    ]
                )
                if result.returncode != 0:
                    die(f"k6 exited with status {result.returncode}")
            finally:
                # Runs on Ctrl-C and on any failure above, so no orphan node.
                proxy.terminate()
                try:
                    proxy.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proxy.kill()

        postprocess_results(path_to_trace, path_to_summary)
        print_summary(path_to_summary)

        print("-" * 72)
        print(f"Artifacts of this run are stored in {current_path}")


if __name__ == "__main__":
    main()
