#!/usr/bin/env python3
"""Use vLLM latency metrics to control one or more NVIDIA GPUs with MIAD.

MIAD here means multiplicative increase/additive decrease:

* If TTFT, TBT, or end-to-end latency exceeds its threshold, multiply the
  target GPU clock (and clamp it to the configured maximum).
* Otherwise, subtract a fixed number of MHz (and clamp it to the minimum).

All GPUs configured for one controller share a single metrics sample, MIAD
decision, and target clock.  This keeps tensor-parallel ranks on the same
frequency policy instead of letting independent controller processes drift.

The vLLM metrics are cumulative Prometheus counters.  Latencies are therefore
calculated from the sum/count deltas between two consecutive scrapes, rather
than from the process-lifetime average exposed by one scrape.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, TextIO


# vLLM has used both colon-separated and underscore-separated metric names.
# TBT is called inter-token latency in newer versions and TPOT in some older
# versions, so both names are accepted.
TTFT_METRICS = (
    "vllm:time_to_first_token_seconds",
    "vllm_time_to_first_token_seconds",
)
TBT_METRICS = (
    "vllm:inter_token_latency_seconds",
    "vllm_inter_token_latency_seconds",
    "vllm:time_per_output_token_seconds",
    "vllm_time_per_output_token_seconds",
)
E2E_METRICS = (
    "vllm:e2e_request_latency_seconds",
    "vllm_e2e_request_latency_seconds",
)

RUNNING_METRICS = ("vllm:num_requests_running", "vllm_num_requests_running")
WAITING_METRICS = ("vllm:num_requests_waiting", "vllm_num_requests_waiting")
KV_CACHE_METRICS = (
    "vllm:gpu_cache_usage_perc",
    "vllm_gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm_kv_cache_usage_perc",
)


@dataclass(frozen=True)
class Latencies:
    """Average latency values observed during one sampling interval."""

    ttft_ms: Optional[float]
    tbt_ms: Optional[float]
    e2e_ms: Optional[float]


@dataclass(frozen=True)
class GpuStats:
    clock_mhz: Optional[int]
    clocks_mhz_by_gpu: Optional[dict[int, int]]
    power_w: Optional[float]
    utilization_pct: Optional[float]
    temperature_c: Optional[float]
    memory_used_mib: Optional[float]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_prometheus(text: str) -> dict[str, float]:
    """Aggregate Prometheus samples by metric name across all label sets."""

    values: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # The metric name is before either the label block or whitespace.  The
        # first field after the sample is its value; an optional timestamp is
        # deliberately ignored.
        name_end = len(line)
        for separator in ("{", " ", "\t"):
            position = line.find(separator)
            if position >= 0:
                name_end = min(name_end, position)
        name = line[:name_end]

        if "}" in line:
            value_part = line.rsplit("}", 1)[1].strip()
        else:
            value_part = line[name_end:].strip()
        if not name or not value_part:
            continue

        try:
            value = float(value_part.split()[0])
        except (ValueError, IndexError):
            continue
        if math.isfinite(value):
            values[name] = values.get(name, 0.0) + value
    return values


def first_metric(metrics: Mapping[str, float], names: Iterable[str]) -> Optional[float]:
    """Return the first supported metric, without combining aliases."""

    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def interval_average_ms(
    previous: Mapping[str, float],
    current: Mapping[str, float],
    bases: Iterable[str],
) -> Optional[float]:
    """Calculate an interval average from one histogram/summary sum and count."""

    for base in bases:
        sum_name = f"{base}_sum"
        count_name = f"{base}_count"
        if all(name in previous and name in current for name in (sum_name, count_name)):
            count_delta = current[count_name] - previous[count_name]
            sum_delta = current[sum_name] - previous[sum_name]
            # A negative delta means vLLM restarted or a labelled series
            # disappeared.  Zero means no completed observation in the window.
            if count_delta <= 0 or sum_delta < 0:
                return None
            return 1000.0 * sum_delta / count_delta
    return None


def metric_family_available(metrics: Mapping[str, float], bases: Iterable[str]) -> bool:
    """Whether at least one compatible sum/count pair exists in a scrape."""

    return any(f"{base}_sum" in metrics and f"{base}_count" in metrics for base in bases)


def calculate_latencies(
    previous: Mapping[str, float], current: Mapping[str, float]
) -> Latencies:
    return Latencies(
        ttft_ms=interval_average_ms(previous, current, TTFT_METRICS),
        tbt_ms=interval_average_ms(previous, current, TBT_METRICS),
        e2e_ms=interval_average_ms(previous, current, E2E_METRICS),
    )


class VllmMetricsClient:
    def __init__(self, url: str, timeout_seconds: float) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds

    def scrape(self) -> dict[str, float]:
        request = urllib.request.Request(
            self.url,
            headers={"Accept": "text/plain", "User-Agent": "gpu-miad-controller/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"cannot scrape {self.url}: {exc}") from exc
        return parse_prometheus(body)


class NvidiaGpuGroup:
    """Control one frequency target across one or more NVIDIA GPUs."""

    def __init__(self, gpu_ids: Iterable[int], dry_run: bool) -> None:
        self.gpu_ids = tuple(gpu_ids)
        if not self.gpu_ids:
            raise ValueError("at least one GPU ID is required")
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("GPU IDs must be unique")
        if any(gpu_id < 0 for gpu_id in self.gpu_ids):
            raise ValueError("GPU IDs cannot be negative")
        self.dry_run = dry_run

    @property
    def selector(self) -> str:
        return ",".join(str(gpu_id) for gpu_id in self.gpu_ids)

    def _run(self, gpu_id: int, arguments: list[str]) -> str:
        command = ["nvidia-smi", "-i", str(gpu_id), *arguments]
        if self.dry_run:
            print(f"[dry-run] {' '.join(command)}")
            return ""
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("nvidia-smi was not found in PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("nvidia-smi timed out") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"nvidia-smi failed: {detail}") from exc
        return completed.stdout.strip()

    def validate(self) -> None:
        if self.dry_run:
            return
        for gpu_id in self.gpu_ids:
            output = self._run(
                gpu_id,
                ["--query-gpu=index", "--format=csv,noheader,nounits"],
            )
            try:
                detected = tuple(
                    int(line.strip()) for line in output.splitlines() if line.strip()
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"unexpected nvidia-smi GPU index output for GPU {gpu_id}: {output!r}"
                ) from exc
            if detected != (gpu_id,):
                raise RuntimeError(
                    f"nvidia-smi selected GPUs {list(detected)}, expected GPU {gpu_id}"
                )

    def _apply_to_all(self, arguments: list[str], action: str) -> None:
        errors: list[str] = []
        for gpu_id in self.gpu_ids:
            try:
                self._run(gpu_id, arguments)
            except RuntimeError as exc:
                errors.append(f"GPU {gpu_id}: {exc}")
        if errors:
            raise RuntimeError(
                f"failed to {action} on all configured GPUs: {'; '.join(errors)}"
            )

    def lock_clock(self, frequency_mhz: int) -> None:
        # A fixed min/max graphics clock gives the controller one unambiguous
        # frequency target.  This normally requires root/admin permission.
        self._apply_to_all(
            ["--lock-gpu-clocks", f"{frequency_mhz},{frequency_mhz}"],
            f"lock clock to {frequency_mhz} MHz",
        )

    def reset_clock(self) -> None:
        self._apply_to_all(["--reset-gpu-clocks"], "reset clocks")

    def stats(self) -> GpuStats:
        if self.dry_run:
            return GpuStats(None, None, None, None, None, None)
        try:
            per_gpu: dict[int, tuple[int, float, float, float, float]] = {}
            outputs: dict[int, str] = {}
            for requested_gpu_id in self.gpu_ids:
                output = self._run(
                    requested_gpu_id,
                    [
                        "--query-gpu=index,clocks.gr,power.draw,utilization.gpu,temperature.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                )
                outputs[requested_gpu_id] = output
                fields = [field.strip() for field in output.split(",")]
                gpu_id = int(fields[0])
                if gpu_id != requested_gpu_id:
                    raise ValueError(
                        f"query for GPU {requested_gpu_id} returned GPU {gpu_id}"
                    )
                per_gpu[gpu_id] = (
                    int(float(fields[1])),
                    float(fields[2]),
                    float(fields[3]),
                    float(fields[4]),
                    float(fields[5]),
                )
            if set(per_gpu) != set(self.gpu_ids):
                raise ValueError("GPU result set does not match the configured IDs")
            clocks = {gpu_id: values[0] for gpu_id, values in per_gpu.items()}
            gpu_count = len(per_gpu)
            return GpuStats(
                clock_mhz=round(sum(clocks.values()) / gpu_count),
                clocks_mhz_by_gpu=clocks,
                power_w=sum(values[1] for values in per_gpu.values()),
                utilization_pct=sum(values[2] for values in per_gpu.values()) / gpu_count,
                temperature_c=max(values[3] for values in per_gpu.values()),
                memory_used_mib=sum(values[4] for values in per_gpu.values()),
            )
        except (IndexError, ValueError, ZeroDivisionError) as exc:
            raise RuntimeError(f"unexpected nvidia-smi output: {outputs!r}") from exc


class CsvLogger:
    # Keep the key type as ``str``.  Without this annotation, Pyright may infer
    # a tuple of Literal keys and consequently reject dict[str, object] in
    # DictWriter.writerow().
    COLUMNS: tuple[str, ...] = (
        "timestamp",
        "gpu_ids",
        "action",
        "reason",
        "target_clock_mhz",
        "actual_clock_mhz",
        "actual_clocks_mhz",
        "power_w",
        "gpu_utilization_pct",
        "temperature_c",
        "memory_used_mib",
        "ttft_ms",
        "tbt_ms",
        "e2e_slo_ms",
        "requests_running",
        "requests_waiting",
        "kv_cache_usage_pct",
    )

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self._file: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter[str]] = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
            if self._file.tell() == 0:
                self._writer.writeheader()
                self._file.flush()

    def write(self, row: dict[str, object]) -> None:
        if self._writer is not None and self._file is not None:
            self._writer.writerow(row)
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def show(value: Optional[float], digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def check_thresholds(
    latencies: Latencies,
    ttft_threshold_ms: float,
    tbt_threshold_ms: float,
    slo_threshold_ms: float,
) -> list[str]:
    violations = []
    for name, value, threshold in (
        ("TTFT", latencies.ttft_ms, ttft_threshold_ms),
        ("TBT", latencies.tbt_ms, tbt_threshold_ms),
        ("SLO(E2E)", latencies.e2e_ms, slo_threshold_ms),
    ):
        if value is not None and value > threshold:
            violations.append(f"{name} {value:.1f}>{threshold:.1f}ms")
    return violations


def run_controller(args: argparse.Namespace) -> int:
    url_host = f"[{args.vllm_ip}]" if ":" in args.vllm_ip else args.vllm_ip
    metrics_url = args.metrics_url or f"http://{url_host}:{args.vllm_port}/metrics"
    client = VllmMetricsClient(metrics_url, args.request_timeout)
    gpu = NvidiaGpuGroup(args.gpu_ids, args.dry_run)
    logger = CsvLogger(None if args.no_log else Path(args.log_file))
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    target_clock = args.initial_frequency
    previous_metrics: Optional[dict[str, float]] = None
    started_at = time.monotonic()

    print(f"vLLM metrics: {metrics_url}")
    print(
        f"GPUs {gpu.selector}: {args.min_frequency}-{args.max_frequency} MHz, "
        f"initial={target_clock} MHz, AD={args.decrease_step} MHz, "
        f"MI=x{args.increase_factor:g}"
    )
    print(
        f"thresholds: TTFT={args.ttft_threshold_ms:g} ms, "
        f"TBT={args.tbt_threshold_ms:g} ms, SLO(E2E)={args.slo_threshold_ms:g} ms"
    )

    try:
        gpu.validate()
        gpu.lock_clock(target_clock)

        while not stop_requested:
            cycle_started_at = time.monotonic()
            if args.duration > 0 and cycle_started_at - started_at >= args.duration:
                break

            try:
                current_metrics = client.scrape()
            except RuntimeError as exc:
                # Fail safe: stale or missing service data must never make the
                # controller lower the GPU clock.
                print(f"[{utc_now()}] HOLD: {exc}", file=sys.stderr)
                elapsed = time.monotonic() - cycle_started_at
                time.sleep(max(0.0, args.interval - elapsed))
                continue

            if previous_metrics is None:
                previous_metrics = current_metrics
                print(f"[{utc_now()}] WARMUP: first metrics sample collected; clock={target_clock} MHz")
                elapsed = time.monotonic() - cycle_started_at
                time.sleep(max(0.0, args.interval - elapsed))
                continue

            latencies = calculate_latencies(previous_metrics, current_metrics)
            previous_metrics = current_metrics
            latency_values = (latencies.ttft_ms, latencies.tbt_ms, latencies.e2e_ms)
            missing_families = [
                name
                for name, bases in (
                    ("TTFT", TTFT_METRICS),
                    ("TBT", TBT_METRICS),
                    ("E2E", E2E_METRICS),
                )
                if not metric_family_available(current_metrics, bases)
            ]
            violations = check_thresholds(
                latencies,
                args.ttft_threshold_ms,
                args.tbt_threshold_ms,
                args.slo_threshold_ms,
            )

            if missing_families:
                # Do not silently downclock if an incompatible vLLM release or
                # metrics configuration prevents evaluation of a threshold.
                action = "HOLD"
                reason = "missing vLLM metric families: " + ", ".join(missing_families)
                new_clock = target_clock
            elif all(value is None for value in latency_values):
                if args.decrease_when_idle:
                    action = "AD"
                    reason = "no completed latency observations"
                    new_clock = max(args.min_frequency, target_clock - args.decrease_step)
                else:
                    action = "HOLD"
                    reason = "no completed latency observations"
                    new_clock = target_clock
            elif violations:
                action = "MI"
                reason = "; ".join(violations)
                new_clock = min(
                    args.max_frequency,
                    max(target_clock + 1, math.ceil(target_clock * args.increase_factor)),
                )
            else:
                action = "AD"
                reason = "all observed latencies within thresholds"
                new_clock = max(args.min_frequency, target_clock - args.decrease_step)

            if new_clock != target_clock:
                gpu.lock_clock(new_clock)
                target_clock = new_clock

            try:
                gpu_stats = gpu.stats()
            except RuntimeError as exc:
                print(f"[{utc_now()}] GPU stats warning: {exc}", file=sys.stderr)
                gpu_stats = GpuStats(None, None, None, None, None, None)

            running = first_metric(current_metrics, RUNNING_METRICS)
            waiting = first_metric(current_metrics, WAITING_METRICS)
            kv_cache = first_metric(current_metrics, KV_CACHE_METRICS)
            # vLLM exposes cache usage as a 0..1 ratio despite the "perc" name.
            kv_cache_pct = None if kv_cache is None else 100.0 * kv_cache

            timestamp = utc_now()
            print(
                f"[{timestamp}] {action:<4} clock={target_clock:4d}MHz "
                f"TTFT={show(latencies.ttft_ms)}ms "
                f"TBT={show(latencies.tbt_ms)}ms "
                f"E2E={show(latencies.e2e_ms)}ms "
                f"run={show(running, 0)} wait={show(waiting, 0)} "
                f"util={show(gpu_stats.utilization_pct, 0)}% power={show(gpu_stats.power_w)}W "
                f"| {reason}"
            )
            logger.write(
                {
                    "timestamp": timestamp,
                    "gpu_ids": gpu.selector,
                    "action": action,
                    "reason": reason,
                    "target_clock_mhz": target_clock,
                    "actual_clock_mhz": gpu_stats.clock_mhz,
                    "actual_clocks_mhz": (
                        None
                        if gpu_stats.clocks_mhz_by_gpu is None
                        else json.dumps(gpu_stats.clocks_mhz_by_gpu, sort_keys=True)
                    ),
                    "power_w": gpu_stats.power_w,
                    "gpu_utilization_pct": gpu_stats.utilization_pct,
                    "temperature_c": gpu_stats.temperature_c,
                    "memory_used_mib": gpu_stats.memory_used_mib,
                    "ttft_ms": latencies.ttft_ms,
                    "tbt_ms": latencies.tbt_ms,
                    "e2e_slo_ms": latencies.e2e_ms,
                    "requests_running": running,
                    "requests_waiting": waiting,
                    "kv_cache_usage_pct": kv_cache_pct,
                }
            )

            elapsed = time.monotonic() - cycle_started_at
            time.sleep(max(0.0, args.interval - elapsed))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        logger.close()
        if args.reset_on_exit:
            try:
                gpu.reset_clock()
                print(f"GPU clocks reset for {gpu.selector}")
            except RuntimeError as exc:
                print(f"warning: failed to reset GPU clock: {exc}", file=sys.stderr)
    return 0


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite, non-negative number")
    return number


def gpu_id_list(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, unique list of non-negative GPU IDs."""
    try:
        parts = [part.strip() for part in value.split(",")]
        if not parts or any(not part for part in parts):
            raise ValueError
        gpu_ids = tuple(int(part) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of non-negative GPU IDs"
        ) from None
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError("GPU IDs cannot be negative")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("GPU IDs must be unique")
    return gpu_ids


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Control one or more NVIDIA GPU clocks from vLLM TTFT/TBT/SLO "
            "metrics using one shared MIAD policy."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu-ids",
        type=gpu_id_list,
        help="comma-separated NVIDIA GPU indexes controlled by one MIAD policy",
    )
    gpu_group.add_argument(
        "--gpu-id",
        type=int,
        help="single NVIDIA GPU index (backward-compatible alias)",
    )
    parser.add_argument("--vllm-ip", default="127.0.0.1", help="vLLM IP or hostname")
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLM HTTP port")
    parser.add_argument(
        "--metrics-url",
        help="full Prometheus URL; overrides --vllm-ip and --vllm-port",
    )
    parser.add_argument("--interval", type=positive_float, default=5.0, help="scan interval in seconds")
    parser.add_argument(
        "--duration",
        type=nonnegative_float,
        default=0.0,
        help="run time in seconds; 0 runs until SIGINT/SIGTERM",
    )
    parser.add_argument(
        "--request-timeout",
        type=positive_float,
        default=3.0,
        help="metrics HTTP timeout in seconds",
    )

    parser.add_argument(
        "--ttft-threshold-ms",
        type=positive_float,
        default=250.0,
        help="average time-to-first-token threshold in milliseconds",
    )
    parser.add_argument(
        "--tbt-threshold-ms",
        type=positive_float,
        default=100.0,
        help="average inter-token latency threshold in milliseconds",
    )
    parser.add_argument(
        "--slo-threshold-ms",
        type=positive_float,
        default=2000.0,
        help="average end-to-end request latency SLO in milliseconds",
    )

    parser.add_argument("--min-frequency", type=positive_int, default=300, help="minimum MHz")
    parser.add_argument("--max-frequency", type=positive_int, default=1410, help="maximum MHz")
    parser.add_argument("--initial-frequency", type=positive_int, default=1410, help="initial MHz")
    parser.add_argument(
        "--decrease-step",
        type=positive_int,
        default=100,
        help="additive decrease in MHz",
    )
    parser.add_argument(
        "--increase-factor",
        type=positive_float,
        default=2.0,
        help="multiplicative increase factor",
    )
    parser.add_argument(
        "--decrease-when-idle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="downclock intervals with no completed requests",
    )

    parser.add_argument("--log-file", default="gpu_miad_control.csv", help="CSV log path")
    parser.add_argument("--no-log", action="store_true", help="disable CSV logging")
    parser.add_argument(
        "--reset-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reset the GPU clock lock on exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print nvidia-smi control commands without executing them",
    )

    args = parser.parse_args(argv)
    if args.gpu_id is not None and args.gpu_id < 0:
        parser.error("--gpu-id cannot be negative")
    if args.gpu_ids is None:
        args.gpu_ids = (0,) if args.gpu_id is None else (args.gpu_id,)
    if not 1 <= args.vllm_port <= 65535:
        parser.error("--vllm-port must be between 1 and 65535")
    if args.min_frequency > args.max_frequency:
        parser.error("--min-frequency cannot exceed --max-frequency")
    if not args.min_frequency <= args.initial_frequency <= args.max_frequency:
        parser.error("--initial-frequency must be within the min/max range")
    if args.increase_factor <= 1.0:
        parser.error("--increase-factor must be greater than 1")
    return args


if __name__ == "__main__":
    raise SystemExit(run_controller(parse_args()))
