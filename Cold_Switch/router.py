#!/usr/bin/env python3
"""Run cold TP switching with online length prediction; no LLF scheduling."""

from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import json
import logging
import math
import queue
import random
import re
import signal
import threading
import time
import urllib.error
import urllib.request
import uuid

from predictors import load_predictor, request_type_for, validate_prediction
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

# Cold experiment defaults. Deployment, traffic and prediction are configurable.
EXPERIMENT_DURATION_SECONDS = 3600
QPS_SEGMENT_SECONDS = 300
DATASET_DIRECTORY = Path(__file__).resolve().parent / "datasets"
MODEL_NAME = None
OPENAI_API_KEY = "vllm"
REQUEST_TIMEOUT_SECONDS = 600
TEMPERATURE = 0
USE_DATASET_OUTPUT_TOKENS = True
DEFAULT_MAX_TOKENS = 512
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096
DEFAULT_REQUEST_SENDER_THREADS = 256
DEFAULT_MAX_IN_FLIGHT_REQUESTS = 2048
DEFAULT_MAX_PENDING_REQUESTS_PER_TYPE = 20000
MONITOR_CONTROL_TIMEOUT_SECONDS = 15
COMPLETION_LOG_FILE = Path("results/requests.jsonl")
# Historical reference SLO: (TTFT + reference output tokens * TBT) * 1.4.
# Predictor output never changes generation limits or the reference SLO.
SLO_MULTIPLIER = 1.4
TTFT_SECONDS_BY_REQUEST_TYPE = {
    "short_prompt_short_response": 0.22839,
    "short_prompt_long_response": 0.06829,
    "long_prompt_short_response": 0.19558,
    "long_prompt_long_response": 0.12039,
}
TBT_SECONDS_BY_REQUEST_TYPE = {
    "short_prompt_short_response": 0.09162,
    "short_prompt_long_response": 0.04899,
    "long_prompt_short_response": 0.06786,
    "long_prompt_long_response": 0.03523,
}
DATASET_FILENAMES = {
    "short_prompt_short_response": "prompts_short_answers_short.jsonl",
    "short_prompt_long_response": "prompts_short_answers_long.jsonl",
    "long_prompt_short_response": "prompts_long_answers_short.jsonl",
    "long_prompt_long_response": "prompts_long_answers_long.jsonl",
}
REQUEST_CONFIGS = {
    kind: {"qps": [1.0] * 12, "dataset": DATASET_DIRECTORY / filename}
    for kind, filename in DATASET_FILENAMES.items()
}


def build_request_configs(
    dataset_directory: Path,
    qps_config_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    profiles = {
        request_type: list(config["qps"])
        for request_type, config in REQUEST_CONFIGS.items()
    }
    if qps_config_path is not None:
        try:
            payload = json.loads(qps_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot load QPS config {qps_config_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("QPS config must be a JSON object")
        missing = [
            request_type
            for request_type in DATASET_FILENAMES
            if request_type not in payload
        ]
        extra = [
            request_type
            for request_type in payload
            if request_type not in DATASET_FILENAMES
        ]
        if missing or extra:
            raise RuntimeError(
                f"QPS config keys mismatch; missing={missing or 'none'}, extra={extra or 'none'}"
            )
        profiles = payload

    return {
        request_type: {
            "qps": profiles[request_type],
            "dataset": dataset_directory / filename,
        }
        for request_type, filename in DATASET_FILENAMES.items()
    }


LOGGER = logging.getLogger("vllm-router")
MAX_REQUEST_BODY_BYTES = 1024 * 1024
HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}\.?$)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*"
    r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.?$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_host(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ip must be a non-empty string")

    host = value.strip()
    if "://" in host or "/" in host or "?" in host or "#" in host:
        raise ValueError(
            "ip must contain only an IP address or hostname, without a URL scheme or path"
        )

    # Allow either an IPv4/IPv6 address or a DNS hostname such as vllm-a.local.
    try:
        return ipaddress.ip_address(host.strip("[]")).compressed
    except ValueError:
        if not HOSTNAME_RE.fullmatch(host):
            raise ValueError("ip is not a valid IP address or hostname") from None
        return host.rstrip(".").lower()


def normalize_port(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("port must be an integer between 1 and 65535")
    if isinstance(value, str):
        value = value.strip()
        if not value.isdecimal():
            raise ValueError("port must be an integer between 1 and 65535")
        value = int(value)
    if not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError("port must be an integer between 1 and 65535")
    return value


def normalize_weight(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("weight must be a positive number")
    try:
        weight = float(value)
    except (TypeError, ValueError):
        raise ValueError("weight must be a positive number") from None
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("weight must be a positive number")
    return weight


def normalize_request_type(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("request_type must be a non-empty string")
    return value.strip()


def normalize_phase(value: Any) -> int:
    """Return phase 1 or 2 while keeping static type narrowing explicit."""
    if isinstance(value, bool):
        raise ValueError("phase must be 1 or 2")
    if isinstance(value, int):
        phase = value
    elif isinstance(value, str) and value.strip() in {"1", "2"}:
        phase = int(value.strip())
    else:
        raise ValueError("phase must be 1 or 2")
    if phase not in (1, 2):
        raise ValueError("phase must be 1 or 2")
    return phase


def make_base_url(host: str, port: int) -> str:
    # IPv6 addresses need brackets when embedded in a URL.
    url_host = f"[{host}]" if ":" in host else host
    return f"http://{url_host}:{port}"


def normalize_http_base_url(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty HTTP URL")
    cleaned = value.strip().rstrip("/")
    parsed = urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field} must be an absolute HTTP or HTTPS URL")
    try:
        parsed_port = parsed.port
    except ValueError:
        raise ValueError(f"{field} contains an invalid port") from None
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{field} cannot contain credentials, a query, or a fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError(f"{field} must not contain a path")
    if parsed_port is not None and not 1 <= parsed_port <= 65535:
        raise ValueError(f"{field} contains an invalid port")
    return cleaned


@dataclass(frozen=True)
class Worker:
    ip: str
    port: int
    request_type: str
    weight: float
    configuration_id: str
    tp_size: int | None
    base_url: str
    registered_at: str
    updated_at: str


class WorkerRegistry:
    """Thread-safe, in-memory vLLM worker registry."""

    def __init__(self, initial_configuration_id: str = "default") -> None:
        self._workers: dict[tuple[str, str, int], Worker] = {}
        self._weighted_current: dict[
            tuple[str, str], dict[tuple[str, str, int], float]
        ] = {}
        self._active_configuration_id = initial_configuration_id
        self._lock = threading.RLock()

    @property
    def active_configuration_id(self) -> str:
        with self._lock:
            return self._active_configuration_id

    def add(
        self,
        ip: str,
        port: int,
        request_type: str,
        weight: float,
        configuration_id: str = "default",
        tp_size: int | None = None,
    ) -> tuple[Worker, str]:
        key = (configuration_id, ip, port)
        now = utc_now()
        with self._lock:
            existing = self._workers.get(key)
            if existing is None:
                worker = Worker(
                    ip=ip,
                    port=port,
                    request_type=request_type,
                    weight=weight,
                    configuration_id=configuration_id,
                    tp_size=tp_size,
                    base_url=make_base_url(ip, port),
                    registered_at=now,
                    updated_at=now,
                )
                action = "registered"
            elif (
                existing.request_type == request_type
                and existing.weight == weight
                and existing.tp_size == tp_size
            ):
                worker = existing
                action = "unchanged"
            else:
                worker = Worker(
                    ip=existing.ip,
                    port=existing.port,
                    request_type=request_type,
                    weight=weight,
                    configuration_id=configuration_id,
                    tp_size=tp_size,
                    base_url=existing.base_url,
                    registered_at=existing.registered_at,
                    updated_at=now,
                )
                action = "updated"

            self._workers[key] = worker
            return worker, action

    def list(
        self,
        request_type: str | None = None,
        configuration_id: str | None = None,
    ) -> list[Worker]:
        with self._lock:
            workers = list(self._workers.values())
        if request_type is not None:
            workers = [
                worker for worker in workers if worker.request_type == request_type
            ]
        if configuration_id is not None:
            workers = [
                worker
                for worker in workers
                if worker.configuration_id == configuration_id
            ]
        return sorted(
            workers,
            key=lambda worker: (
                worker.configuration_id,
                worker.request_type,
                worker.ip,
                worker.port,
            ),
        )

    def remove(self, ip: str, port: int, configuration_id: str) -> Worker | None:
        with self._lock:
            return self._workers.pop((configuration_id, ip, port), None)

    def activate(self, configuration_id: str) -> tuple[str, str]:
        with self._lock:
            if not any(
                worker.configuration_id == configuration_id
                for worker in self._workers.values()
            ):
                raise ValueError(
                    f"configuration has no registered workers: {configuration_id}"
                )
            previous = self._active_configuration_id
            self._active_configuration_id = configuration_id
            self._weighted_current.clear()
            return previous, configuration_id

    def configurations(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for worker in self._workers.values():
                counts[worker.configuration_id] = (
                    counts.get(worker.configuration_id, 0) + 1
                )
            return counts

    def count(self) -> int:
        with self._lock:
            return len(self._workers)

    def choose(
        self,
        request_type: str,
        eligible_keys: set[tuple[str, str, int]] | None = None,
    ) -> Worker | None:
        """Choose a matching worker using smooth weighted round-robin."""
        with self._lock:
            configuration_id = self._active_configuration_id
            candidates = sorted(
                (
                    worker
                    for worker in self._workers.values()
                    if worker.request_type == request_type
                    and worker.configuration_id == configuration_id
                ),
                key=lambda worker: (worker.ip, worker.port),
            )
            if not candidates:
                return None

            current = self._weighted_current.setdefault(
                (configuration_id, request_type), {}
            )
            candidate_keys = {
                (worker.configuration_id, worker.ip, worker.port)
                for worker in candidates
            }
            for stale_key in set(current) - candidate_keys:
                del current[stale_key]

            if eligible_keys is not None:
                candidates = [
                    worker
                    for worker in candidates
                    if (worker.configuration_id, worker.ip, worker.port)
                    in eligible_keys
                ]
            if not candidates:
                return None

            total_weight = 0.0
            selected: Worker | None = None
            selected_current = float("-inf")
            for worker in candidates:
                key = (worker.configuration_id, worker.ip, worker.port)
                current[key] = current.get(key, 0.0) + worker.weight
                total_weight += worker.weight
                if current[key] > selected_current:
                    selected = worker
                    selected_current = current[key]

            assert selected is not None
            selected_key = (selected.configuration_id, selected.ip, selected.port)
            current[selected_key] -= total_weight
            return selected

    def count_for_type(self, request_type: str, active_only: bool = True) -> int:
        with self._lock:
            return sum(
                worker.request_type == request_type
                and (
                    not active_only
                    or worker.configuration_id == self._active_configuration_id
                )
                for worker in self._workers.values()
            )


worker_registry = WorkerRegistry()
phase_config_paths: dict[int, Path] = {}


def _normalize_command_list(value: Any, field: str) -> list[list[str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array of command arrays")
    commands: list[list[str]] = []
    for index, command in enumerate(value, start=1):
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise ValueError(f"{field}[{index}] must be a non-empty array of strings")
        commands.append(command)
    return commands


def load_phase_config(phase: int) -> dict[str, Any]:
    path = phase_config_paths.get(phase)
    if path is None:
        raise ValueError(f"phase {phase} config path is not configured")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load phase {phase} config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"phase {phase} config must be a JSON object")
    raw_configuration_id = payload.get("configuration_id")
    if not isinstance(raw_configuration_id, str) or not raw_configuration_id.strip():
        raise ValueError("configuration_id must be a non-empty string")
    configuration_id = raw_configuration_id.strip()
    raw_workers = payload.get("workers")
    if not isinstance(raw_workers, list) or not raw_workers:
        raise ValueError("workers must be a non-empty array")

    workers: list[dict[str, Any]] = []
    worker_ids: set[str] = set()
    endpoints: set[tuple[str, int]] = set()
    monitor_urls: set[str] = set()
    for index, raw_worker in enumerate(raw_workers, start=1):
        if not isinstance(raw_worker, dict):
            raise ValueError(f"workers[{index}] must be a JSON object")
        if "worker_id" not in raw_worker:
            raise ValueError(f"workers[{index}].worker_id is required")
        raw_worker_id = raw_worker.get("worker_id")
        if not isinstance(raw_worker_id, str) or not raw_worker_id.strip():
            raise ValueError(f"workers[{index}].worker_id must be a non-empty string")
        worker_id = raw_worker_id.strip()
        if worker_id in worker_ids:
            raise ValueError(f"workers[{index}].worker_id must be non-empty and unique")
        worker_ids.add(worker_id)
        ip = normalize_host(raw_worker.get("ip"))
        port = normalize_port(raw_worker.get("port"))
        if (ip, port) in endpoints:
            raise ValueError(f"duplicate phase endpoint: {ip}:{port}")
        endpoints.add((ip, port))
        request_type = normalize_request_type(raw_worker.get("request_type"))
        if request_type not in REQUEST_CONFIGS:
            raise ValueError(f"unsupported request_type: {request_type}")
        weight = normalize_weight(raw_worker.get("weight"))
        raw_tp_size = raw_worker.get("tp_size")
        if raw_tp_size is not None and (
            isinstance(raw_tp_size, bool)
            or not isinstance(raw_tp_size, int)
            or raw_tp_size <= 0
        ):
            raise ValueError("tp_size must be a positive integer or null")
        tp_size = raw_tp_size
        if "gpu_ids" not in raw_worker:
            raise ValueError(f"workers[{index}].gpu_ids is required")
        raw_gpu_ids = raw_worker.get("gpu_ids")
        if (
            not isinstance(raw_gpu_ids, list)
            or not raw_gpu_ids
            or any(
                isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0
                for gpu_id in raw_gpu_ids
            )
            or len(set(raw_gpu_ids)) != len(raw_gpu_ids)
        ):
            raise ValueError(
                f"workers[{index}].gpu_ids must be a non-empty array of unique "
                "non-negative integers"
            )
        if "health_url" not in raw_worker:
            raise ValueError(f"workers[{index}].health_url is required")
        raw_health_url = raw_worker.get("health_url")
        if not isinstance(raw_health_url, str) or not raw_health_url.strip():
            raise ValueError(f"workers[{index}].health_url cannot be empty")
        health_url = raw_health_url.strip()
        raw_start_command = raw_worker.get("start_command")
        start_command = None
        if raw_start_command is not None:
            if (
                not isinstance(raw_start_command, list)
                or not raw_start_command
                or not all(isinstance(item, str) and item for item in raw_start_command)
            ):
                raise ValueError(
                    f"workers[{index}].start_command must be an array of strings"
                )
            start_command = raw_start_command
        raw_monitor = raw_worker.get("monitor")
        if not isinstance(raw_monitor, dict):
            raise ValueError(f"workers[{index}].monitor must be a JSON object")
        monitor_url = normalize_http_base_url(
            raw_monitor.get("url"), f"workers[{index}].monitor.url"
        )
        if monitor_url in monitor_urls:
            raise ValueError(f"duplicate monitor URL: {monitor_url}")
        monitor_urls.add(monitor_url)
        required = raw_monitor.get("required", True)
        if not isinstance(required, bool):
            raise ValueError(f"workers[{index}].monitor.required must be a boolean")
        monitor = {"url": monitor_url, "required": required}
        workers.append(
            {
                "worker_id": worker_id,
                "ip": ip,
                "port": port,
                "request_type": request_type,
                "weight": weight,
                "tp_size": tp_size,
                "gpu_ids": raw_gpu_ids,
                "health_url": health_url,
                "start_command": start_command,
                "monitor": monitor,
            }
        )
    return {
        "phase": phase,
        "source": str(path),
        "configuration_id": configuration_id,
        "start_commands": _normalize_command_list(
            payload.get("start_commands"), "start_commands"
        ),
        "stop_commands": _normalize_command_list(
            payload.get("stop_commands"), "stop_commands"
        ),
        "workers": workers,
    }


def validate_distinct_phase_configuration_ids() -> None:
    """Prevent phase workers from being merged into one registry configuration."""
    if 1 not in phase_config_paths or 2 not in phase_config_paths:
        return
    phase1 = load_phase_config(1)
    phase2 = load_phase_config(2)
    if phase1["configuration_id"] == phase2["configuration_id"]:
        raise ValueError(
            "phase 1 and phase 2 configuration_id values must be different"
        )


def register_phase_workers(
    phase: int, selected_worker_ids: set[str] | None
) -> dict[str, Any]:
    validate_distinct_phase_configuration_ids()
    config = load_phase_config(phase)
    available = {worker["worker_id"] for worker in config["workers"]}
    if selected_worker_ids is not None:
        unknown = selected_worker_ids - available
        if unknown:
            raise ValueError(
                f"unknown worker_id values for phase {phase}: {sorted(unknown)}"
            )
    registered = []
    for definition in config["workers"]:
        if (
            selected_worker_ids is not None
            and definition["worker_id"] not in selected_worker_ids
        ):
            continue
        worker, action = worker_registry.add(
            definition["ip"],
            definition["port"],
            definition["request_type"],
            definition["weight"],
            configuration_id=config["configuration_id"],
            tp_size=definition["tp_size"],
        )
        registered.append(
            {
                "worker_id": definition["worker_id"],
                "gpu_ids": definition["gpu_ids"],
                "action": action,
                "worker": asdict(worker),
            }
        )
    expected_endpoints = {
        (worker["ip"], worker["port"]) for worker in config["workers"]
    }
    current_endpoints = {
        (worker.ip, worker.port)
        for worker in worker_registry.list(configuration_id=config["configuration_id"])
    }
    return {
        "phase": phase,
        "configuration_id": config["configuration_id"],
        "registered": registered,
        "all_configured_workers_registered": expected_endpoints <= current_endpoints,
    }


@dataclass(frozen=True)
class DatasetRequest:
    prompt: str
    prompt_tokens: int
    reference_output_tokens: int
    max_tokens: int


@dataclass
class PendingRequest:
    """One logical arrival, retained across a cold TP switch."""

    task_id: str
    ground_truth_request_type: str
    classified_request_type: str | None
    source_request_type: str
    sample: DatasetRequest
    arrival_time_mono: float
    arrival_time_utc: str
    arrival_elapsed_s: float
    phase: int
    predicted_output_tokens: int | None = None
    prediction_latency_s: float | None = None
    prediction_started_mono: float | None = None
    attempt_count: int = 0
    first_dispatch_time_mono: float | None = None
    last_dispatch_time_mono: float | None = None
    last_dispatch_generation: int = 0
    retry_errors: list[str] | None = None
    active_attempt_id: int = 0
    state: str = "pending"
    abandoned: bool = False
    recovery_enqueued: bool = False
    last_worker: Worker | None = None
    last_attempt_started_utc: str | None = None


def load_dataset_requests(path: Path) -> list[DatasetRequest]:
    """Load only fields needed to issue requests from a JSONL dataset."""
    requests_to_send: list[DatasetRequest] = []
    invalid_rows = 0
    try:
        dataset_file = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot open dataset {path}: {exc}") from exc

    with dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                prompt = record["prompt"]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("prompt is empty")

                raw_prompt_tokens = record["input_token_count"]
                raw_output_tokens = record["output_token_count"]
                if isinstance(raw_prompt_tokens, bool) or isinstance(
                    raw_output_tokens, bool
                ):
                    raise ValueError("token count is not an integer")
                prompt_tokens = int(raw_prompt_tokens)
                reference_output_tokens = int(raw_output_tokens)
                if prompt_tokens < 0 or reference_output_tokens < 0:
                    raise ValueError("token count is negative")

                max_tokens = DEFAULT_MAX_TOKENS
                if USE_DATASET_OUTPUT_TOKENS:
                    max_tokens = reference_output_tokens
                max_tokens = max(MIN_MAX_TOKENS, min(MAX_MAX_TOKENS, max_tokens))
                requests_to_send.append(
                    DatasetRequest(
                        prompt=prompt.strip(),
                        prompt_tokens=prompt_tokens,
                        reference_output_tokens=reference_output_tokens,
                        max_tokens=max_tokens,
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                invalid_rows += 1
                LOGGER.debug("ignored invalid dataset row %s:%d", path, line_number)

    if not requests_to_send:
        raise RuntimeError(f"dataset contains no usable requests: {path}")
    LOGGER.info(
        "loaded %d requests from %s%s",
        len(requests_to_send),
        path.name,
        f" ({invalid_rows} invalid rows ignored)" if invalid_rows else "",
    )
    return requests_to_send


def validate_request_configs(
    configs: dict[str, dict[str, Any]],
    duration_seconds: float = EXPERIMENT_DURATION_SECONDS,
    segment_seconds: float = QPS_SEGMENT_SECONDS,
) -> None:
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or duration_seconds <= 0
    ):
        raise RuntimeError("experiment duration must be a positive number")
    if (
        isinstance(segment_seconds, bool)
        or not isinstance(segment_seconds, (int, float))
        or segment_seconds <= 0
    ):
        raise RuntimeError("QPS segment duration must be a positive number")

    expected_segments = math.ceil(duration_seconds / segment_seconds)
    for request_type, config in configs.items():
        if (
            request_type not in TTFT_SECONDS_BY_REQUEST_TYPE
            or request_type not in TBT_SECONDS_BY_REQUEST_TYPE
        ):
            raise RuntimeError(
                f"no TTFT/TBT configured for request type: {request_type}"
            )
        profile = config.get("qps")
        if not isinstance(profile, list) or len(profile) != expected_segments:
            raise RuntimeError(
                f"{request_type} QPS profile must contain exactly {expected_segments} values "
                f"for the configured experiment duration"
            )
        for qps in profile:
            if (
                isinstance(qps, bool)
                or not isinstance(qps, (int, float))
                or not math.isfinite(qps)
                or qps < 0
            ):
                raise RuntimeError(
                    f"{request_type} QPS values must be non-negative numbers"
                )


def estimate_latency_seconds(request_type: str, reference_output_tokens: int) -> float:
    ttft = TTFT_SECONDS_BY_REQUEST_TYPE[request_type]
    tbt = TBT_SECONDS_BY_REQUEST_TYPE[request_type]
    return ttft + reference_output_tokens * tbt


class CompletionLog:
    """Concurrency-safe JSONL writer that flushes every completed request."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._closed = False
        LOGGER.info("completion log initialized at %s", self.path)

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            if self._closed:
                raise RuntimeError("completion log is already closed")
            self._file.write(line)
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._file.close()
                self._closed = True


class TrafficGenerator:
    """Generate logical arrivals and dispatch them independently to vLLM."""

    def __init__(
        self,
        registry: WorkerRegistry,
        *,
        predictor: Any,
        input_threshold: int,
        configs: dict[str, dict[str, Any]] | None = None,
        completion_log_path: Path = COMPLETION_LOG_FILE,
        request_sender_threads: int = DEFAULT_REQUEST_SENDER_THREADS,
        max_in_flight_requests: int = DEFAULT_MAX_IN_FLIGHT_REQUESTS,
        max_pending_requests_per_type: int = DEFAULT_MAX_PENDING_REQUESTS_PER_TYPE,
        output_threshold: int = 444,
        max_prediction_pending: int = 20000,
        experiment_duration_seconds: float = EXPERIMENT_DURATION_SECONDS,
        qps_segment_seconds: float = QPS_SEGMENT_SECONDS,
        experiment_id: str = "experiment",
        switch_at_seconds: float | None = None,
    ) -> None:
        if request_sender_threads <= 0:
            raise ValueError("request_sender_threads must be greater than zero")
        if max_in_flight_requests < request_sender_threads:
            raise ValueError(
                "max_in_flight_requests must be at least request_sender_threads"
            )
        if max_pending_requests_per_type <= 0:
            raise ValueError("max_pending_requests_per_type must be greater than zero")
        if predictor is None or not callable(getattr(predictor, "predict", None)):
            raise ValueError("a predictor exposing predict(prompt_text) is required")
        if input_threshold <= 0 or output_threshold <= 0 or max_prediction_pending <= 0:
            raise ValueError("thresholds and prediction capacity must be positive")
        self.predictor = predictor
        self.input_threshold = input_threshold
        self.output_threshold = output_threshold
        self.max_prediction_pending = max_prediction_pending
        self._prediction_queue = queue.Queue(maxsize=max_prediction_pending)

        self.registry = registry
        self.configs = configs if configs is not None else REQUEST_CONFIGS
        self.request_sender_threads = request_sender_threads
        self.max_in_flight_requests = max_in_flight_requests
        self.max_pending_requests_per_type = max_pending_requests_per_type
        self.scenario = "cold-switch"
        self.experiment_duration_seconds = experiment_duration_seconds
        self.qps_segment_seconds = qps_segment_seconds
        self.experiment_id = experiment_id
        self.switch_at_seconds = (
            experiment_duration_seconds / 2
            if switch_at_seconds is None
            else switch_at_seconds
        )
        if not 0 < self.switch_at_seconds < experiment_duration_seconds:
            raise ValueError(
                "switch_at_seconds must be between zero and the experiment duration"
            )
        validate_request_configs(
            self.configs,
            duration_seconds=self.experiment_duration_seconds,
            segment_seconds=self.qps_segment_seconds,
        )
        self.datasets = {
            request_type: load_dataset_requests(Path(config["dataset"]))
            for request_type, config in self.configs.items()
        }
        self._pending_queues: dict[str, queue.Queue[PendingRequest]] = {
            request_type: queue.Queue(maxsize=self.max_pending_requests_per_type)
            for request_type in self.configs
        }
        # Requests interrupted exactly at a cold boundary are kept separately
        # and always dispatched before requests arriving while vLLM starts.
        self._recovery_queues: dict[str, queue.Queue[PendingRequest]] = {
            request_type: queue.Queue() for request_type in self.configs
        }
        self._stop_event = threading.Event()
        self._activated_event = threading.Event()
        self._arrival_enabled = threading.Event()
        self._arrival_enabled.set()
        self._dispatch_enabled = threading.Event()
        self._dispatch_enabled.set()
        self._generation_finished_event = threading.Event()
        self._started_at: float | None = None
        self._clock_accumulated_s = 0.0
        self._clock_run_started_mono: float | None = None
        self._clock_running = False
        self._arrival_generation = 0
        self._start_lock = threading.Lock()
        self._control_lock = threading.RLock()
        self._dispatch_generation = 0
        self._switch_in_progress = False
        self._switch_boundary_reached = False
        self._end_boundary_reached = False
        self._experiment_state = "waiting_to_start"
        self._pause_reason: str | None = None
        self._paused_at_mono: float | None = None
        self._run_id = uuid.uuid4().hex
        self._monitor_sessions: dict[int, dict[str, str]] = {}
        self._monitor_workers: dict[int, list[dict[str, Any]]] = {}
        self._monitor_status: dict[int, dict[str, Any]] = {}
        self._cold_monitor_transitioned_workers: set[str] = set()
        self._threads: list[threading.Thread] = []
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.request_sender_threads,
            thread_name_prefix="vllm-request",
        )
        # Do not put a pre-routed backlog into ThreadPoolExecutor's FIFO queue.
        # Excess work stays in the pending/recovery queues until a sender is free.
        self._capacity = threading.BoundedSemaphore(self.request_sender_threads)
        self._sender_active: dict[tuple[str, str, int], int] = {}
        self._sender_plans: dict[str, dict[tuple[str, str, int], int]] = {}
        self._retired_executors: list[concurrent.futures.ThreadPoolExecutor] = []
        self._tasks_lock = threading.RLock()
        self._active_tasks: dict[str, PendingRequest] = {}
        self._stats_lock = threading.Lock()
        self._stats = {
            request_type: {
                "arrived": 0,
                "submitted": 0,
                "misclassified": 0,
                "succeeded": 0,
                "failed": 0,
                "dropped": 0,
                "retried": 0,
                "interrupted": 0,
                "abandoned": 0,
                "pending": 0,
                "in_flight": 0,
            }
            for request_type in self.configs
        }
        self._model_cache: dict[str, str] = {}
        self._model_lock = threading.Lock()
        self._request_log = CompletionLog(completion_log_path)
        event_log_path = completion_log_path.with_name(
            f"{completion_log_path.stem}_events{completion_log_path.suffix or '.jsonl'}"
        )
        self._event_log = CompletionLog(event_log_path)
        switch_log_path = completion_log_path.with_name(
            f"{completion_log_path.stem}_switch{completion_log_path.suffix or '.jsonl'}"
        )
        self._switch_log = CompletionLog(switch_log_path)

    @staticmethod
    def _monitor_json_request(
        url: str,
        method: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=MONITOR_CONTROL_TIMEOUT_SECONDS
            ) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace")
            raise RuntimeError(f"{url} returned HTTP {exc.code}: {detail}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot contact {url}: {exc}") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"{url} returned a non-object JSON response")
        return result

    def _configured_monitors(self, phase: int) -> list[dict[str, Any]]:
        if phase not in phase_config_paths:
            return []
        config = load_phase_config(phase)
        # Capture the complete phase roster, including workers not yet healthy.
        # This prevents the first ready worker from monopolizing all senders.
        self._sender_plans[config["configuration_id"]] = self._allocate_sender_limits(
            [
                (
                    (config["configuration_id"], worker["ip"], worker["port"]),
                    worker["weight"],
                )
                for worker in config["workers"]
            ]
        )
        return [worker for worker in config["workers"] if worker["monitor"] is not None]

    def _allocate_sender_limits(
        self, weighted_keys: list[tuple[tuple[str, str, int], float]]
    ) -> dict[tuple[str, str, int], int]:
        """Reserve at least one sender per configured endpoint, then weight the rest."""
        if not weighted_keys:
            return {}
        if len(weighted_keys) > self.request_sender_threads:
            raise ValueError(
                "sender_threads must be at least the number of configured workers"
            )
        remaining = self.request_sender_threads - len(weighted_keys)
        total = sum(weight for _, weight in weighted_keys)
        shares = {key: remaining * weight / total for key, weight in weighted_keys}
        limits = {key: 1 + int(share) for key, share in shares.items()}
        spare = self.request_sender_threads - sum(limits.values())
        order = sorted(shares, key=lambda key: (-(shares[key] - int(shares[key])), key))
        for key in order[:spare]:
            limits[key] += 1
        return limits

    def _sender_limits(self) -> dict[tuple[str, str, int], int]:
        configuration_id = self.registry.active_configuration_id
        planned = self._sender_plans.get(configuration_id)
        if planned is not None:
            return planned
        # Compatibility for programmatic registries without phase JSON.
        return self._allocate_sender_limits(
            [
                ((worker.configuration_id, worker.ip, worker.port), worker.weight)
                for worker in self.registry.list(configuration_id=configuration_id)
            ]
        )

    def _control_one_monitor(
        self,
        worker: dict[str, Any],
        action: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        monitor = worker["monitor"]
        assert monitor is not None
        base_url = monitor["url"]
        if action == "start":
            health = self._monitor_json_request(f"{base_url}/health", "GET")
            if health.get("worker_id") != worker["worker_id"]:
                raise RuntimeError(
                    f"{base_url} identifies as worker_id={health.get('worker_id')!r}, "
                    f"expected {worker['worker_id']!r}"
                )
            if health.get("gpu_ids") != worker["gpu_ids"]:
                raise RuntimeError(
                    f"{base_url} reports gpu_ids={health.get('gpu_ids')!r}, "
                    f"expected {worker['gpu_ids']!r}"
                )
        response = self._monitor_json_request(
            f"{base_url}/recording/{action}", "POST", payload
        )
        return {
            "worker_id": worker["worker_id"],
            "monitor_url": base_url,
            "required": monitor["required"],
            "status": "ok",
            "response": response,
        }

    def _run_monitor_action(
        self,
        phase: int,
        action: str,
        workers: list[dict[str, Any]],
        payloads: dict[str, dict[str, Any]],
        recording_window: str,
    ) -> list[dict[str, Any]]:
        if not workers:
            return []
        results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(workers), thread_name_prefix=f"monitor-{action}"
        ) as executor:
            futures = {
                executor.submit(
                    self._control_one_monitor,
                    worker,
                    action,
                    payloads[worker["worker_id"]],
                ): worker
                for worker in workers
            }
            for future in concurrent.futures.as_completed(futures):
                worker = futures[future]
                monitor = worker["monitor"]
                assert monitor is not None
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        {
                            "worker_id": worker["worker_id"],
                            "monitor_url": monitor["url"],
                            "required": monitor["required"],
                            "status": "error",
                            "error": str(exc),
                        }
                    )
        results.sort(key=lambda item: item["worker_id"])
        self._switch_log.write(
            {
                "record_type": "monitor_event",
                "event": f"phase_monitor_{action}",
                "timestamp": utc_now(),
                "experiment_id": self.experiment_id,
                "scenario": self.scenario,
                "phase": phase,
                "recording_window": recording_window,
                "experiment_elapsed_s": self.elapsed_seconds(),
                "results": results,
            }
        )
        return results

    def _record_monitor_status(
        self,
        phase: int,
        action: str,
        recording_window: str,
        results: list[dict[str, Any]],
    ) -> None:
        status = self._monitor_status.setdefault(phase, {"history": []})
        status["recording"] = bool(self._monitor_sessions.get(phase))
        status["results"] = results
        status["recording_window"] = recording_window
        status["history"].append(
            {
                "action": action,
                "recording_window": recording_window,
                "timestamp": utc_now(),
                "results": results,
            }
        )

    @staticmethod
    def _required_monitor_errors(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            item for item in results if item["required"] and item["status"] == "error"
        ]

    def _start_monitor_workers(
        self,
        phase: int,
        workers: list[dict[str, Any]],
        logical_start_seconds: float,
        recording_window: str,
    ) -> list[dict[str, Any]]:
        sessions = self._monitor_sessions.setdefault(phase, {})
        payloads: dict[str, dict[str, Any]] = {}
        expected_duration = (
            self.switch_at_seconds - logical_start_seconds
            if phase == 1
            else self.experiment_duration_seconds - logical_start_seconds
        )
        for worker in workers:
            worker_id = worker["worker_id"]
            session_id = f"{self.experiment_id}:{self._run_id}:phase{phase}:{recording_window}:{worker_id}"
            payloads[worker_id] = {
                "session_id": session_id,
                "experiment_id": self.experiment_id,
                "scenario": self.scenario,
                "phase": phase,
                "recording_window": recording_window,
                "logical_start_seconds": logical_start_seconds,
                "expected_duration_seconds": max(expected_duration, 0.0),
            }
        results = self._run_monitor_action(
            phase, "start", workers, payloads, recording_window
        )
        for result in results:
            worker_id = result["worker_id"]
            if result["status"] == "ok":
                sessions[worker_id] = payloads[worker_id]["session_id"]
            else:
                sessions.pop(worker_id, None)
        # Keep this dictionary stable: other workers may be concurrently
        # completing a stop/start transition for the same phase.
        self._record_monitor_status(phase, "start", recording_window, results)
        return results

    def _start_phase_monitors(
        self,
        phase: int,
        logical_start_seconds: float,
        recording_window: str,
    ) -> None:
        workers = self._configured_monitors(phase)
        results = self._start_monitor_workers(
            phase,
            workers,
            logical_start_seconds,
            recording_window,
        )
        self._monitor_workers[phase] = workers
        required_errors = [
            item for item in results if item["required"] and item["status"] == "error"
        ]
        if required_errors:
            details = "; ".join(
                f"{item['worker_id']}: {item['error']}" for item in required_errors
            )
            raise RuntimeError(
                f"required phase {phase} monitors did not start: {details}"
            )

    def _stop_monitor_workers(
        self,
        phase: int,
        workers: list[dict[str, Any]],
        reason: str,
        recording_window: str,
    ) -> list[dict[str, Any]]:
        sessions = self._monitor_sessions.get(phase, {})
        selected = [worker for worker in workers if worker["worker_id"] in sessions]
        if not selected:
            return []
        payloads = {
            worker["worker_id"]: {
                "session_id": sessions[worker["worker_id"]],
                "reason": reason,
            }
            for worker in selected
        }
        results = self._run_monitor_action(
            phase,
            "stop",
            selected,
            payloads,
            recording_window,
        )
        for result in results:
            if result["status"] == "ok":
                sessions.pop(result["worker_id"], None)
        # Keep this dictionary stable: other workers may be concurrently
        # completing a stop/start transition for the same phase.
        self._record_monitor_status(phase, "stop", recording_window, results)
        return results

    def _stop_phase_monitors(self, phase: int, reason: str) -> None:
        workers = self._monitor_workers.get(phase, [])
        results = self._stop_monitor_workers(
            phase,
            workers,
            reason,
            f"phase{phase}-complete",
        )
        failures = [item for item in results if item["status"] == "error"]
        if failures:
            LOGGER.error("phase %d monitor stop failures: %s", phase, failures)

    def start(self) -> None:
        LOGGER.info(
            "traffic configured: scenario=%s sender_threads=%d max_in_flight=%d "
            "max_pending_per_type=%d",
            self.scenario,
            self.request_sender_threads,
            self.max_in_flight_requests,
            self.max_pending_requests_per_type,
        )
        for request_type in self.configs:
            for target, prefix in (
                (self._schedule_loop, "qps"),
                (self._dispatch_loop, "dispatch"),
            ):
                thread = threading.Thread(
                    target=target,
                    args=(request_type,),
                    name=f"{prefix}-{request_type}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
        prediction_thread = threading.Thread(
            target=self._prediction_loop, name="predictor", daemon=True
        )
        prediction_thread.start()
        self._threads.append(prediction_thread)
        boundary_thread = threading.Thread(
            target=self._switch_boundary_loop,
            name="switch-boundary",
            daemon=True,
        )
        boundary_thread.start()
        self._threads.append(boundary_thread)
        end_thread = threading.Thread(
            target=self._experiment_end_loop,
            name="experiment-end-boundary",
            daemon=True,
        )
        end_thread.start()
        self._threads.append(end_thread)
        LOGGER.info("traffic generator ready; start it via POST /traffic/start")

    def activate(self) -> bool:
        with self._start_lock:
            if self._started_at is not None:
                return False
            # Phase-1 monitor setup is outside logical experiment time.  The
            # clock starts only after all required monitor services acknowledge.
            self._start_phase_monitors(1, 0.0, "phase1-running")
            now = time.monotonic()
            self._started_at = now
            with self._control_lock:
                self._clock_accumulated_s = 0.0
                self._clock_run_started_mono = now
                self._clock_running = True
                self._experiment_state = "phase1_running"
                self._arrival_enabled.set()
                self._dispatch_enabled.set()
            self._activated_event.set()
            self._event_log.write(
                {
                    "event": "traffic_started",
                    "timestamp": utc_now(),
                    "experiment_id": self.experiment_id,
                    "scenario": self.scenario,
                    "input_threshold": self.input_threshold,
                    "output_threshold": self.output_threshold,
                    "active_configuration_id": self.registry.active_configuration_id,
                }
            )
            LOGGER.info("traffic started with online output-length prediction")
            return True

    def stop(self) -> None:
        self._stop_event.set()
        self._activated_event.set()
        self._arrival_enabled.set()
        self._dispatch_enabled.set()
        for thread in self._threads:
            thread.join(timeout=2)
        for phase, monitor_state in list(self._monitor_status.items()):
            if monitor_state.get("recording"):
                self._stop_phase_monitors(phase, "router_stopped")
        wait_for_requests = not self._end_boundary_reached
        self._executor.shutdown(
            wait=wait_for_requests,
            cancel_futures=self._end_boundary_reached,
        )
        for executor in self._retired_executors:
            executor.shutdown(wait=wait_for_requests, cancel_futures=True)
        self._event_log.write(
            {
                "event": "router_stopped",
                "timestamp": utc_now(),
                "experiment_id": self.experiment_id,
                "status": self.status(),
            }
        )
        self._request_log.close()
        self._event_log.close()
        self._switch_log.close()

    def _clock_elapsed_locked(self) -> float:
        if self._started_at is None:
            return 0.0
        if not self._clock_running or self._clock_run_started_mono is None:
            return self._clock_accumulated_s
        return (
            self._clock_accumulated_s + time.monotonic() - self._clock_run_started_mono
        )

    def _pause_clock_locked(self, clamp_seconds: float | None = None) -> None:
        elapsed = self._clock_elapsed_locked()
        self._clock_accumulated_s = elapsed if clamp_seconds is None else clamp_seconds
        self._clock_run_started_mono = None
        self._clock_running = False

    def _resume_clock_locked(self) -> None:
        if not self._clock_running:
            self._clock_run_started_mono = time.monotonic()
            self._clock_running = True

    def _rotate_sender_pool_locked(self) -> None:
        old_executor = self._executor
        old_executor.shutdown(wait=False, cancel_futures=True)
        self._retired_executors.append(old_executor)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.request_sender_threads,
            thread_name_prefix="vllm-request",
        )
        self._capacity = threading.BoundedSemaphore(self.request_sender_threads)
        # Old callbacks retain their own accounting dictionary and semaphore.
        self._sender_active = {}
        with self._model_lock:
            self._model_cache.clear()

    @staticmethod
    def _drain_queue(target: queue.Queue[PendingRequest]) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return
            else:
                target.task_done()

    def _switch_boundary_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._activated_event.wait(timeout=0.25):
                continue
            if self.elapsed_seconds() >= self.switch_at_seconds:
                self.begin_scheduled_switch()
                return
            self._stop_event.wait(0.02)

    def _experiment_end_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._activated_event.wait(timeout=0.25):
                continue
            if self.elapsed_seconds() >= self.experiment_duration_seconds:
                self.finish_experiment()
                return
            self._stop_event.wait(0.02)

    def begin_scheduled_switch(self) -> dict[str, Any]:
        """Freeze logical time at the phase boundary and quiesce phase 1."""
        with self._control_lock:
            if self._switch_boundary_reached:
                return self.dispatch_status()
            self._switch_boundary_reached = True
            self._switch_in_progress = True
            self._pause_reason = f"{self.scenario}_boundary"
            self._paused_at_mono = time.monotonic()
            self._dispatch_generation += 1
            self._arrival_generation += 1
            self._arrival_enabled.clear()
            self._dispatch_enabled.clear()
            self._pause_clock_locked(self.switch_at_seconds)
            self._stop_phase_monitors(1, "phase_boundary")
            with self._tasks_lock:
                active = sorted(
                    self._active_tasks.values(),
                    key=lambda task: (task.arrival_time_mono, task.task_id),
                )
                pending_counts = {request_type: 0 for request_type in self.configs}
                inflight_counts = {request_type: 0 for request_type in self.configs}
                for task in active:
                    request_type = task.source_request_type
                    if task.state == "in_flight":
                        inflight_counts[request_type] += 1
                    else:
                        pending_counts[request_type] += 1
                for pending_queue in self._pending_queues.values():
                    self._drain_queue(pending_queue)
                for recovery_queue in self._recovery_queues.values():
                    self._drain_queue(recovery_queue)
                for task in active:
                    request_type = task.source_request_type
                    was_in_flight = task.state == "in_flight"
                    task.active_attempt_id += 1
                    if task.classified_request_type is None:
                        continue  # Still queued/running in the single ordered predictor.
                    task.state = "recovery"
                    task.recovery_enqueued = True
                    if was_in_flight:
                        if task.retry_errors is None:
                            task.retry_errors = []
                        task.retry_errors.append("interrupted by cold switch")
                        self._switch_log.write(
                            {
                                "record_type": "cold_interrupted_attempt",
                                "task_id": task.task_id,
                                "request_type": request_type,
                                "attempt_count": task.attempt_count,
                                "worker": (
                                    asdict(task.last_worker)
                                    if task.last_worker is not None
                                    else None
                                ),
                                "attempt_started_utc": task.last_attempt_started_utc,
                                "dispatch_experiment_elapsed_s": (
                                    None
                                    if task.last_dispatch_time_mono is None
                                    or self._started_at is None
                                    else task.last_dispatch_time_mono - self._started_at
                                ),
                                "switch_experiment_elapsed_s": self.switch_at_seconds,
                                "active_attempt_duration_s": (
                                    None
                                    if task.last_dispatch_time_mono is None
                                    or self._started_at is None
                                    else max(
                                        self.switch_at_seconds
                                        - (
                                            task.last_dispatch_time_mono
                                            - self._started_at
                                        ),
                                        0.0,
                                    )
                                ),
                                "outcome": "interrupted_by_cold_switch",
                            }
                        )
                    self._recovery_queues[task.classified_request_type].put_nowait(task)
                self._experiment_state = "cold_waiting_for_gpu"
                with self._stats_lock:
                    for request_type, values in self._stats.items():
                        values["pending"] = (
                            pending_counts[request_type] + inflight_counts[request_type]
                        )
                        values["in_flight"] = 0
                        values["interrupted"] += inflight_counts[request_type]
                        values["retried"] += inflight_counts[request_type]
            self._rotate_sender_pool_locked()
            boundary_record = {
                "record_type": "switch_event",
                "event": "switch_boundary_reached",
                "timestamp": utc_now(),
                "experiment_id": self.experiment_id,
                "scenario": self.scenario,
                "experiment_elapsed_s": self.switch_at_seconds,
                "state": self._experiment_state,
                "pending_counts": pending_counts,
                "in_flight_counts": inflight_counts,
            }
            self._switch_log.write(boundary_record)
            self._event_log.write(
                {
                    "event": "dispatch_paused",
                    "timestamp": boundary_record["timestamp"],
                    "experiment_id": self.experiment_id,
                    "reason": self._pause_reason,
                    "dispatch_generation": self._dispatch_generation,
                    "active_configuration_id": self.registry.active_configuration_id,
                    "elapsed_seconds": self.switch_at_seconds,
                }
            )
            LOGGER.warning(
                "%s boundary reached at logical %.3fs",
                self.scenario,
                self.switch_at_seconds,
            )
            return self.dispatch_status()

    def finish_experiment(self) -> dict[str, Any]:
        """Apply the logical-duration hard cutoff without draining requests."""
        with self._control_lock:
            if self._end_boundary_reached:
                return self.dispatch_status()
            self._end_boundary_reached = True
            self._arrival_generation += 1
            self._dispatch_generation += 1
            self._arrival_enabled.clear()
            self._dispatch_enabled.clear()
            self._pause_clock_locked(self.experiment_duration_seconds)
            self._switch_in_progress = False
            self._pause_reason = "experiment_complete"
            self._paused_at_mono = None
            self._experiment_state = "completed"
            current_phase = 2
            self._stop_phase_monitors(current_phase, "experiment_complete")
            with self._tasks_lock:
                active = list(self._active_tasks.values())
                abandoned_counts = {request_type: 0 for request_type in self.configs}
                pending_counts = {request_type: 0 for request_type in self.configs}
                in_flight_counts = {request_type: 0 for request_type in self.configs}
                for task in active:
                    request_type = task.source_request_type
                    abandoned_counts[request_type] += 1
                    if task.state == "in_flight":
                        in_flight_counts[request_type] += 1
                    else:
                        pending_counts[request_type] += 1
                    task.abandoned = True
                    task.state = "abandoned"
                    task.active_attempt_id += 1
                self._active_tasks.clear()
                for target_queue in self._pending_queues.values():
                    self._drain_queue(target_queue)
                for target_queue in self._recovery_queues.values():
                    self._drain_queue(target_queue)
            with self._stats_lock:
                for request_type, values in self._stats.items():
                    values["abandoned"] += abandoned_counts[request_type]
                    values["pending"] = 0
                    values["in_flight"] = 0
            self._drain_queue(self._prediction_queue)
            self._generation_finished_event.set()
            self._switch_log.write(
                {
                    "record_type": "experiment_event",
                    "event": "experiment_hard_cutoff",
                    "timestamp": utc_now(),
                    "experiment_id": self.experiment_id,
                    "scenario": self.scenario,
                    "experiment_elapsed_s": self.experiment_duration_seconds,
                    "abandoned_counts": abandoned_counts,
                    "pending_counts": pending_counts,
                    "in_flight_counts": in_flight_counts,
                }
            )
        LOGGER.info(
            "experiment reached hard cutoff at logical %.3fs; unfinished requests abandoned",
            self.experiment_duration_seconds,
        )
        return self.dispatch_status()

    def cold_resource_ready(self) -> dict[str, Any]:
        """Exclude GPU rental time, then count vLLM startup as phase-2 time."""
        with self._control_lock:
            if self._experiment_state != "cold_waiting_for_gpu":
                raise ValueError(
                    "cold resource-ready is only valid while waiting for a phase-2 GPU"
                )
            self._start_phase_monitors(2, self.switch_at_seconds, "phase2-vllm-startup")
            self._resume_clock_locked()
            self._arrival_generation += 1
            self._arrival_enabled.set()
            self._experiment_state = "cold_starting_vllm"
            record = {
                "record_type": "switch_event",
                "event": "cold_resource_ready",
                "timestamp": utc_now(),
                "experiment_id": self.experiment_id,
                "experiment_elapsed_s": self._clock_elapsed_locked(),
            }
            self._switch_log.write(record)
            return self.dispatch_status()

    def cold_worker_ready(self, worker_id: str) -> dict[str, Any]:
        """Rotate one phase-2 Monitor from startup to steady-state recording."""
        with self._control_lock:
            if self._experiment_state not in {"cold_starting_vllm", "phase2_running"}:
                raise ValueError(
                    "cold worker-ready is valid only after phase-2 resources are ready"
                )
            if worker_id in self._cold_monitor_transitioned_workers:
                return {
                    "worker_id": worker_id,
                    "already_transitioned": True,
                    "experiment_elapsed_s": self._clock_elapsed_locked(),
                }
            workers = self._monitor_workers.get(2, [])
            worker = next(
                (
                    definition
                    for definition in workers
                    if definition["worker_id"] == worker_id
                ),
                None,
            )
            if worker is None:
                raise ValueError(f"unknown phase-2 worker_id: {worker_id}")
        stop_results = self._stop_monitor_workers(
            2, [worker], "vllm_ready", "phase2-vllm-startup"
        )
        required_stop_errors = self._required_monitor_errors(stop_results)
        if required_stop_errors:
            raise RuntimeError(
                f"required startup Monitor did not stop for {worker_id}: {required_stop_errors[0]['error']}"
            )
        with self._control_lock:
            if self._end_boundary_reached:
                raise ValueError("experiment ended while rotating the worker Monitor")
            logical_start = self._clock_elapsed_locked()
        start_results = self._start_monitor_workers(
            2, [worker], logical_start, "phase2-running"
        )
        required_start_errors = self._required_monitor_errors(start_results)
        if required_start_errors:
            raise RuntimeError(
                f"required steady-state Monitor did not start for {worker_id}: {required_start_errors[0]['error']}"
            )
        with self._control_lock:
            if self._end_boundary_reached:
                # Cutoff may happen while the start HTTP request is in flight.
                # Stop the just-acknowledged session instead of leaving it live.
                self._stop_monitor_workers(
                    2, [worker], "experiment_complete", "phase2-running"
                )
                raise ValueError("experiment ended while starting the worker Monitor")
            self._cold_monitor_transitioned_workers.add(worker_id)
        record = {
            "record_type": "switch_event",
            "event": "cold_worker_monitor_transitioned",
            "timestamp": utc_now(),
            "experiment_id": self.experiment_id,
            "worker_id": worker_id,
            "experiment_elapsed_s": logical_start,
            "stop_results": stop_results,
            "start_results": start_results,
        }
        self._switch_log.write(record)
        return {
            "worker_id": worker_id,
            "already_transitioned": False,
            "experiment_elapsed_s": logical_start,
            "stop_results": stop_results,
            "start_results": start_results,
        }

    def resume_dispatch(self) -> dict[str, Any]:
        with self._control_lock:
            if self._experiment_state != "cold_starting_vllm":
                raise ValueError("cold dispatch can resume only after the GPU is ready")
            paused_for = None
            if self._paused_at_mono is not None:
                paused_for = time.monotonic() - self._paused_at_mono
            self._switch_in_progress = False
            self._pause_reason = None
            self._paused_at_mono = None
            self._experiment_state = "phase2_running"
            self._dispatch_enabled.set()
            self._event_log.write(
                {
                    "event": "dispatch_resumed",
                    "timestamp": utc_now(),
                    "experiment_id": self.experiment_id,
                    "dispatch_generation": self._dispatch_generation,
                    "active_configuration_id": self.registry.active_configuration_id,
                    "pause_duration_seconds": paused_for,
                    "elapsed_seconds": self.elapsed_seconds(),
                }
            )
            LOGGER.info("request dispatch resumed")
            return self.dispatch_status()

    def activate_configuration(self, configuration_id: str) -> dict[str, Any]:
        previous, current = self.registry.activate(configuration_id)
        with self._model_lock:
            self._model_cache.clear()
        self._event_log.write(
            {
                "event": "configuration_activated",
                "timestamp": utc_now(),
                "experiment_id": self.experiment_id,
                "previous_configuration_id": previous,
                "configuration_id": current,
                "dispatch_generation": self._dispatch_generation,
                "elapsed_seconds": self.elapsed_seconds(),
            }
        )
        LOGGER.info(
            "active worker configuration changed from %s to %s", previous, current
        )
        return {
            "previous_configuration_id": previous,
            "active_configuration_id": current,
        }

    def dispatch_status(self) -> dict[str, Any]:
        with self._control_lock:
            paused_for = None
            if self._paused_at_mono is not None:
                paused_for = time.monotonic() - self._paused_at_mono
            return {
                "dispatch_enabled": self._dispatch_enabled.is_set(),
                "arrival_enabled": self._arrival_enabled.is_set(),
                "experiment_clock_running": self._clock_running,
                "experiment_state": self._experiment_state,
                "switch_in_progress": self._switch_in_progress,
                "pause_reason": self._pause_reason,
                "pause_duration_seconds": paused_for,
                "dispatch_generation": self._dispatch_generation,
            }

    def elapsed_seconds(self) -> float:
        with self._control_lock:
            return self._clock_elapsed_locked()

    def logical_monotonic(self) -> float:
        with self._control_lock:
            if self._started_at is None:
                return 0.0
            return self._started_at + self._clock_elapsed_locked()

    def _profile_position(self, request_type: str) -> tuple[int, float]:
        elapsed = self.elapsed_seconds()
        if (not self._switch_boundary_reached) and (elapsed >= self.switch_at_seconds):
            return (int(elapsed // self.qps_segment_seconds), 0.0)
        raw_segment = int(elapsed // self.qps_segment_seconds)
        profile = self.configs[request_type]["qps"]
        if elapsed >= self.experiment_duration_seconds or raw_segment >= len(profile):
            return (raw_segment, 0.0)
        return (raw_segment, float(profile[raw_segment]))

    def _schedule_loop(self, request_type: str) -> None:
        next_arrival_at = time.monotonic()
        previous_segment = None
        observed_generation = self._arrival_generation
        while not self._stop_event.is_set():
            if not self._activated_event.wait(0.1) or not self._arrival_enabled.wait(
                0.1
            ):
                continue
            if self._stop_event.is_set():
                return
            with self._control_lock:
                generation = self._arrival_generation
            if generation != observed_generation:
                next_arrival_at = time.monotonic()
                previous_segment = None
                observed_generation = generation
            segment, qps = self._profile_position(request_type)
            now = time.monotonic()
            if self.elapsed_seconds() >= self.experiment_duration_seconds:
                return
            if segment != previous_segment:
                next_arrival_at = now
                previous_segment = segment
            if qps <= 0:
                self._stop_event.wait(0.1)
                continue
            if next_arrival_at > now:
                self._stop_event.wait(min(next_arrival_at - now, 0.1))
                continue
            self.enqueue_arrival(
                request_type, random.choice(self.datasets[request_type])
            )
            next_arrival_at = now + 1.0 / qps

    def enqueue_arrival(
        self, source_type: str, sample: DatasetRequest
    ) -> PendingRequest | None:
        # Serialize timestamp assignment and insertion across arrival threads.
        # Prediction never runs in an arrival thread or under this lock.
        with self._control_lock:
            elapsed = self._clock_elapsed_locked()
            if (
                self._started_at is None
                or not self._arrival_enabled.is_set()
                or self._end_boundary_reached
                or elapsed >= self.experiment_duration_seconds
                or (
                    not self._switch_boundary_reached
                    and elapsed >= self.switch_at_seconds
                )
            ):
                return None
            task = PendingRequest(
                task_id=str(uuid.uuid4()),
                source_request_type=source_type,
                ground_truth_request_type=request_type_for(
                    sample.prompt_tokens,
                    sample.reference_output_tokens,
                    self.input_threshold,
                    self.output_threshold,
                ),
                classified_request_type=None,
                sample=sample,
                arrival_time_mono=self.logical_monotonic(),
                arrival_time_utc=utc_now(),
                arrival_elapsed_s=elapsed,
                phase=1 if elapsed < self.switch_at_seconds else 2,
                retry_errors=[],
                state="predicting",
            )
            self._change_stats(source_type, arrived=1)
            try:
                self._prediction_queue.put_nowait(task)
            except queue.Full:
                self._change_stats(source_type, dropped=1)
                self._write_terminal(
                    task, "dropped", None, error="prediction queue is full"
                )
            else:
                with self._tasks_lock:
                    self._active_tasks[task.task_id] = task
                self._change_stats(source_type, submitted=1, pending=1)
            return task

    def _prediction_loop(self) -> None:
        # One reusable model, one ordered consumer: later predictions cannot
        # overtake earlier arrivals in the same destination queue.
        while not self._stop_event.is_set():
            if not self._activated_event.wait(0.1) or not self._arrival_enabled.wait(
                0.1
            ):
                continue
            try:
                task = self._prediction_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self._control_lock:
                    if task.abandoned or self._end_boundary_reached:
                        continue
                    task.prediction_started_mono = self.logical_monotonic()
                error = None
                try:
                    prediction = validate_prediction(
                        self.predictor.predict(task.sample.prompt)
                    )
                except Exception as exc:
                    error = f"prediction failed: {exc}"
                    prediction = None
                # A prediction already running at the boundary may finish during
                # the frozen interval. Commit it only after resource-ready.
                while not self._stop_event.is_set():
                    if not self._arrival_enabled.wait(0.1):
                        continue
                    with self._control_lock:
                        if self._end_boundary_reached or task.abandoned:
                            break
                        if not self._arrival_enabled.is_set():
                            continue
                        if (
                            self._clock_elapsed_locked()
                            >= self.experiment_duration_seconds
                        ):
                            break
                        task.prediction_latency_s = (
                            self.logical_monotonic() - task.prediction_started_mono
                        )
                        if error is not None:
                            self._change_stats(
                                task.source_request_type, pending=-1, failed=1
                            )
                            self._write_terminal(task, "failed", None, error=error)
                            break
                        task.predicted_output_tokens = prediction
                        task.classified_request_type = request_type_for(
                            task.sample.prompt_tokens,
                            prediction,
                            self.input_threshold,
                            self.output_threshold,
                        )
                        self._change_stats(
                            task.source_request_type,
                            misclassified=int(
                                task.classified_request_type
                                != task.ground_truth_request_type
                            ),
                        )
                        try:
                            self._pending_queues[
                                task.classified_request_type
                            ].put_nowait(task)
                            task.state = "pending"
                        except queue.Full:
                            self._change_stats(
                                task.source_request_type, pending=-1, dropped=1
                            )
                            self._write_terminal(
                                task,
                                "dropped",
                                None,
                                error="predicted-type queue is full",
                            )
                        break
            finally:
                self._prediction_queue.task_done()

    def _dispatch_loop(self, request_type: str) -> None:
        pending_queue = self._pending_queues[request_type]
        recovery_queue = self._recovery_queues[request_type]
        while not self._stop_event.is_set():
            source_queue: queue.Queue[PendingRequest]
            try:
                task = recovery_queue.get_nowait()
                source_queue = recovery_queue
            except queue.Empty:
                try:
                    task = pending_queue.get(timeout=0.25)
                    source_queue = pending_queue
                except queue.Empty:
                    continue

            with self._tasks_lock:
                if task.abandoned:
                    source_queue.task_done()
                    continue
                if source_queue is recovery_queue:
                    task.recovery_enqueued = False
                    task.state = "pending"
                elif task.recovery_enqueued:
                    # The boundary also placed this held task into the priority
                    # recovery queue; discard this stale normal-queue copy.
                    source_queue.task_done()
                    continue
            submitted = False
            while not self._stop_event.is_set() and not submitted:
                if not self._dispatch_enabled.wait(timeout=0.25):
                    continue
                with self._control_lock:
                    capacity = self._capacity
                if not capacity.acquire(timeout=0.25):
                    continue
                worker: Worker | None = None
                with self._control_lock:
                    if (
                        not self._dispatch_enabled.is_set()
                        or capacity is not self._capacity
                    ):
                        capacity.release()
                        continue
                    with self._tasks_lock:
                        if task.abandoned or task.recovery_enqueued:
                            capacity.release()
                            break
                    elapsed = self._clock_elapsed_locked()
                    if elapsed >= self.experiment_duration_seconds or (
                        not self._switch_boundary_reached
                        and elapsed >= self.switch_at_seconds
                    ):
                        capacity.release()
                        continue
                    limits = self._sender_limits()
                    eligible = {
                        key
                        for key, limit in limits.items()
                        if self._sender_active.get(key, 0) < limit
                    }
                    worker = self.registry.choose(
                        task.classified_request_type, eligible
                    )
                    if worker is None:
                        capacity.release()
                    else:
                        submitted = self._submit_attempt(
                            worker, task, self._executor, capacity
                        )
                if worker is None:
                    self._stop_event.wait(0.02)
            if not submitted:
                with self._tasks_lock:
                    should_requeue = not task.abandoned and not task.recovery_enqueued
                if should_requeue:
                    try:
                        pending_queue.put_nowait(task)
                    except queue.Full:
                        self._change_stats(
                            task.source_request_type, pending=-1, dropped=1
                        )
                        self._write_terminal(
                            task,
                            "dropped",
                            None,
                            error="router stopped with full pending queue",
                        )
            source_queue.task_done()

    def _submit_attempt(
        self,
        worker: Worker,
        task: PendingRequest,
        executor: concurrent.futures.ThreadPoolExecutor,
        capacity: threading.BoundedSemaphore,
    ) -> bool:
        sender_key = (worker.configuration_id, worker.ip, worker.port)
        sender_active = self._sender_active
        sender_active[sender_key] = sender_active.get(sender_key, 0) + 1

        def release_sender() -> None:
            with self._control_lock:
                sender_active[sender_key] -= 1
                capacity.release()

        task.attempt_count += 1
        task.active_attempt_id += 1
        attempt_id = task.active_attempt_id
        task.last_dispatch_time_mono = self.logical_monotonic()
        if task.first_dispatch_time_mono is None:
            task.first_dispatch_time_mono = task.last_dispatch_time_mono
        task.last_dispatch_generation = self._dispatch_generation
        task.last_worker = worker
        task.last_attempt_started_utc = utc_now()
        task.state = "in_flight"
        self._change_stats(task.source_request_type, pending=-1, in_flight=1)
        try:
            future = executor.submit(self._send_request, worker, task.sample)
        except RuntimeError as exc:
            release_sender()
            self._change_stats(task.source_request_type, in_flight=-1, failed=1)
            self._write_terminal(task, "failed", worker, error=str(exc))
            return True

        def finish_request(done: concurrent.futures.Future[int]) -> None:
            try:
                error = done.exception()
            except concurrent.futures.CancelledError as exc:
                error = exc
            if error is None:
                actual_output_tokens = done.result()
                with self._control_lock:
                    if (
                        self._end_boundary_reached
                        or self._clock_elapsed_locked()
                        >= self.experiment_duration_seconds
                    ):
                        return
                    with self._tasks_lock:
                        if task.abandoned or attempt_id != task.active_attempt_id:
                            return
                        task.state = "terminal"
                        self._active_tasks.pop(task.task_id, None)
                    self._change_stats(
                        task.source_request_type, succeeded=1, in_flight=-1
                    )
                    self._write_terminal(
                        task,
                        "succeeded",
                        worker,
                        actual_output_tokens=actual_output_tokens,
                    )
                return
            error_text = str(error)
            with self._control_lock:
                if (
                    self._end_boundary_reached
                    or self._clock_elapsed_locked() >= self.experiment_duration_seconds
                ):
                    return
                with self._tasks_lock:
                    if task.abandoned or attempt_id != task.active_attempt_id:
                        return
                self._change_stats(task.source_request_type, failed=1, in_flight=-1)
                task.state = "terminal"
                self._write_terminal(task, "failed", worker, error=error_text)
                LOGGER.warning("request to %s failed: %s", worker.base_url, error_text)

        def request_finished(done: concurrent.futures.Future[int]) -> None:
            try:
                finish_request(done)
            finally:
                release_sender()

        future.add_done_callback(request_finished)
        return True

    def _write_terminal(
        self,
        task: PendingRequest,
        outcome: str,
        worker: Worker | None,
        actual_output_tokens: int | None = None,
        error: str | None = None,
    ) -> None:
        finish_time = self.logical_monotonic()
        estimated_latency = estimate_latency_seconds(
            task.source_request_type,
            task.sample.reference_output_tokens,
        )
        slo = estimated_latency * SLO_MULTIPLIER
        queue_delay = None
        if task.first_dispatch_time_mono is not None:
            queue_delay = task.first_dispatch_time_mono - task.arrival_time_mono
        self._request_log.write(
            {
                "record_type": "request",
                "experiment_id": self.experiment_id,
                "scenario": self.scenario,
                "task_id": task.task_id,
                "outcome": outcome,
                "error": error,
                "ground_truth_request_type": task.ground_truth_request_type,
                "classified_request_type": task.classified_request_type,
                "misclassified": (
                    None
                    if task.classified_request_type is None
                    else task.classified_request_type != task.ground_truth_request_type
                ),
                "phase": task.phase,
                "arrival_time_utc": task.arrival_time_utc,
                "finish_time_utc": utc_now(),
                "arrival_time_mono": task.arrival_time_mono,
                "finish_time_mono": finish_time,
                "arrival_elapsed_s": task.arrival_elapsed_s,
                "queue_delay_s": queue_delay,
                "e2e_latency_s": finish_time - task.arrival_time_mono,
                "estimated_latency_s": estimated_latency,
                "slo_s": slo,
                "slo_met": outcome == "succeeded"
                and finish_time - task.arrival_time_mono <= slo,
                "attempt_count": task.attempt_count,
                "retry_errors": task.retry_errors or [],
                "prompt_tokens": task.sample.prompt_tokens,
                "reference_output_tokens": task.sample.reference_output_tokens,
                "predicted_output_tokens": task.predicted_output_tokens,
                "prediction_latency_s": task.prediction_latency_s,
                "source_request_type": task.source_request_type,
                "max_tokens": task.sample.max_tokens,
                "actual_output_tokens": actual_output_tokens,
                "worker": asdict(worker) if worker is not None else None,
            }
        )
        with self._tasks_lock:
            task.state = "terminal"
            self._active_tasks.pop(task.task_id, None)

    def _change_stats(self, request_type: str, **changes: int) -> None:
        with self._stats_lock:
            for key, amount in changes.items():
                self._stats[request_type][key] += amount

    def _discover_model(self, worker: Worker) -> str:
        if MODEL_NAME:
            return MODEL_NAME
        with self._model_lock:
            cached = self._model_cache.get(worker.base_url)
        if cached:
            return cached

        request = urllib.request.Request(
            f"{worker.base_url}/v1/models",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        )
        with urllib.request.urlopen(
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            payload = json.load(response)
        models = payload.get("data", [])
        if not models or not isinstance(models[0].get("id"), str):
            raise RuntimeError("/v1/models returned no usable model id")
        model = models[0]["id"]
        with self._model_lock:
            self._model_cache[worker.base_url] = model
        LOGGER.info("discovered model %s on %s", model, worker.base_url)
        return model

    def _send_request(self, worker: Worker, sample: DatasetRequest) -> int:
        model = self._discover_model(worker)
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": sample.prompt}],
                "temperature": TEMPERATURE,
                "max_tokens": sample.max_tokens,
                "stream": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{worker.base_url}/v1/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS
            ) as response:
                response_payload = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(500).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        try:
            actual_output_tokens = response_payload["usage"]["completion_tokens"]
            if isinstance(actual_output_tokens, bool):
                raise TypeError
            actual_output_tokens = int(actual_output_tokens)
        except (KeyError, TypeError, ValueError):
            raise RuntimeError(
                "OpenAI response has no valid usage.completion_tokens"
            ) from None
        if actual_output_tokens < 0:
            raise RuntimeError("OpenAI response has a negative usage.completion_tokens")
        return actual_output_tokens

    def status(self) -> dict[str, Any]:
        # Keep the global lock order control -> stats. Boundary/end transitions
        # use the same order, which avoids a status request deadlocking with a
        # Monitor HTTP call followed by backlog bookkeeping.
        with self._control_lock:
            started_at = self._started_at
            elapsed = self._clock_elapsed_locked()
            generation_finished = self._end_boundary_reached
            monitor_status = self._monitor_status.copy()
            dispatch_status = self.dispatch_status()
            sender_limits = self._sender_limits()
            sender_status = [
                {
                    "configuration_id": key[0],
                    "ip": key[1],
                    "port": key[2],
                    "limit": limit,
                    "active": self._sender_active.get(key, 0),
                }
                for key, limit in sorted(sender_limits.items())
            ]
            with self._stats_lock:
                stats = {
                    request_type: values.copy()
                    for request_type, values in self._stats.items()
                }
        drained = all(
            values["pending"] == 0 and values["in_flight"] == 0
            for values in stats.values()
        )
        result: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "scenario": self.scenario,
            "active": started_at is not None
            and not generation_finished
            and not self._stop_event.is_set(),
            "generation_finished": generation_finished,
            "drained": generation_finished and drained,
            "elapsed_seconds": round(elapsed, 3),
            "remaining_seconds": round(
                max(self.experiment_duration_seconds - elapsed, 0.0), 3
            ),
            "experiment_duration_seconds": self.experiment_duration_seconds,
            "started": self._started_at is not None,
            "prediction": {
                "input_threshold": self.input_threshold,
                "output_threshold": self.output_threshold,
                "queued": self._prediction_queue.qsize(),
                "capacity": self.max_prediction_pending,
            },
            "segment_seconds": self.qps_segment_seconds,
            "request_log_file": str(self._request_log.path),
            "event_log_file": str(self._event_log.path),
            "switch_log_file": str(self._switch_log.path),
            "switch_at_seconds": self.switch_at_seconds,
            "monitors": monitor_status,
            "active_configuration_id": self.registry.active_configuration_id,
            "registered_configurations": self.registry.configurations(),
            "concurrency": {
                "sender_threads": self.request_sender_threads,
                "max_in_flight": self.max_in_flight_requests,
                "max_pending_per_type": self.max_pending_requests_per_type,
                "effective_sender_limit": self.request_sender_threads,
                "worker_senders": sender_status,
            },
            "dispatch": dispatch_status,
            "routing_queues": {
                kind: {
                    "pending_queue_size": self._pending_queues[kind].qsize(),
                    "recovery_queue_size": self._recovery_queues[kind].qsize(),
                }
                for kind in self.configs
            },
            "request_types": {},
        }
        for request_type in self.configs:
            segment, qps = self._profile_position(request_type)
            result["request_types"][request_type] = {
                "segment": segment + 1,
                "current_qps": qps,
                "qps_profile": self.configs[request_type]["qps"],
                "dataset_size": len(self.datasets[request_type]),
                "active_worker_count": self.registry.count_for_type(request_type),
                "all_configuration_worker_count": self.registry.count_for_type(
                    request_type, active_only=False
                ),
                **stats[request_type],
            }
        return result


traffic_generator: TrafficGenerator | None = None


class RouterHandler(BaseHTTPRequestHandler):
    server_version = "VLLMRegistryRouter/1.0"

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"error": message})

    def _read_json_object(self) -> dict[str, Any] | None:
        content_type = (
            self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        )
        if content_type != "application/json":
            self._error(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "Content-Type must be application/json",
            )
            return None

        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else -1
        except ValueError:
            length = -1
        if length < 0:
            self._error(
                HTTPStatus.LENGTH_REQUIRED, "a valid Content-Length header is required"
            )
            return None
        if length > MAX_REQUEST_BODY_BYTES:
            self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large"
            )
            return None

        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "request body must be valid JSON")
            return None
        if not isinstance(payload, dict):
            self._error(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
            return None
        return payload

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = urlsplit(self.path).path.rstrip("/") or "/"
        payload = self._read_json_object()
        if payload is None:
            return

        if path == "/traffic/start":
            if traffic_generator is None:
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "traffic generator is not initialized",
                )
                return
            try:
                started = traffic_generator.activate()
            except RuntimeError as exc:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            status = HTTPStatus.CREATED if started else HTTPStatus.CONFLICT
            self._send_json(
                status,
                {
                    "status": "started" if started else "already_started",
                    **traffic_generator.status(),
                },
            )
            return

        if path == "/control/resume":
            if traffic_generator is None:
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "traffic generator is not initialized",
                )
                return
            try:
                result = traffic_generator.resume_dispatch()
            except ValueError as exc:
                self._error(HTTPStatus.CONFLICT, str(exc))
                return
            except RuntimeError as exc:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            self._send_json(HTTPStatus.OK, result)
            return

        if path == "/control/cold/resource-ready":
            if traffic_generator is None:
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "traffic generator is not initialized",
                )
                return
            try:
                result = traffic_generator.cold_resource_ready()
            except ValueError as exc:
                self._error(HTTPStatus.CONFLICT, str(exc))
                return
            except RuntimeError as exc:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            self._send_json(HTTPStatus.OK, result)
            return

        if path == "/control/cold/worker-ready":
            if traffic_generator is None:
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "traffic generator is not initialized",
                )
                return
            worker_id = str(payload.get("worker_id", "")).strip()
            if not worker_id:
                self._error(
                    HTTPStatus.BAD_REQUEST, "worker_id must be a non-empty string"
                )
                return
            try:
                result = traffic_generator.cold_worker_ready(worker_id)
            except ValueError as exc:
                self._error(HTTPStatus.CONFLICT, str(exc))
                return
            except RuntimeError as exc:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            self._send_json(HTTPStatus.OK, result)
            return

        if path == "/configurations/activate":
            if traffic_generator is None:
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "traffic generator is not initialized",
                )
                return
            configuration_id = str(payload.get("configuration_id", "")).strip()
            if not configuration_id:
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "configuration_id must be a non-empty string",
                )
                return
            try:
                result = traffic_generator.activate_configuration(configuration_id)
            except ValueError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json(HTTPStatus.OK, result)
            return

        if path == "/phases/register":
            raw_phase = payload.get("phase")
            try:
                phase = normalize_phase(raw_phase)
            except ValueError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            raw_worker_ids = payload.get("worker_ids")
            selected_worker_ids = None
            if raw_worker_ids is not None:
                if (
                    not isinstance(raw_worker_ids, list)
                    or not raw_worker_ids
                    or not all(
                        isinstance(worker_id, str) and worker_id.strip()
                        for worker_id in raw_worker_ids
                    )
                ):
                    self._error(
                        HTTPStatus.BAD_REQUEST,
                        "worker_ids must be a non-empty array of strings",
                    )
                    return
                selected_worker_ids = {
                    worker_id.strip() for worker_id in raw_worker_ids
                }
            try:
                result = register_phase_workers(phase, selected_worker_ids)
            except ValueError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            if traffic_generator is not None:
                traffic_generator._switch_log.write(
                    {
                        "record_type": "switch_event",
                        "event": "phase_workers_registered",
                        "timestamp": utc_now(),
                        "experiment_id": traffic_generator.experiment_id,
                        "experiment_elapsed_s": traffic_generator.elapsed_seconds(),
                        **result,
                    }
                )
            self._send_json(HTTPStatus.OK, result)
            return

        if path != "/add_worker":
            self._error(HTTPStatus.NOT_FOUND, "endpoint not found")
            return

        missing = [
            field
            for field in ("ip", "port", "request_type", "weight")
            if field not in payload
        ]
        if missing:
            self._error(
                HTTPStatus.BAD_REQUEST,
                f"missing required field(s): {', '.join(missing)}",
            )
            return

        try:
            ip = normalize_host(payload["ip"])
            port = normalize_port(payload["port"])
            request_type = normalize_request_type(payload["request_type"])
            weight = normalize_weight(payload["weight"])
            raw_configuration_id = payload.get("configuration_id", "default")
            if (
                not isinstance(raw_configuration_id, str)
                or not raw_configuration_id.strip()
            ):
                raise ValueError("configuration_id must be a non-empty string")
            configuration_id = raw_configuration_id.strip()
            raw_tp_size = payload.get("tp_size")
            if raw_tp_size is not None and (
                isinstance(raw_tp_size, bool)
                or not isinstance(raw_tp_size, int)
                or raw_tp_size <= 0
            ):
                raise ValueError("tp_size must be a positive integer or null")
            tp_size = raw_tp_size
        except (TypeError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if request_type not in REQUEST_CONFIGS:
            allowed = ", ".join(REQUEST_CONFIGS)
            self._error(
                HTTPStatus.BAD_REQUEST,
                f"unsupported request_type; allowed values: {allowed}",
            )
            return

        worker, action = worker_registry.add(
            ip,
            port,
            request_type,
            weight,
            configuration_id=configuration_id,
            tp_size=tp_size,
        )
        status = HTTPStatus.CREATED if action == "registered" else HTTPStatus.OK
        self._send_json(status, {"status": action, "worker": asdict(worker)})

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlsplit(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "worker_count": worker_registry.count(),
                    "active_configuration_id": worker_registry.active_configuration_id,
                },
            )
            return
        if path == "/workers":
            query = parse_qs(parsed.query)
            request_type = query.get("request_type", [None])[0]
            configuration_id = query.get("configuration_id", [None])[0]
            workers = [
                asdict(worker)
                for worker in worker_registry.list(request_type, configuration_id)
            ]
            self._send_json(HTTPStatus.OK, {"count": len(workers), "workers": workers})
            return
        if path == "/traffic":
            if traffic_generator is None:
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "traffic generator is not initialized",
                )
                return
            self._send_json(HTTPStatus.OK, traffic_generator.status())
            return
        if path == "/phase-configs":
            query = parse_qs(parsed.query)
            raw_phase = query.get("phase", [None])[0]
            try:
                phase = normalize_phase(raw_phase)
                validate_distinct_phase_configuration_ids()
                config = load_phase_config(phase)
            except ValueError as exc:
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    str(exc) or "phase query parameter must be 1 or 2",
                )
                return
            self._send_json(HTTPStatus.OK, config)
            return
        self._error(HTTPStatus.NOT_FOUND, "endpoint not found")

    def do_DELETE(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlsplit(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path != "/workers":
            self._error(HTTPStatus.NOT_FOUND, "endpoint not found")
            return

        query = parse_qs(parsed.query)
        try:
            ip = normalize_host(query.get("ip", [None])[0])
            port = normalize_port(query.get("port", [None])[0])
            configuration_id = str(
                query.get("configuration_id", ["default"])[0]
            ).strip()
            if not configuration_id:
                raise ValueError("configuration_id must be a non-empty string")
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        worker = worker_registry.remove(ip, port, configuration_id)
        if worker is None:
            self._error(HTTPStatus.NOT_FOUND, "worker not found")
            return
        self._send_json(HTTPStatus.OK, {"status": "removed", "worker": asdict(worker)})

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cold TP switching with pluggable online length prediction (no LLF)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--experiment-id", default="cold-run")
    parser.add_argument("--duration", type=float, default=3600)
    parser.add_argument("--segment-seconds", type=float, default=300)
    parser.add_argument("--switch-at-seconds", type=float, help="default: duration / 2")
    parser.add_argument("--qps-config", type=Path, required=True)
    parser.add_argument("--phase1-config", type=Path, required=True)
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--dataset-directory", type=Path, required=True)
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument(
        "--input-threshold",
        type=int,
        required=True,
        help="input_token_count >= threshold is long",
    )
    parser.add_argument(
        "--output-threshold",
        type=int,
        default=444,
        help="predicted tokens >= threshold is long",
    )
    parser.add_argument(
        "--predictor-factory",
        default="predictors:bert_predictor",
        help="importable module:factory; returns an object with predict(prompt_text)",
    )
    parser.add_argument(
        "--predictor-kwargs",
        default="{}",
        help="JSON object passed to the predictor factory",
    )
    parser.add_argument("--sender-threads", type=int, default=256)
    parser.add_argument("--max-pending-per-type", type=int, default=20000)
    parser.add_argument("--max-prediction-pending", type=int, default=20000)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument(
        "--shutdown-after-complete",
        type=float,
        default=0,
        help="seconds to keep API alive after cutoff; 0 keeps it alive until Ctrl+C",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    for name in ("duration", "segment_seconds", "request_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    for name in (
        "input_threshold",
        "output_threshold",
        "sender_threads",
        "max_pending_per_type",
        "max_prediction_pending",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if (
        args.switch_at_seconds is not None
        and not 0 < args.switch_at_seconds < args.duration
    ):
        parser.error("switch-at-seconds must be between 0 and duration")
    if (
        not math.isfinite(args.shutdown_after_complete)
        or args.shutdown_after_complete < 0
    ):
        parser.error("shutdown-after-complete must be finite and non-negative")
    if not args.experiment_id.strip():
        parser.error("experiment-id cannot be empty")
    return args


def main() -> None:
    global traffic_generator, worker_registry, phase_config_paths, REQUEST_TIMEOUT_SECONDS
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    REQUEST_TIMEOUT_SECONDS = args.request_timeout
    phase_config_paths = {
        1: args.phase1_config.resolve(),
        2: args.phase2_config.resolve(),
    }
    validate_distinct_phase_configuration_ids()
    for phase in (1, 2):
        phase_config = load_phase_config(phase)
        missing = set(DATASET_FILENAMES) - {
            w["request_type"] for w in phase_config["workers"]
        }
        if missing:
            raise ValueError(
                f"phase {phase} must cover all four predicted types; missing {sorted(missing)}"
            )
    configs = build_request_configs(args.dataset_directory.resolve(), args.qps_config)
    validate_request_configs(configs, args.duration, args.segment_seconds)
    # Fail before opening logs or starting logical time if weights/config are missing.
    predictor = load_predictor(args.predictor_factory, args.predictor_kwargs)
    worker_registry = WorkerRegistry(load_phase_config(1)["configuration_id"])
    traffic_generator = TrafficGenerator(
        worker_registry,
        configs=configs,
        predictor=predictor,
        input_threshold=args.input_threshold,
        output_threshold=args.output_threshold,
        max_prediction_pending=args.max_prediction_pending,
        completion_log_path=args.request_log.resolve(),
        request_sender_threads=args.sender_threads,
        max_in_flight_requests=args.sender_threads,
        max_pending_requests_per_type=args.max_pending_per_type,
        experiment_duration_seconds=args.duration,
        qps_segment_seconds=args.segment_seconds,
        experiment_id=args.experiment_id.strip(),
        switch_at_seconds=args.switch_at_seconds,
    )
    server = ThreadingHTTPServer((args.host, args.port), RouterHandler)
    server_thread = threading.Thread(
        target=server.serve_forever, name="router-http", daemon=True
    )

    def stop_on_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_sigterm)
    traffic_generator.start()
    server_thread.start()
    LOGGER.info(
        "cold router listening on http://%s:%d; waiting for Controller",
        args.host,
        args.port,
    )
    completed_at = None
    try:
        while server_thread.is_alive():
            server_thread.join(timeout=0.5)
            if traffic_generator.status()["drained"] and args.shutdown_after_complete:
                completed_at = completed_at or time.monotonic()
                if time.monotonic() - completed_at >= args.shutdown_after_complete:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server_thread.join(timeout=5)
        server.server_close()
        traffic_generator.stop()


if __name__ == "__main__":
    main()
