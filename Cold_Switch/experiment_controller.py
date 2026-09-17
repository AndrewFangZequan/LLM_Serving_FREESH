#!/usr/bin/env python3
"""Orchestrate cold switching; Router owns the logical clock and phase files."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def request_json(method, url, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"{method} {url}: HTTP {exc.code}: {exc.read(1000).decode(errors='replace')}"
        ) from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{method} {url}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"{url} did not return a JSON object")
    return result


def run_commands(commands, label, timeout):
    for command in commands:
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(x, str) and x for x in command)
        ):
            raise ValueError(f"{label}: commands must be non-empty arrays of strings")
        print(f"[{label}] {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, timeout=timeout)


def wait_for_url(url, timeout, interval):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            request_json(
                "GET", url, timeout=min(10, max(0.1, deadline - time.monotonic()))
            )
            return
        except RuntimeError as exc:
            last_error = exc
        time.sleep(min(interval, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"Not healthy within {timeout}s: {url}; {last_error}")


def parallel(function, items):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(items))) as pool:
        futures = [pool.submit(function, item) for item in items]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def wait_for_manual_ready(ready_file):
    if ready_file:
        path = Path(ready_file).expanduser()
        print(
            f"[paused] Prepare GPUs/SSH and phase-2 Monitors, then create {path}",
            flush=True,
        )
        while not path.exists():
            time.sleep(1)
    else:
        if not sys.stdin.isatty():
            raise RuntimeError("Use an interactive terminal or configure ready_file")
        input(
            "Prepare GPUs/SSH and all phase-2 Monitors. Do not start NEW vLLM workers yet.\n"
            "Update phase2.json if needed. Retained workers can stay running.\nPress Enter to continue: "
        )


def status_line(status):
    pending = sum(x["pending"] for x in status["request_types"].values())
    inflight = sum(x["in_flight"] for x in status["request_types"].values())
    print(
        f"[status] elapsed={status['elapsed_seconds']:.1f}s "
        f"state={status['dispatch']['experiment_state']} pending={pending} in_flight={inflight}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate controller settings only; no network or commands",
    )
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        if not isinstance(config, dict):
            raise ValueError("controller config must be an object")
        settings = {}
        for name, default in (
            ("health_timeout_seconds", 900),
            ("health_poll_interval_seconds", 2),
            ("status_poll_interval_seconds", 5),
            ("command_timeout_seconds", 60),
        ):
            value = float(config.get(name, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            settings[name] = value
        if args.dry_run:
            print(
                "Cold: prepare phase 1 -> start traffic -> await automatic boundary -> manual ready"
            )
            print(
                "-> record startup/resume clock -> start/register workers independently -> hard cutoff -> cleanup"
            )
            print(
                "Phase JSON, model weights and endpoints are validated during the real run."
            )
            return 0
        ready_file = config.get("ready_file")
        if ready_file and Path(ready_file).expanduser().exists():
            raise ValueError("ready_file already exists; use a fresh path for this run")
        base = str(config.get("router_url", "http://127.0.0.1:8010")).rstrip("/")
        health_timeout = settings["health_timeout_seconds"]
        health_interval = settings["health_poll_interval_seconds"]
        status_interval = settings["status_poll_interval_seconds"]
        command_timeout = settings["command_timeout_seconds"]

        def api(method, path, payload=None):
            return request_json(method, base + path, payload)

        status = api("GET", "/traffic")
        if status["started"]:
            raise RuntimeError("Router has already started; use a fresh run")
        phase1 = api("GET", "/phase-configs?phase=1")
        run_commands(phase1["start_commands"], "start phase 1", command_timeout)

        def prepare(worker):
            if worker.get("start_command"):
                run_commands(
                    [worker["start_command"]], worker["worker_id"], command_timeout
                )
            wait_for_url(worker["health_url"], health_timeout, health_interval)

        parallel(prepare, phase1["workers"])
        api("POST", "/phases/register", {"phase": 1})
        api(
            "POST",
            "/configurations/activate",
            {"configuration_id": phase1["configuration_id"]},
        )
        api("POST", "/traffic/start", {})
        while True:
            status = api("GET", "/traffic")
            status_line(status)
            if status["dispatch"]["experiment_state"] == "cold_waiting_for_gpu":
                break
            if status["drained"]:
                raise RuntimeError("Experiment ended before resource preparation")
            time.sleep(status_interval)
        # Do not proceed to reuse GPUs if any phase-1 monitor failed to stop.
        failures = [
            r
            for r in status["monitors"].get("1", {}).get("results", [])
            if r["status"] != "ok"
        ]
        if failures:
            raise RuntimeError(f"Phase-1 Monitor stop failed: {failures}")
        run_commands(phase1["stop_commands"], "stop phase 1", command_timeout)
        wait_for_manual_ready(ready_file)
        phase2 = api("GET", "/phase-configs?phase=2")
        api("POST", "/control/cold/resource-ready", {})
        # Stage-level commands must return promptly; per-worker launch commands
        # are preferred so one slow launch does not gate another worker.
        parallel(
            lambda command: run_commands([command], "start phase 2", command_timeout),
            phase2["start_commands"],
        )
        activation_lock = threading.Lock()
        activated = False

        def start_and_register(worker):
            nonlocal activated
            prepare(worker)
            api(
                "POST", "/control/cold/worker-ready", {"worker_id": worker["worker_id"]}
            )
            api(
                "POST",
                "/phases/register",
                {"phase": 2, "worker_ids": [worker["worker_id"]]},
            )
            with activation_lock:
                if not activated:
                    api(
                        "POST",
                        "/configurations/activate",
                        {"configuration_id": phase2["configuration_id"]},
                    )
                    api("POST", "/control/resume", {})
                    activated = True
            print(f"[available] {worker['worker_id']}", flush=True)

        parallel(start_and_register, phase2["workers"])
        while True:
            status = api("GET", "/traffic")
            status_line(status)
            if status["drained"]:
                run_commands(phase2["stop_commands"], "stop phase 2", command_timeout)
                print(
                    "[complete] Hard cutoff reached; collect JSONL and Monitor/MIAD CSVs."
                )
                return 0
            time.sleep(status_interval)
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
