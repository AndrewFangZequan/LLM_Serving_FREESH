# -*- coding: utf-8 -*-
import time
import json
import requests
import subprocess
import sys
import threading
import pandas as pd
import argparse
import signal
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, List, Dict, Optional
from datetime import datetime
from collections import deque

class VLLMMonitor:
    """
    Monitor the performance and power consumption of a vLLM service.
    (Combines moving-window averages with a broader set of performance metrics.)
    """
    def __init__(self,
                 worker_id: str = "worker",
                 vllm_host: str = "localhost",
                 vllm_port: int = 8000,
                 gpu_ids: Optional[List[int]] = None,
                 interval_seconds: float = 1.0, # A one-second interval is recommended for more stable metrics.
                 output_csv_file: Optional[str] = None,
                 output_directory: Optional[str] = "monitor_outputs",
                 window_size: int = 5):
        """
        Initialize the monitor.
        """
        self.worker_id = worker_id
        self.vllm_host = vllm_host
        self.vllm_port = vllm_port
        self.metrics_url = f"http://{vllm_host}:{vllm_port}/metrics"
        self.interval_seconds = interval_seconds
        self.output_csv_file = output_csv_file
        self.output_directory = None if output_directory is None else Path(output_directory)
        self.history = deque(maxlen=window_size)

        if gpu_ids is None:
            self.gpu_ids = self._detect_gpu_ids()
        else:
            self.gpu_ids = gpu_ids

        self._stop_event = threading.Event()
        self._monitor_thread = None
        self.monitoring_data = []
        self.csv_columns = [
            "timestamp", "session_id", "experiment_id", "scenario", "phase",
            "recording_window",
            "experiment_elapsed_s", "vllm_available", "total_power_w", "gpu_freq_mhz",
            "throughput_tps", "avg_e2e_latency_ms", "avg_ttft_ms",
            "avg_tpot_ms", "success_rate"
        ]
        self._state_lock = threading.RLock()
        self._recording_state = "idle"
        self._session: Dict[str, Any] = {}
        self._recording_started_mono: Optional[float] = None
        self._rows_written = 0
        self._last_sample_utc: Optional[str] = None
        self._last_vllm_available: Optional[bool] = None
        self._last_metrics_error_log_mono = 0.0

    @staticmethod
    def _safe_filename_component(value: Any) -> str:
        component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
        return component or "unknown"

    def _new_output_path(self, session: Dict[str, Any]) -> Optional[str]:
        if self.output_directory is None:
            return self.output_csv_file
        self.output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        experiment = self._safe_filename_component(session["experiment_id"])
        scenario = self._safe_filename_component(session["scenario"])
        phase = self._safe_filename_component(session["phase"])
        recording_window = self._safe_filename_component(
            session.get("recording_window", f"phase{phase}")
        )
        worker = self._safe_filename_component(self.worker_id)
        filename = (
            f"{timestamp}_{experiment}_{scenario}_{recording_window}_{worker}.csv"
        )
        return str(self.output_directory / filename)

    def _detect_gpu_ids(self) -> List[int]:
        # ... (No changes are needed in this method.) ...
        try:
            cmd = "nvidia-smi --query-gpu=index --format=csv,noheader,nounits"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            ids = [int(i) for i in result.stdout.strip().split('\n')]
            print(f"[INFO] Automatically detected {len(ids)} GPU(s): {ids}")
            return ids
        except Exception as e:
            print(f"\n[WARNING] GPU auto-detection failed: {e}. GPU metrics will not be monitored.", file=sys.stderr)
            return []


    def _get_gpu_stats(self) -> tuple[Optional[float], Optional[int]]:
        # ... (No changes are needed in this method.) ...
        if not self.gpu_ids:
            return None, None

        try:
            gpu_id_str = ",".join(map(str, self.gpu_ids))
            power_cmd = f"nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i {gpu_id_str}"
            power_result = subprocess.run(power_cmd, shell=True, capture_output=True, text=True, check=True)
            power_draws = [float(p) for p in power_result.stdout.strip().split('\n')]
            total_power = sum(power_draws)

            freq_cmd = f"nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits -i {self.gpu_ids[0]}"
            freq_result = subprocess.run(freq_cmd, shell=True, capture_output=True, text=True, check=True)
            frequency = int(freq_result.stdout.strip().split(' ')[0])

            return total_power, frequency
        except Exception as e:
            print(f"\n[ERROR] Failed to run nvidia-smi: {e}", file=sys.stderr)
            return None, None

    def _get_vllm_metrics(self) -> Optional[Dict[str, float]]:
        """Fetch and parse SLO-related metrics from vLLM's /metrics endpoint."""
        # <-- Updated: add the metrics to parse.
        keys_to_parse = {
            "vllm:generation_tokens_total",
            "vllm:e2e_request_latency_seconds_sum",
            "vllm:e2e_request_latency_seconds_count",
            "vllm:time_to_first_token_seconds_sum",
            "vllm:time_to_first_token_seconds_count",
            # New metrics
            "vllm:time_per_output_token_seconds_sum",
            "vllm:time_per_output_token_seconds_count",
            "vllm:request_success_total",
        }
        try:
            response = requests.get(self.metrics_url, timeout=(2, 5))
            response.raise_for_status()
            metrics = {key: 0.0 for key in keys_to_parse}
            for line in response.text.splitlines():
                if line.startswith('#'): continue
                for key in keys_to_parse:
                    # Use startswith to support metrics that include labels.
                    if line.startswith(key):
                        try:
                            value = float(line.rsplit(' ', 1)[1])
                            metrics[key] += value
                        except (ValueError, IndexError):
                            continue
            return metrics
        except requests.exceptions.RequestException as e:
            now = time.monotonic()
            if now - self._last_metrics_error_log_mono >= 30:
                print(
                    f"\n[WARNING] vLLM metrics unavailable ({self.metrics_url}): {e}",
                    file=sys.stderr,
                )
                self._last_metrics_error_log_mono = now
            return None

    def _monitor_loop(self):
        """Run the main monitoring loop and moving-window average calculations."""
        while not self._stop_event.is_set():
            current_time = time.monotonic()
            total_power, frequency = self._get_gpu_stats()
            current_metrics = self._get_vllm_metrics()
            vllm_available = current_metrics is not None
            if current_metrics is not None:
                self.history.append({'time': current_time, 'metrics': current_metrics})

            throughput_tps, avg_e2e_ms, avg_ttft_ms = 0.0, 0.0, 0.0
            avg_tpot_ms, success_rate = 0.0, 1.0 # <-- Added: initialize the new metrics.

            if current_metrics is not None and len(self.history) > 1:
                oldest_sample = self.history[0]
                latest_sample = self.history[-1]
                window_interval = latest_sample['time'] - oldest_sample['time']

                if window_interval > 0:
                    # --- Calculate throughput (unchanged) ---
                    delta_tokens = latest_sample['metrics']["vllm:generation_tokens_total"] - oldest_sample['metrics']["vllm:generation_tokens_total"]
                    throughput_tps = delta_tokens / window_interval

                    # --- Calculate latency and success rate ---
                    # Note: all latency and success-rate calculations use e2e_count,
                    # which is the number of completed requests.
                    delta_e2e_count = latest_sample['metrics']["vllm:e2e_request_latency_seconds_count"] - oldest_sample['metrics']["vllm:e2e_request_latency_seconds_count"]

                    if delta_e2e_count > 0:
                        # E2E latency
                        delta_e2e_sum = latest_sample['metrics']["vllm:e2e_request_latency_seconds_sum"] - oldest_sample['metrics']["vllm:e2e_request_latency_seconds_sum"]
                        avg_e2e_ms = (delta_e2e_sum / delta_e2e_count) * 1000

                        # Success rate
                        delta_success = latest_sample['metrics']["vllm:request_success_total"] - oldest_sample['metrics']["vllm:request_success_total"]
                        success_rate = delta_success / delta_e2e_count

                    # --- Calculate TTFT ---
                    delta_ttft_count = latest_sample['metrics']["vllm:time_to_first_token_seconds_count"] - oldest_sample['metrics']["vllm:time_to_first_token_seconds_count"]
                    if delta_ttft_count > 0:
                        delta_ttft_sum = latest_sample['metrics']["vllm:time_to_first_token_seconds_sum"] - oldest_sample['metrics']["vllm:time_to_first_token_seconds_sum"]
                        avg_ttft_ms = (delta_ttft_sum / delta_ttft_count) * 1000

                    # <-- Added: calculate TPOT (Time Per Output Token). -->
                    delta_tpot_count = latest_sample['metrics']["vllm:time_per_output_token_seconds_count"] - oldest_sample['metrics']["vllm:time_per_output_token_seconds_count"]
                    if delta_tpot_count > 0:
                        delta_tpot_sum = latest_sample['metrics']["vllm:time_per_output_token_seconds_sum"] - oldest_sample['metrics']["vllm:time_per_output_token_seconds_sum"]
                        avg_tpot_ms = (delta_tpot_sum / delta_tpot_count) * 1000

            # <-- Updated: include the new metrics in the log and output. -->
            log_time = datetime.now().isoformat()
            with self._state_lock:
                session = self._session.copy()
                recording_started = self._recording_started_mono
            recording_elapsed = (
                0.0 if recording_started is None else current_time - recording_started
            )
            logical_start = float(session.get("logical_start_seconds", 0.0))
            current_data = {
                "timestamp": log_time,
                "session_id": session.get("session_id"),
                "experiment_id": session.get("experiment_id"),
                "scenario": session.get("scenario"),
                "phase": session.get("phase"),
                "recording_window": session.get("recording_window"),
                "experiment_elapsed_s": logical_start + recording_elapsed,
                "vllm_available": vllm_available,
                "total_power_w": total_power,
                "gpu_freq_mhz": frequency,
                "throughput_tps": throughput_tps if vllm_available else None,
                "avg_e2e_latency_ms": avg_e2e_ms if vllm_available else None,
                "avg_ttft_ms": avg_ttft_ms if vllm_available else None,
                "avg_tpot_ms": avg_tpot_ms if vllm_available else None,
                "success_rate": success_rate if vllm_available else None,
            }
            self.monitoring_data.append(current_data)

            if self.output_csv_file:
                pd.DataFrame([current_data]).to_csv(self.output_csv_file, mode='a', header=False, index=False, encoding='utf-8')
            with self._state_lock:
                self._rows_written += 1
                self._last_sample_utc = log_time
                self._last_vllm_available = vllm_available

            # Update the real-time output.
            power_str = f"{total_power:.1f}W" if total_power is not None else "N/A"
            freq_str = f"{frequency}MHz" if frequency is not None else "N/A"
            tps_str = f"{throughput_tps:.1f} tok/s" if vllm_available else "N/A"
            e2e_str = f"{avg_e2e_ms:.1f} ms" if vllm_available else "N/A"
            ttft_str = f"{avg_ttft_ms:.1f} ms" if vllm_available else "N/A"
            tpot_str = f"{avg_tpot_ms:.2f} ms/tok" if vllm_available else "N/A"
            succ_str = f"{success_rate:.1%}" if vllm_available else "N/A"

            output = (
                f"\r[{time.strftime('%H:%M:%S')}] "
                f"Pwr:{power_str:<8} | "
                f"Freq:{freq_str:<10} | "
                f"TPS:{tps_str:<12} | "
                f"E2E:{e2e_str:<10} | "
                f"TTFT:{ttft_str:<10} | "
                f"TPOT:{tpot_str:<12} | "
                f"Succ:{succ_str:<8}"
            )
            sys.stdout.write(output)
            sys.stdout.flush()

            self._stop_event.wait(self.interval_seconds)

        with self._state_lock:
            if self._recording_state == "recording":
                self._recording_state = "stopped"

    @staticmethod
    def _normalize_session(payload: Dict[str, Any]) -> Dict[str, Any]:
        required_strings = ("session_id", "experiment_id", "scenario")
        result: Dict[str, Any] = {}
        for field in required_strings:
            value = payload.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
            result[field] = value.strip()
        phase = payload.get("phase")
        if isinstance(phase, bool) or not isinstance(phase, int) or phase not in (1, 2):
            raise ValueError("phase must be 1 or 2")
        result["phase"] = phase
        recording_window = payload.get("recording_window", f"phase{phase}")
        if not isinstance(recording_window, str) or not recording_window.strip():
            raise ValueError("recording_window must be a non-empty string")
        result["recording_window"] = recording_window.strip()
        for field in ("logical_start_seconds", "expected_duration_seconds"):
            value = payload.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{field} must be a non-negative number")
            result[field] = float(value)
        return result

    def identity(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "worker_id": self.worker_id,
            "gpu_ids": self.gpu_ids,
            "vllm_host": self.vllm_host,
            "vllm_port": self.vllm_port,
            "metrics_url": self.metrics_url,
        }

    def status(self) -> Dict[str, Any]:
        with self._state_lock:
            return {
                **self.identity(),
                "recording_state": self._recording_state,
                "recording": self._recording_state == "recording",
                "session": self._session.copy() or None,
                "output_csv_file": self.output_csv_file,
                "rows_written": self._rows_written,
                "last_sample_utc": self._last_sample_utc,
                "last_vllm_available": self._last_vllm_available,
            }

    def start_recording(self, payload: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        session = self._normalize_session(payload)
        with self._state_lock:
            if self._recording_state == "recording":
                if self._session.get("session_id") == session["session_id"]:
                    return False, self.status()
                raise RuntimeError(
                    f"monitor is already recording session {self._session.get('session_id')}"
                )
            if self._session.get("session_id") == session["session_id"]:
                raise RuntimeError(f"session {session['session_id']} has already been stopped")

            output_path = self._new_output_path(session)
            if output_path:
                output_parent = Path(output_path).parent
                output_parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(columns=self.csv_columns).to_csv(
                    output_path, index=False, encoding="utf-8"
                )
            self.output_csv_file = output_path
            self._session = session
            self._recording_state = "recording"
            self._recording_started_mono = time.monotonic()
            self._rows_written = 0
            self._last_sample_utc = None
            self._last_vllm_available = None
            self.monitoring_data.clear()
            self.history.clear()
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name=f"monitor-{self.worker_id}",
                daemon=True,
            )
            self._monitor_thread.start()

        print("--- vLLM recording started ---")
        print(f"Session: {session['session_id']}")
        print(f"Metrics endpoint: {self.metrics_url}")
        print(f"Monitored GPUs: {self.gpu_ids}")
        if self.output_csv_file:
            print(f"CSV: {self.output_csv_file}")
        return True, self.status()

    def stop_recording(self, payload: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        session_id = payload.get("session_id")
        if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
            raise ValueError("session_id must be a non-empty string when provided")
        with self._state_lock:
            active_session_id = self._session.get("session_id")
            if session_id is not None and active_session_id not in (None, session_id.strip()):
                raise RuntimeError(
                    f"session mismatch: active session is {active_session_id}"
                )
            thread = self._monitor_thread
            if self._recording_state != "recording" or thread is None or not thread.is_alive():
                return False, self.status()
            self._stop_event.set()
        thread.join(timeout=max(self.interval_seconds + 8.0, 10.0))
        if thread.is_alive():
            raise RuntimeError("monitor thread did not stop in time")
        with self._state_lock:
            self._recording_state = "stopped"
            self._recording_started_mono = None
        print("\n--- vLLM recording stopped ---")
        return True, self.status()

    # Kept as small compatibility wrappers for code that imported the old class.
    def start(self) -> None:
        self.start_recording(
            {
                "session_id": f"manual-{int(time.time())}",
                "experiment_id": "manual",
                "scenario": "manual",
                "phase": 1,
                "logical_start_seconds": 0,
                "expected_duration_seconds": 0,
            }
        )

    def stop(self) -> None:
        self.stop_recording({})


def make_handler(monitor: VLLMMonitor) -> type[BaseHTTPRequestHandler]:
    class MonitorHandler(BaseHTTPRequestHandler):
        server_version = "VLLMMonitor/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[HTTP] {self.address_string()} {format % args}")

        def _send(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_payload(self) -> Optional[Dict[str, Any]]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid Content-Length"})
                return None
            if length < 0 or length > 64 * 1024:
                self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
                return None
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send(HTTPStatus.BAD_REQUEST, {"error": "body must be valid JSON"})
                return None
            if not isinstance(payload, dict):
                self._send(HTTPStatus.BAD_REQUEST, {"error": "body must be a JSON object"})
                return None
            return payload

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/health":
                self._send(HTTPStatus.OK, monitor.status())
                return
            if self.path.rstrip("/") == "/recording/status":
                self._send(HTTPStatus.OK, monitor.status())
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})

        def do_POST(self) -> None:  # noqa: N802
            payload = self._read_payload()
            if payload is None:
                return
            try:
                if self.path.rstrip("/") == "/recording/start":
                    started, status = monitor.start_recording(payload)
                    self._send(HTTPStatus.CREATED if started else HTTPStatus.OK, status)
                    return
                if self.path.rstrip("/") == "/recording/stop":
                    _, status = monitor.stop_recording(payload)
                    self._send(HTTPStatus.OK, status)
                    return
            except ValueError as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except RuntimeError as exc:
                self._send(HTTPStatus.CONFLICT, {"error": str(exc)})
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})

    return MonitorHandler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HTTP-controlled GPU and vLLM monitor service."
    )
    parser.add_argument("--worker-id", required=True, help="Worker ID used in phase JSON.")
    parser.add_argument("--host", default="localhost", help="vLLM host.")
    parser.add_argument("--port", type=int, default=8000, help="vLLM port.")
    parser.add_argument("--gpu-ids", required=True, help="Comma-separated GPU IDs.")
    parser.add_argument("--api-host", default="0.0.0.0", help="Monitor API bind host.")
    parser.add_argument("--api-port", type=int, default=9100, help="Monitor API bind port.")
    parser.add_argument("--output-dir", default="monitor_outputs", help="Directory for session CSV files.")
    parser.add_argument("--interval", type=float, default=0.3, help="Sampling interval in seconds.")
    parser.add_argument("--window-size", type=int, default=5, help="Moving-average sample window.")
    args = parser.parse_args()

    try:
        gpu_ids_list = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
        if not gpu_ids_list or any(gpu_id < 0 for gpu_id in gpu_ids_list):
            raise ValueError("GPU IDs must be non-negative integers")
        if not 1 <= args.api_port <= 65535 or not 1 <= args.port <= 65535:
            raise ValueError("ports must be from 1 to 65535")
        if args.interval <= 0:
            raise ValueError("interval must be greater than zero")
    except ValueError as exc:
        parser.error(str(exc))

    monitor = VLLMMonitor(
        worker_id=args.worker_id,
        vllm_host=args.host,
        vllm_port=args.port,
        gpu_ids=gpu_ids_list,
        interval_seconds=args.interval,
        output_directory=args.output_dir,
        window_size=args.window_size,
    )
    server = ThreadingHTTPServer((args.api_host, args.api_port), make_handler(monitor))
    shutdown_event = threading.Event()

    def request_shutdown(signum: int, frame: Any) -> None:
        del signum, frame
        shutdown_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    server.timeout = 0.5
    print(
        f"Monitor API for {args.worker_id} listening on "
        f"http://{args.api_host}:{args.api_port}"
    )
    try:
        while not shutdown_event.is_set():
            server.handle_request()
    finally:
        monitor.stop_recording({})
        server.server_close()
