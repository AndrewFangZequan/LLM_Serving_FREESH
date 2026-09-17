"""Real local HTTP + Controller/Router/Monitor integration, using fake vLLM/GPU data."""

import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import experiment_controller as controller
import monitor
import router
from test_prediction import Predictor


class HttpIntegrationTests(unittest.TestCase):
    def test_four_type_cold_lifecycle_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            servers, threads, monitors = [], [], []
            old_paths, old_registry, old_generator = (
                router.phase_config_paths,
                router.worker_registry,
                router.traffic_generator,
            )
            g = None
            try:
                phases = {}
                for phase in (1, 2):
                    workers = []
                    for i, kind in enumerate(router.DATASET_FILENAMES):
                        identity = f"phase{phase}-{i}"
                        ready = root / f"{identity}.ready"
                        if phase == 1:
                            ready.touch()
                        service = monitor.VLLMMonitor(
                            worker_id=identity,
                            gpu_ids=[0],
                            interval_seconds=0.02,
                            output_directory=str(root / identity),
                        )
                        service._get_gpu_stats = lambda: (100.0, 1200)
                        service._get_vllm_metrics = lambda: None
                        monitors.append(service)
                        base_handler = monitor.make_handler(service)

                        def handler_for(parent, ready_path):
                            class Handler(parent):
                                def do_GET(self):
                                    if self.path == "/v1/models":
                                        self.reply(
                                            {"data": [{"id": "fake-model"}]},
                                            200 if ready_path.exists() else 503,
                                        )
                                    else:
                                        super().do_GET()

                                def do_POST(self):
                                    if self.path == "/v1/chat/completions":
                                        payload = json.loads(
                                            self.rfile.read(
                                                int(self.headers["Content-Length"])
                                            )
                                        )
                                        self.reply(
                                            {
                                                "usage": {
                                                    "completion_tokens": payload[
                                                        "max_tokens"
                                                    ]
                                                }
                                            }
                                        )
                                    else:
                                        super().do_POST()

                                def reply(self, value, status=200):
                                    body = json.dumps(value).encode()
                                    self.send_response(status)
                                    self.send_header("Content-Type", "application/json")
                                    self.send_header("Content-Length", str(len(body)))
                                    self.end_headers()
                                    self.wfile.write(body)

                                def log_message(self, *args):
                                    pass

                            return Handler

                        try:
                            server = ThreadingHTTPServer(
                                ("127.0.0.1", 0), handler_for(base_handler, ready)
                            )
                        except PermissionError:
                            self.skipTest(
                                "local socket binding is not permitted in this environment"
                            )
                        servers.append(server)
                        t = threading.Thread(target=server.serve_forever, daemon=True)
                        t.start()
                        threads.append(t)
                        port = server.server_address[1]
                        worker = {
                            "worker_id": identity,
                            "ip": "127.0.0.1",
                            "port": port,
                            "request_type": kind,
                            "weight": 1,
                            "tp_size": 1,
                            "gpu_ids": [0],
                            "health_url": f"http://127.0.0.1:{port}/v1/models",
                            "monitor": {
                                "url": f"http://127.0.0.1:{port}",
                                "required": True,
                            },
                        }
                        if phase == 2:
                            worker["start_command"] = [
                                sys.executable,
                                "-c",
                                "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
                                str(ready),
                            ]
                        workers.append(worker)
                    phases[phase] = root / f"phase{phase}.json"
                    phases[phase].write_text(
                        json.dumps(
                            {
                                "configuration_id": f"phase{phase}",
                                "workers": workers,
                                "stop_commands": (
                                    [
                                        [
                                            sys.executable,
                                            "-c",
                                            "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
                                            str(root / "cleaned"),
                                        ]
                                    ]
                                    if phase == 2
                                    else []
                                ),
                            }
                        )
                    )
                configs = {}
                for kind in router.DATASET_FILENAMES:
                    path = root / (kind + ".jsonl")
                    path.write_text(
                        json.dumps(
                            {
                                "prompt": kind,
                                "input_token_count": (
                                    200 if kind.startswith("long") else 20
                                ),
                                "output_token_count": 10,
                            }
                        )
                        + "\n"
                    )
                    configs[kind] = {"qps": [10], "dataset": path}

                class FourWayPredictor:
                    def predict(self, prompt):
                        return 500 if prompt.endswith("long_response") else 10

                router.phase_config_paths = phases
                registry = router.WorkerRegistry("phase1")
                router.worker_registry = registry
                g = router.TrafficGenerator(
                    registry,
                    predictor=FourWayPredictor(),
                    input_threshold=100,
                    configs=configs,
                    completion_log_path=root / "requests.jsonl",
                    request_sender_threads=8,
                    experiment_duration_seconds=2,
                    qps_segment_seconds=2,
                    switch_at_seconds=0.5,
                )
                router.traffic_generator = g
                api = ThreadingHTTPServer(("127.0.0.1", 0), router.RouterHandler)
                servers.append(api)
                t = threading.Thread(target=api.serve_forever, daemon=True)
                t.start()
                threads.append(t)
                settings = root / "controller.json"
                settings.write_text(
                    json.dumps(
                        {
                            "router_url": f"http://127.0.0.1:{api.server_address[1]}",
                            "health_timeout_seconds": 3,
                            "health_poll_interval_seconds": 0.02,
                            "status_poll_interval_seconds": 0.03,
                            "command_timeout_seconds": 3,
                        }
                    )
                )
                g.start()
                with patch.object(
                    sys, "argv", ["controller", str(settings)]
                ), patch.object(
                    controller, "wait_for_manual_ready", lambda _: None
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    result = controller.main()
                self.assertEqual(result, 0)
                self.assertTrue((root / "cleaned").exists())
                self.assertTrue(g.status()["drained"])
                self.assertTrue(all(not m.status()["recording"] for m in monitors))
                records = [
                    json.loads(line)
                    for line in (root / "requests.jsonl").read_text().splitlines()
                ]
                self.assertEqual(
                    {r["classified_request_type"] for r in records},
                    set(router.DATASET_FILENAMES),
                )
                self.assertEqual(
                    {r["worker"]["configuration_id"] for r in records if r["worker"]},
                    {"phase1", "phase2"},
                )
                self.assertEqual(len(list(root.glob("phase*/*.csv"))), 12)
                events = [
                    json.loads(line)
                    for line in (root / "requests_switch.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(
                    sum(
                        e.get("event") == "cold_worker_monitor_transitioned"
                        for e in events
                    ),
                    4,
                )
            finally:
                if g is not None:
                    g.stop()
                for server in servers:
                    server.shutdown()
                    server.server_close()
                for t in threads:
                    t.join(2)
                for m in monitors:
                    m.stop()
                (
                    router.phase_config_paths,
                    router.worker_registry,
                    router.traffic_generator,
                ) = (old_paths, old_registry, old_generator)


if __name__ == "__main__":
    unittest.main()
