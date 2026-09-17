import json
import tempfile
import time
import unittest
from pathlib import Path

import monitor as monitor_module
import router
from test_prediction import Predictor

REQUEST_LOG_KEYS = {
    "record_type",
    "experiment_id",
    "scenario",
    "task_id",
    "outcome",
    "error",
    "ground_truth_request_type",
    "classified_request_type",
    "misclassified",
    "phase",
    "arrival_time_utc",
    "finish_time_utc",
    "arrival_time_mono",
    "finish_time_mono",
    "arrival_elapsed_s",
    "queue_delay_s",
    "e2e_latency_s",
    "estimated_latency_s",
    "slo_s",
    "slo_met",
    "attempt_count",
    "retry_errors",
    "prompt_tokens",
    "predicted_output_tokens",
    "reference_output_tokens",
    "prediction_latency_s",
    "source_request_type",
    "max_tokens",
    "actual_output_tokens",
    "worker",
}


class RouterSwitchTests(unittest.TestCase):
    def make_configs(self, root: Path, qps: float) -> dict[str, dict[str, object]]:
        configs = {}
        for request_type in router.REQUEST_CONFIGS:
            path = root / f"{request_type}.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "prompt": "hello",
                        "input_token_count": 1,
                        "output_token_count": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            configs[request_type] = {
                "qps": [qps if request_type == "short_prompt_short_response" else 0],
                "dataset": path,
            }
        return configs

    @staticmethod
    def wait_state(
        generator: router.TrafficGenerator, state: str, timeout: float = 3
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if generator.dispatch_status()["experiment_state"] == state:
                return
            time.sleep(0.01)
        raise AssertionError(f"state did not become {state}: {generator.status()}")

    def test_phase_config_supports_incremental_registration(self) -> None:
        previous_registry = router.worker_registry
        previous_paths = router.phase_config_paths
        self.addCleanup(setattr, router, "worker_registry", previous_registry)
        self.addCleanup(setattr, router, "phase_config_paths", previous_paths)
        router.worker_registry = router.WorkerRegistry("phase1")
        router.phase_config_paths = {
            2: Path(__file__).resolve().parents[1] / "configs" / "phase2.json"
        }

        config = router.load_phase_config(2)
        self.assertEqual(
            config["workers"][0]["monitor"],
            {"url": "http://phase2-ss.example.org:9100", "required": True},
        )

        first = router.register_phase_workers(2, {"phase2-ss"})
        self.assertEqual(len(first["registered"]), 1)
        self.assertFalse(first["all_configured_workers_registered"])
        remaining = router.register_phase_workers(2, None)
        self.assertTrue(remaining["all_configured_workers_registered"])
        self.assertEqual(
            router.worker_registry.count_for_type(
                "short_prompt_short_response", active_only=False
            ),
            1,
        )

    def test_phase_configs_require_explicit_worker_fields_and_distinct_ids(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = {
                "worker_id": "worker-1",
                "ip": "127.0.0.1",
                "port": 8000,
                "request_type": "short_prompt_short_response",
                "weight": 1,
                "gpu_ids": [0],
                "health_url": "http://127.0.0.1:8000/v1/models",
                "monitor": {"url": "http://127.0.0.1:9100", "required": True},
            }
            phase1 = root / "phase1.json"
            phase2 = root / "phase2.json"
            phase1.write_text(
                json.dumps({"configuration_id": "same", "workers": [worker]}),
                encoding="utf-8",
            )
            phase2_worker = {**worker, "worker_id": "worker-2", "port": 8100}
            phase2_worker["health_url"] = "http://127.0.0.1:8100/v1/models"
            phase2_worker["monitor"] = {
                "url": "http://127.0.0.1:9200",
                "required": True,
            }
            phase2.write_text(
                json.dumps({"configuration_id": "same", "workers": [phase2_worker]}),
                encoding="utf-8",
            )
            previous_paths = router.phase_config_paths
            router.phase_config_paths = {1: phase1, 2: phase2}
            self.addCleanup(setattr, router, "phase_config_paths", previous_paths)
            with self.assertRaisesRegex(ValueError, "must be different"):
                router.validate_distinct_phase_configuration_ids()

            del worker["worker_id"]
            phase1.write_text(
                json.dumps({"configuration_id": "phase1", "workers": [worker]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "worker_id is required"):
                router.load_phase_config(1)

    def test_cold_clock_excludes_gpu_wait_and_preserves_request_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = router.WorkerRegistry("phase1")
            generator = router.TrafficGenerator(
                registry,
                predictor=Predictor(1),
                input_threshold=100,
                configs=self.make_configs(root, 20),
                completion_log_path=root / "requests.jsonl",
                request_sender_threads=2,
                max_in_flight_requests=4,
                experiment_duration_seconds=0.8,
                qps_segment_seconds=0.8,
                switch_at_seconds=0.3,
            )
            generator._send_request = lambda _worker, _sample: 1
            generator.start()
            self.addCleanup(generator.stop)
            generator.activate()
            self.wait_state(generator, "cold_waiting_for_gpu")
            frozen = generator.elapsed_seconds()
            time.sleep(0.15)
            self.assertAlmostEqual(generator.elapsed_seconds(), frozen, delta=0.02)

            generator.cold_resource_ready()
            registry.add(
                "127.0.0.1",
                9000,
                "short_prompt_short_response",
                1,
                configuration_id="phase2",
                tp_size=1,
            )
            generator.activate_configuration("phase2")
            generator.resume_dispatch()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not generator.status()["drained"]:
                time.sleep(0.02)
            self.assertTrue(generator.status()["drained"])
            records = [
                json.loads(line)
                for line in (root / "requests.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertGreater(len(records), 0)
            self.assertEqual(set(records[0]), REQUEST_LOG_KEYS)
            self.assertLess(records[0]["e2e_latency_s"], 0.8)

    def test_monitor_records_gpu_rows_while_vllm_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = monitor_module.VLLMMonitor(
                worker_id="phase2-spsr",
                vllm_host="127.0.0.1",
                vllm_port=8999,
                gpu_ids=[0, 1],
                interval_seconds=0.01,
                output_directory=temporary,
            )
            service._get_gpu_stats = lambda: (321.5, 1410)
            service._get_vllm_metrics = lambda: None
            started, _ = service.start_recording(
                {
                    "session_id": "run-1:phase2",
                    "experiment_id": "run-1",
                    "scenario": "cold-switch",
                    "phase": 2,
                    "logical_start_seconds": 1800,
                    "expected_duration_seconds": 1800,
                }
            )
            self.assertTrue(started)
            time.sleep(0.04)
            stopped, status = service.stop_recording({"session_id": "run-1:phase2"})
            self.assertTrue(stopped)
            self.assertFalse(status["recording"])
            self.assertGreater(status["rows_written"], 0)
            rows = (
                Path(str(status["output_csv_file"]))
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertIn("vllm_available", rows[0])
            self.assertIn("False", rows[1])

    def test_monitor_creates_distinct_startup_and_running_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = monitor_module.VLLMMonitor(
                worker_id="phase2-spsr",
                gpu_ids=[0],
                interval_seconds=0.01,
                output_directory=temporary,
            )
            service._get_gpu_stats = lambda: (100.0, 1200)
            service._get_vllm_metrics = lambda: None
            common = {
                "experiment_id": "run-1",
                "scenario": "cold-switch",
                "phase": 2,
                "logical_start_seconds": 1800,
                "expected_duration_seconds": 1800,
            }
            service.start_recording(
                {
                    **common,
                    "session_id": "run-1:startup",
                    "recording_window": "phase2-vllm-startup",
                }
            )
            time.sleep(0.02)
            _, startup_status = service.stop_recording({"session_id": "run-1:startup"})
            service.start_recording(
                {
                    **common,
                    "session_id": "run-1:running",
                    "recording_window": "phase2-running",
                }
            )
            time.sleep(0.02)
            _, running_status = service.stop_recording({"session_id": "run-1:running"})
            startup_path = Path(str(startup_status["output_csv_file"]))
            running_path = Path(str(running_status["output_csv_file"]))
            self.assertNotEqual(startup_path, running_path)
            self.assertIn("phase2-vllm-startup", startup_path.name)
            self.assertIn("phase2-running", running_path.name)
            self.assertTrue(startup_path.exists())
            self.assertTrue(running_path.exists())

    def test_cold_monitor_rotates_each_worker_from_startup_to_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase2_path = root / "phase2.json"
            phase2_path.write_text(
                json.dumps(
                    {
                        "configuration_id": "phase2",
                        "workers": [
                            {
                                "worker_id": "phase2-spsr",
                                "ip": "127.0.0.1",
                                "port": 8100,
                                "request_type": "short_prompt_short_response",
                                "weight": 1,
                                "tp_size": 4,
                                "gpu_ids": [0, 1, 2, 3],
                                "health_url": "http://127.0.0.1:8100/v1/models",
                                "monitor": {
                                    "url": "http://127.0.0.1:9200",
                                    "required": True,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            previous_paths = router.phase_config_paths
            router.phase_config_paths = {2: phase2_path}
            self.addCleanup(setattr, router, "phase_config_paths", previous_paths)
            calls = []
            generator = router.TrafficGenerator(
                router.WorkerRegistry("phase1"),
                predictor=Predictor(1),
                input_threshold=100,
                configs=self.make_configs(root, 0),
                completion_log_path=root / "requests.jsonl",
                request_sender_threads=1,
                max_in_flight_requests=1,
                experiment_duration_seconds=0.5,
                qps_segment_seconds=0.5,
                switch_at_seconds=0.15,
            )

            def fake_monitor_request(url, method, payload=None):
                calls.append((url, method, payload))
                if url.endswith("/health"):
                    return {"worker_id": "phase2-spsr", "gpu_ids": [0, 1, 2, 3]}
                return {"recording": url.endswith("/start")}

            generator._monitor_json_request = fake_monitor_request
            generator.start()
            self.addCleanup(generator.stop)
            generator.activate()
            self.wait_state(generator, "cold_waiting_for_gpu")
            generator.cold_resource_ready()
            transition = generator.cold_worker_ready("phase2-spsr")
            self.assertFalse(transition["already_transitioned"])
            self.assertEqual(
                [call[0].rsplit("/", 1)[-1] for call in calls],
                ["health", "start", "stop", "health", "start"],
            )
            start_payloads = [
                payload for url, _, payload in calls if url.endswith("/start")
            ]
            self.assertEqual(
                [payload["recording_window"] for payload in start_payloads],
                ["phase2-vllm-startup", "phase2-running"],
            )

    def test_cutoff_during_running_monitor_start_stops_late_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = {
                "worker_id": "late",
                "ip": "127.0.0.1",
                "port": 8100,
                "request_type": "short_prompt_short_response",
                "weight": 1,
                "gpu_ids": [0],
                "health_url": "http://127.0.0.1:8100/v1/models",
                "monitor": {"url": "http://127.0.0.1:9200", "required": True},
            }
            phase = root / "phase2.json"
            phase.write_text(
                json.dumps({"configuration_id": "phase2", "workers": [worker]})
            )
            previous = router.phase_config_paths
            router.phase_config_paths = {2: phase}
            self.addCleanup(setattr, router, "phase_config_paths", previous)
            generator = router.TrafficGenerator(
                router.WorkerRegistry("phase1"),
                predictor=Predictor(1),
                input_threshold=100,
                configs=self.make_configs(root, 0),
                completion_log_path=root / "requests.jsonl",
                experiment_duration_seconds=2,
                qps_segment_seconds=2,
                switch_at_seconds=1,
            )
            self.addCleanup(generator.stop)
            calls = []

            def request(url, method, payload=None):
                calls.append(url)
                if url.endswith("/health"):
                    return {"worker_id": "late", "gpu_ids": [0]}
                if (
                    url.endswith("/start")
                    and payload["recording_window"] == "phase2-running"
                ):
                    generator.finish_experiment()
                return {"recording": url.endswith("/start")}

            generator._monitor_json_request = request
            generator.activate()
            generator.begin_scheduled_switch()
            generator.cold_resource_ready()
            with self.assertRaisesRegex(ValueError, "experiment ended"):
                generator.cold_worker_ready("late")
            self.assertTrue(calls[-1].endswith("/stop"))
            self.assertFalse(generator._monitor_sessions[2])
            self.assertTrue(generator.status()["drained"])


if __name__ == "__main__":
    unittest.main()
