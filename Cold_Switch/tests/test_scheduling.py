"""Deterministic admission/routing regressions; no GPU or network required."""

import collections
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import router

LS = "long_prompt_short_response"


class Predictor:
    def predict(self, prompt_text):
        return 1


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.previous_paths = router.phase_config_paths
        router.phase_config_paths = {}
        self.addCleanup(setattr, router, "phase_config_paths", self.previous_paths)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def configs(self, qps=100):
        result = {}
        for kind in router.REQUEST_CONFIGS:
            path = self.root / (kind + ".jsonl")
            path.write_text(
                json.dumps(
                    {"prompt": "test", "input_token_count": 1, "output_token_count": 1}
                )
                + "\n"
            )
            result[kind] = {"qps": [qps if kind == LS else 0], "dataset": path}
        return result

    def wait(self, condition, message, timeout=3):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if condition():
                return
            time.sleep(0.005)
        self.fail(message)

    def phase2(self):
        workers = []
        for i, weight in enumerate((1, 2, 2, 2)):
            workers.append(
                {
                    "worker_id": "worker-" + str(i),
                    "ip": "127.0.0.1",
                    "port": 8100 + i,
                    "request_type": LS,
                    "weight": weight,
                    "tp_size": 2 if i == 0 else 4,
                    "gpu_ids": [i],
                    "health_url": f"http://127.0.0.1:{8100+i}/v1/models",
                    "monitor": {"url": f"http://127.0.0.1:{9200+i}", "required": True},
                }
            )
        path = self.root / "phase2.json"
        path.write_text(json.dumps({"configuration_id": "phase2", "workers": workers}))
        router.phase_config_paths = {2: path}
        return workers

    def cold_late_join(self, block_phase1=False):
        workers = self.phase2()
        registry = router.WorkerRegistry("phase1")
        registry.add("127.0.0.1", 8000, LS, 1, "phase1")
        generator = router.TrafficGenerator(
            registry,
            configs=self.configs(),
            predictor=Predictor(),
            input_threshold=1,
            completion_log_path=self.root / "requests.jsonl",
            request_sender_threads=7,
            max_in_flight_requests=32,
            experiment_duration_seconds=15,
            qps_segment_seconds=15,
            switch_at_seconds=0.15,
        )
        old_release, new_release = threading.Event(), threading.Event()
        starts = collections.Counter()
        lock = threading.Lock()

        def send(worker, sample):
            with lock:
                starts[(worker.configuration_id, worker.port)] += 1
            if worker.configuration_id == "phase1":
                if block_phase1:
                    old_release.wait(10)
            else:
                new_release.wait(10)
            return 1

        def monitor(url, method, payload=None):
            if url.endswith("/health"):
                worker = next(
                    w for w in workers if url == w["monitor"]["url"] + "/health"
                )
                return {"worker_id": worker["worker_id"], "gpu_ids": worker["gpu_ids"]}
            return {"recording": url.endswith("/start")}

        generator._send_request = send
        generator._monitor_json_request = monitor
        generator.start()
        try:
            generator.activate()
            self.wait(
                lambda: generator.dispatch_status()["experiment_state"]
                == "cold_waiting_for_gpu",
                "no boundary",
            )
            generator.cold_resource_ready()
            registry.add("127.0.0.1", 8100, LS, 1, "phase2")
            generator.activate_configuration("phase2")
            generator.resume_dispatch()
            self.wait(
                lambda: starts[("phase2", 8100)] > 0, "retained worker did not send"
            )
            self.wait(
                lambda: generator.status()["request_types"][LS]["pending"] >= 12,
                "no backlog",
            )
            with generator._tasks_lock:
                original = {
                    task.task_id: (task.active_attempt_id, task.last_worker)
                    for task in generator._active_tasks.values()
                    if task.state == "in_flight"
                }
            # No old Phase-2 HTTP request is released. Newly ready workers must
            # still receive work, one by one, before the final worker joins.
            for port in (8101, 8102, 8103):
                registry.add("127.0.0.1", port, LS, 2, "phase2")
                self.wait(
                    lambda: starts[("phase2", port)] >= 2,
                    f"new worker {port} starved behind existing HTTP calls",
                )
            with lock:
                self.assertEqual(
                    [starts[("phase2", p)] for p in range(8100, 8104)], [1, 2, 2, 2]
                )
            status = generator.status()["request_types"][LS]
            self.assertEqual(status["in_flight"], 7)
            self.assertGreater(status["pending"], 0)
            with generator._tasks_lock:
                for task_id, attempt in original.items():
                    task = generator._active_tasks[task_id]
                    self.assertEqual(
                        (task.active_attempt_id, task.last_worker), attempt
                    )
                untouched = [
                    t
                    for t in generator._active_tasks.values()
                    if t.state == "pending" and t.attempt_count == 0
                ]
                self.assertTrue(untouched)
                self.assertTrue(all(t.last_worker is None for t in untouched))
            if block_phase1:
                old_release.set()
                for executor in generator._retired_executors:
                    executor.shutdown(wait=True)
                # Stale completions must not release this generation's slots.
                self.assertEqual(sum(generator._sender_active.values()), 7)
                self.assertFalse(generator._capacity.acquire(blocking=False))
            new_release.set()
            self.wait(
                lambda: generator.status()["request_types"][LS]["succeeded"] >= 7,
                "completions did not resume",
            )
        finally:
            old_release.set()
            new_release.set()
            generator.stop()

    def test_cold_new_workers_send_before_old_requests_finish(self):
        self.cold_late_join()

    def test_retired_cold_callbacks_do_not_release_new_sender_slots(self):
        self.cold_late_join(block_phase1=True)

    def test_weighted_selection_after_workers_join(self):
        registry = router.WorkerRegistry("phase2")
        registry.add("127.0.0.1", 8100, LS, 1, "phase2")
        for _ in range(20):
            self.assertEqual(registry.choose(LS).port, 8100)
        for port in (8101, 8102, 8103):
            registry.add("127.0.0.1", port, LS, 2, "phase2")
        eligible = {("phase2", "127.0.0.1", p) for p in range(8100, 8104)}
        counts = collections.Counter(
            registry.choose(LS, eligible).port for _ in range(700)
        )
        self.assertEqual([counts[p] for p in range(8100, 8104)], [100, 200, 200, 200])
        self.assertIsNone(registry.choose(LS, set()))
        for _ in range(10):
            self.assertEqual(
                registry.choose(LS, {("phase2", "127.0.0.1", 8103)}).port, 8103
            )

    def test_plan_reserves_all_256_senders_for_full_phase(self):
        self.phase2()
        generator = router.TrafficGenerator(
            router.WorkerRegistry("phase2"),
            configs=self.configs(0),
            predictor=Predictor(),
            input_threshold=1,
            completion_log_path=self.root / "requests.jsonl",
            experiment_duration_seconds=15,
            qps_segment_seconds=15,
        )
        try:
            generator._configured_monitors(2)
            limits = generator._sender_limits()
            self.assertEqual(
                [limits[("phase2", "127.0.0.1", p)] for p in range(8100, 8104)],
                [37, 73, 73, 73],
            )
            self.assertEqual(sum(limits.values()), 256)
        finally:
            generator.stop()

    def test_too_few_senders_rejected_before_recording(self):
        self.phase2()
        generator = router.TrafficGenerator(
            router.WorkerRegistry("phase2"),
            configs=self.configs(0),
            predictor=Predictor(),
            input_threshold=1,
            completion_log_path=self.root / "requests.jsonl",
            request_sender_threads=2,
            experiment_duration_seconds=15,
            qps_segment_seconds=15,
        )
        try:
            with self.assertRaisesRegex(ValueError, "number of configured workers"):
                generator._configured_monitors(2)
        finally:
            generator.stop()

    def test_submission_and_http_failures_release_sender_slots(self):
        registry = router.WorkerRegistry("phase1")
        registry.add("127.0.0.1", 8000, LS, 1, "phase1")
        generator = router.TrafficGenerator(
            registry,
            configs=self.configs(),
            predictor=Predictor(),
            input_threshold=1,
            completion_log_path=self.root / "requests.jsonl",
            request_sender_threads=1,
            max_in_flight_requests=32,
            experiment_duration_seconds=15,
            qps_segment_seconds=15,
        )
        submit = generator._executor.submit
        submissions, sends = 0, 0

        def fail_first_submission(*args, **kwargs):
            nonlocal submissions
            submissions += 1
            if submissions == 1:
                raise RuntimeError("test submission failure")
            return submit(*args, **kwargs)

        def fail_first_send(worker, sample):
            nonlocal sends
            sends += 1
            if sends == 1:
                raise RuntimeError("test HTTP failure")
            return 1

        generator._executor.submit = fail_first_submission
        generator._send_request = fail_first_send
        generator.start()
        try:
            generator.activate()
            self.wait(
                lambda: generator.status()["request_types"][LS]["succeeded"] >= 3,
                "sender slot leaked after failure",
            )
            self.assertEqual(generator.status()["request_types"][LS]["failed"], 2)
        finally:
            generator.stop()
        self.assertEqual(sum(generator._sender_active.values()), 0)


if __name__ == "__main__":
    unittest.main()
