"""Online prediction, ordering, failure and cold-clock regressions; no model/GPU needed."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import router
from predictors import (
    bert_predictor,
    load_predictor,
    request_type_for,
    validate_prediction,
)

SS, SL, LS, LL = router.DATASET_FILENAMES


class Predictor:
    def __init__(self, result=500, delay=0):
        self.result, self.delay, self.calls = result, delay, []

    def predict(self, prompt):
        self.calls.append(prompt)
        time.sleep(self.delay)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.old_paths = router.phase_config_paths
        router.phase_config_paths = {}
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(setattr, router, "phase_config_paths", self.old_paths)

    def wait(self, condition, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.005)
        self.fail("condition timed out")

    def make(self, predictor, **kwargs):
        configs = {}
        for kind in router.DATASET_FILENAMES:
            path = self.root / (kind + ".jsonl")
            path.write_text(
                json.dumps(
                    {
                        "prompt": "prompt",
                        "input_token_count": 20,
                        "output_token_count": 10,
                    }
                )
                + "\n"
            )
            configs[kind] = {"qps": [0], "dataset": path}
        registry = router.WorkerRegistry("phase1")
        for i, kind in enumerate(router.DATASET_FILENAMES):
            registry.add("127.0.0.1", 8000 + i, kind, 1, "phase1")
        defaults = dict(
            experiment_duration_seconds=20,
            qps_segment_seconds=20,
            switch_at_seconds=10,
            request_sender_threads=4,
        )
        defaults.update(kwargs)
        g = router.TrafficGenerator(
            registry,
            predictor=predictor,
            input_threshold=100,
            configs=configs,
            completion_log_path=self.root / "requests.jsonl",
            **defaults
        )
        g._send_request = lambda worker, sample: sample.max_tokens
        self.addCleanup(g.stop)
        g.start()
        g.activate()
        return g

    def records(self):
        return [
            json.loads(x)
            for x in (self.root / "requests.jsonl").read_text().splitlines()
        ]

    def test_threshold_boundaries_and_validation(self):
        self.assertEqual(request_type_for(99, 443, 100, 444), SS)
        self.assertEqual(request_type_for(100, 443, 100, 444), LS)
        self.assertEqual(request_type_for(99, 444, 100, 444), SL)
        self.assertEqual(request_type_for(100, 444, 100, 444), LL)
        for value in (-1, float("nan"), float("inf"), True, "500"):
            with self.assertRaises(ValueError):
                validate_prediction(value)

    def test_routing_uses_prediction_but_preserves_generation_budget_and_slo(self):
        predictor = Predictor(500, 0.04)
        g = self.make(predictor)
        # Deliberately supply an inconsistent source group: routing uses token counts.
        task = g.enqueue_arrival(LL, router.DatasetRequest("hello", 20, 10, 10))
        self.wait(lambda: len(self.records()) == 1)
        record = self.records()[0]
        self.assertEqual(record["worker"]["request_type"], SL)
        self.assertEqual(record["ground_truth_request_type"], SS)
        self.assertEqual(record["reference_output_tokens"], 10)
        self.assertEqual(record["predicted_output_tokens"], 500)
        self.assertEqual(record["max_tokens"], 10)
        self.assertEqual(
            record["slo_s"],
            router.estimate_latency_seconds(LL, 10) * router.SLO_MULTIPLIER,
        )
        self.assertGreaterEqual(record["e2e_latency_s"], 0.035)
        self.assertGreaterEqual(record["queue_delay_s"], record["prediction_latency_s"])
        self.assertTrue(record["misclassified"])
        self.assertEqual(record["task_id"], task.task_id)

    def test_prediction_does_not_block_arrivals_and_orders_mixed_sources(self):
        g = self.make(Predictor(500, 0.04))
        sent = []
        g._send_request = lambda worker, sample: (sent.append(sample.prompt) or 1)
        start = time.monotonic()
        for i in range(6):
            g.enqueue_arrival(
                SS if i % 2 else LL, router.DatasetRequest(str(i), 20, 10, 10)
            )
        self.assertLess(time.monotonic() - start, 0.1)
        self.assertGreater(
            sum(v["pending"] for v in g.status()["request_types"].values()), 1
        )
        self.wait(lambda: len(sent) == 6)
        self.assertEqual(sent, [str(i) for i in range(6)])

    def test_prediction_failure_is_logged_without_oracle_fallback(self):
        g = self.make(Predictor(RuntimeError("model failed")))
        g.enqueue_arrival(SS, router.DatasetRequest("hello", 20, 10, 10))
        self.wait(lambda: len(self.records()) == 1)
        record = self.records()[0]
        self.assertEqual(record["outcome"], "failed")
        self.assertIsNone(record["classified_request_type"])
        self.assertIsNone(record["worker"])
        self.assertIn("model failed", record["error"])
        self.assertEqual(record["attempt_count"], 0)

    def test_missing_weights_fail_before_importing_torch(self):
        with self.assertRaisesRegex(FileNotFoundError, "Releases"):
            bert_predictor(str(self.root / "no-model"))
        predictor = load_predictor("test_prediction:Predictor", '{"result": 100}')
        self.assertEqual(predictor.predict("hello"), 100)

    def test_cold_reuses_prediction_and_excludes_frozen_time(self):
        predictor = Predictor(500)
        g = self.make(
            predictor,
            experiment_duration_seconds=1,
            qps_segment_seconds=1,
            switch_at_seconds=0.2,
        )
        release_old = threading.Event()
        self.addCleanup(release_old.set)

        def send(worker, sample):
            if worker.configuration_id == "phase1":
                release_old.wait(3)
            return 10

        g._send_request = send
        task = g.enqueue_arrival(SS, router.DatasetRequest("hello", 20, 10, 10))
        self.wait(
            lambda: g.dispatch_status()["experiment_state"] == "cold_waiting_for_gpu"
        )
        time.sleep(0.15)
        self.assertAlmostEqual(g.elapsed_seconds(), 0.2, delta=0.005)
        g.cold_resource_ready()
        g.registry.add("127.0.0.1", 8100, SL, 1, "phase2")
        g.activate_configuration("phase2")
        g.resume_dispatch()
        self.wait(lambda: len(self.records()) == 1)
        release_old.set()
        record = self.records()[0]
        self.assertEqual(record["task_id"], task.task_id)
        self.assertEqual(record["attempt_count"], 2)
        self.assertEqual(len(predictor.calls), 1)
        self.assertLess(record["e2e_latency_s"], 0.35)
        self.wait(lambda: g.status()["drained"])
        self.assertEqual(len(self.records()), 1)

    def test_unfinished_prediction_survives_boundary_and_cutoff_discards_pending(self):
        predictor = Predictor(500, 0.12)
        g = self.make(
            predictor,
            experiment_duration_seconds=0.5,
            qps_segment_seconds=0.5,
            switch_at_seconds=0.08,
        )
        task = g.enqueue_arrival(SS, router.DatasetRequest("hello", 20, 10, 10))
        self.wait(
            lambda: g.dispatch_status()["experiment_state"] == "cold_waiting_for_gpu"
        )
        time.sleep(0.1)
        self.assertEqual(self.records(), [])
        g.cold_resource_ready()
        self.wait(lambda: task.classified_request_type == SL)
        # No phase-2 dispatch: hard cutoff also discards the predicted task.
        self.wait(lambda: g.status()["drained"])
        self.assertEqual(self.records(), [])
        self.assertEqual(g.status()["request_types"][SS]["abandoned"], 1)
        self.assertEqual(len(predictor.calls), 1)

    def test_full_prediction_queue_logs_drop(self):
        entered, release = threading.Event(), threading.Event()

        class Slow:
            def predict(self, prompt):
                entered.set()
                release.wait(3)
                return 10

        g = self.make(Slow(), max_prediction_pending=1)
        self.addCleanup(release.set)
        g.enqueue_arrival(SS, router.DatasetRequest("one", 20, 10, 10))
        self.assertTrue(entered.wait(1))
        g.enqueue_arrival(SS, router.DatasetRequest("two", 20, 10, 10))
        g.enqueue_arrival(SS, router.DatasetRequest("three", 20, 10, 10))
        self.assertEqual(self.records()[0]["outcome"], "dropped")
        release.set()


if __name__ == "__main__":
    unittest.main()
