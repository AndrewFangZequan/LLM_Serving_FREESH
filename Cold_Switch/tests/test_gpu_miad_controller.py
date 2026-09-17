import contextlib
import io
import unittest

import gpu_miad_controller as miad


class MultiGpuMiadTests(unittest.TestCase):
    def test_parse_multiple_gpu_ids(self) -> None:
        args = miad.parse_args(["--gpu-ids", "0,1,2,3"])
        self.assertEqual(args.gpu_ids, (0, 1, 2, 3))

    def test_single_gpu_argument_remains_compatible(self) -> None:
        args = miad.parse_args(["--gpu-id", "2"])
        self.assertEqual(args.gpu_ids, (2,))

    def test_group_lock_applies_the_same_target_to_every_gpu(self) -> None:
        group = miad.NvidiaGpuGroup((0, 1, 2, 3), dry_run=True)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            group.lock_clock(1200)
        commands = output.getvalue().splitlines()
        self.assertEqual(
            commands,
            [
                f"[dry-run] nvidia-smi -i {gpu_id} --lock-gpu-clocks 1200,1200"
                for gpu_id in range(4)
            ],
        )

    def test_group_stats_are_aggregated_and_keep_per_gpu_clocks(self) -> None:
        group = miad.NvidiaGpuGroup((0, 1, 2, 3), dry_run=False)
        rows = {
            0: "0, 1200, 100, 80, 50, 1000",
            1: "1, 1200, 110, 70, 52, 1100",
            2: "2, 1200, 120, 60, 54, 1200",
            3: "3, 1200, 130, 50, 56, 1300",
        }
        group._run = lambda gpu_id, _arguments: rows[gpu_id]
        stats = group.stats()
        self.assertEqual(stats.clock_mhz, 1200)
        self.assertEqual(
            stats.clocks_mhz_by_gpu,
            {0: 1200, 1: 1200, 2: 1200, 3: 1200},
        )
        self.assertEqual(stats.power_w, 460)
        self.assertEqual(stats.utilization_pct, 65)
        self.assertEqual(stats.temperature_c, 56)
        self.assertEqual(stats.memory_used_mib, 4600)


if __name__ == "__main__":
    unittest.main()
