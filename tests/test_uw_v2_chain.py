"""Exercise fail-fast cleanup, state triggers, and statistics without training."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/uwv2'))
import chain
from common import read_json, write_json


class TestChain(unittest.TestCase):
    def test_three_workers_failure_terminates_siblings(self):
        original = subprocess.Popen
        processes = []
        def spawn(command, **kwargs):
            seed = int(command[command.index('--seed') + 1])
            code = 'import time; time.sleep(0.3); raise SystemExit(7)' if seed == 1 else 'import time; time.sleep(60)'
            process = original([sys.executable, '-c', code], **kwargs)
            processes.append(process)
            return process
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'train').mkdir()
            write_json(root / 'state.json', dict(status='running'))
            args = SimpleNamespace(variant='B0', resume=False)
            with patch.object(chain.subprocess, 'Popen', side_effect=spawn):
                with self.assertRaisesRegex(RuntimeError, 'exited 7'):
                    chain.run_workers(args, root, 'train', [0, 1, 2])
            self.assertEqual(len(processes), 3)
            self.assertTrue(all(p.poll() is not None for p in processes))
            self.assertEqual(read_json(root / 'state.json')['worker_pids'], [])

    def test_all_terminal_states_trigger_and_missing_state_waits(self):
        class Executed(Exception):
            pass
        for terminal in ('completed', 'failed', 'cancelled'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                own = dict(status='waiting')
                upstream = dict(status=terminal, failure_reason='upstream example')
                # Own-state check, missing upstream, then terminal upstream.
                with patch.object(chain, 'read_json', side_effect=[own, FileNotFoundError(), upstream]), \
                     patch.object(chain, 'state_update') as update, \
                     patch.object(chain.time, 'sleep') as sleep, \
                     patch.object(chain.os, 'execv', side_effect=Executed) as execute:
                    with self.assertRaises(Executed):
                        chain.wait_upstream(SimpleNamespace(upstream=Path('/tmp/other/state.json'),
                            variant='B1', run_id='test'), root)
                    sleep.assert_called_once_with(30)
                    execute.assert_called_once()
                    self.assertEqual(update.call_args.kwargs['upstream_status'], terminal)

    def test_seed_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed, value in enumerate([0.4, 0.5, 0.6]):
                out = root / 'test' / f'seed{seed}'
                write_json(out / 'summary_metrics.json', dict(seed=seed, metrics={'mAP50_95': value}))
                write_json(out / 'scale_ap_metrics.json', {})
            report = chain.aggregate(root)
            self.assertAlmostEqual(report['statistics']['mAP50_95']['mean'], 0.5)
            self.assertAlmostEqual(report['statistics']['mAP50_95']['std_sample'], 0.1)
            self.assertEqual(report['best_seed'], 2)
            self.assertEqual(report['worst_seed'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
