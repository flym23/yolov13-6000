from argparse import Namespace
from pathlib import Path

from tools.run_tdr_matrix import TDRMatrixRunner, atomic_json, mean_std


def _args(tmp_path: Path) -> Namespace:
    root = tmp_path / "project"
    return Namespace(
        project_root=root,
        chain_root=root / "runs" / "chain" / "tdr_unit",
        run_id="unit",
        upstream_state=tmp_path / "upstream" / "state.json",
        upstream_status="completed",
        upstream_reason="",
        resume_d0_root=tmp_path / "resume",
    )


def test_atomic_json_replaces_file_and_preserves_utf8(tmp_path: Path):
    path = tmp_path / "state.json"
    atomic_json(path, {"status": "running", "reason": "标签"})
    assert path.read_text(encoding="utf-8") == '{\n  "status": "running",\n  "reason": "标签"\n}'
    assert not path.with_suffix(".json.tmp").exists()


def test_resume_weight_requires_exactly_one_checkpoint(tmp_path: Path):
    args = _args(tmp_path)
    checkpoint = args.resume_d0_root / "seed0" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    runner = TDRMatrixRunner(args)
    assert runner.resume_weight(0) == checkpoint
    assert runner.training_root == args.project_root / "runs" / "train" / "tdr_unit"
    assert runner.test_root == args.project_root / "runs" / "test" / "tdr_unit"


def test_mean_std_reports_zero_spread_for_single_seed():
    assert mean_std([12.5]) == {"mean": 12.5, "std": 0.0}


def test_summary_excludes_checkpoint_path_from_numeric_statistics(tmp_path: Path):
    args = _args(tmp_path)
    runner = TDRMatrixRunner(args)
    runner.completed = ["d0"]
    record = {
        "variant": "d0",
        "seed": 0,
        "weights": "/tmp/best.pt",
        "P": 80.0,
        "R": 70.0,
        "mAP50": 75.0,
        "mAP75": 60.0,
        "mAP50-95": 55.0,
        "APS": 50.0,
        "APM": 65.0,
        "APL": 70.0,
    }
    summary = runner.write_summary([record])
    assert '"mAP50-95"' in summary.read_text(encoding="utf-8")
    assert summary == args.project_root / "runs" / "test" / "tdr_unit" / "summary" / "summary_metrics.json"
