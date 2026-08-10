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
    checkpoint = args.resume_d0_root / "legacy_d0_seed0" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    runner = TDRMatrixRunner(args)
    assert runner.resume_weight(0) == checkpoint


def test_mean_std_reports_zero_spread_for_single_seed():
    assert mean_std([12.5]) == {"mean": 12.5, "std": 0.0}
