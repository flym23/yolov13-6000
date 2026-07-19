import json
import sys

from tools import collect_dcra_ablation


def test_dcra_stage_summary_matches_d0_schema(tmp_path, monkeypatch):
    run_id = "schema_test"
    stage = "a1_main"
    for seed in (0, 1, 2):
        directory = tmp_path / "runs/test" / f"dcra_{run_id}_{stage}_seed{seed}"
        directory.mkdir(parents=True)
        payload = {
            "metrics": {
                "metrics/precision(B)": 0.80 + seed * 0.01,
                "metrics/recall(B)": 0.70 + seed * 0.01,
                "metrics/mAP50(B)": 0.85 + seed * 0.01,
                "metrics/mAP75(B)": 0.55 + seed * 0.01,
                "metrics/mAP50-95(B)": 0.50 + seed * 0.01,
            },
            "scale_metrics_percent": {
                "APS": 18.0 + seed,
                "APM": 44.0 + seed,
                "APL": 54.0 + seed,
            },
        }
        (directory / "summary_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_dcra_ablation.py",
            "--root",
            str(tmp_path),
            "--run-id",
            run_id,
            "--stages",
            stage,
            "--seeds",
            "0",
            "1",
            "2",
        ],
    )
    collect_dcra_ablation.main()
    result = json.loads(
        (tmp_path / "runs/test" / f"dcra_{run_id}_{stage}_summary.json").read_text(encoding="utf-8")
    )
    assert list(result) == [
        "run_id",
        "stage",
        "config",
        "updated_at",
        "seeds",
        "rows_percent",
        "mean",
        "std",
        "detail",
    ]
    assert list(result["rows_percent"][0]) == [
        "stage",
        "seed",
        "yaml",
        "structure",
        "summary_path",
        "P",
        "R",
        "mAP50",
        "mAP75",
        "mAP50-95",
        "APS",
        "APM",
        "APL",
    ]
    assert list(result["mean"]) == ["P", "R", "mAP50", "mAP75", "mAP50-95", "APS", "APM", "APL"]
    assert list(result["std"]) == list(result["mean"])
    assert list(result["detail"]["mAP50-95"]) == ["mean", "std", "values", "best", "worst"]
