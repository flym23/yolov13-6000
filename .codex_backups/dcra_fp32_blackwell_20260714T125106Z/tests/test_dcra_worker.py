from argparse import Namespace

from tools import train_dcra_worker


def test_server2_worker_uses_reproducible_fp32_training(monkeypatch, tmp_path):
    model_dir = tmp_path / "ultralytics/cfg/models/v13"
    model_dir.mkdir(parents=True)
    (model_dir / "yolov13-dcra.yaml").touch()
    (tmp_path / "data.yaml").touch()
    (tmp_path / "yolov13n.pt").touch()

    calls = {}

    class DummyYOLO:
        def __init__(self, model):
            calls["model"] = model

        def load(self, weights):
            calls["weights"] = weights

        def train(self, **kwargs):
            calls["train"] = kwargs

    monkeypatch.setattr(train_dcra_worker, "YOLO", DummyYOLO)
    monkeypatch.setattr(
        train_dcra_worker,
        "parse_args",
        lambda: Namespace(root=tmp_path, stage="a1_main", name="smoke", seed=2, epochs=1),
    )

    train_dcra_worker.main()

    assert calls["train"]["amp"] is False
    assert calls["train"]["deterministic"] is True
    assert calls["train"]["seed"] == 2
    assert calls["train"]["epochs"] == 1
