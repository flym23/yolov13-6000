from tools import prepare_dcra_dataset_cache


def test_prepare_label_caches_builds_train_and_val_serially(monkeypatch, tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.touch()
    calls = []
    data = {"train": "train-images", "val": "val-images"}

    monkeypatch.setattr(prepare_dcra_dataset_cache, "check_det_dataset", lambda path, autodownload: data)
    monkeypatch.setattr(prepare_dcra_dataset_cache, "get_cfg", lambda overrides: overrides)

    class DummyDataset:
        def __len__(self):
            return 4

    def build(cfg, img_path, batch, data, mode, stride):
        calls.append((cfg, img_path, batch, data, mode, stride))
        return DummyDataset()

    monkeypatch.setattr(prepare_dcra_dataset_cache, "build_yolo_dataset", build)
    prepare_dcra_dataset_cache.prepare_label_caches(data_yaml, imgsz=640, batch=16)

    assert [call[4] for call in calls] == ["train", "val"]
    assert all(call[2] == 16 and call[5] == 32 for call in calls)
