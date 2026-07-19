from ultralytics.data import utils as data_utils


def test_dataset_cache_write_uses_pid_temp_file_and_atomic_replacement(tmp_path, monkeypatch):
    cache_path = tmp_path / "train.cache"
    monkeypatch.setattr(data_utils, "is_dir_writeable", lambda _: True)

    data_utils.save_dataset_cache_file("train: ", cache_path, {"labels": ["first"]}, "v1")
    data_utils.save_dataset_cache_file("train: ", cache_path, {"labels": ["second"]}, "v2")

    cache = data_utils.load_dataset_cache_file(cache_path)
    assert cache["version"] == "v2"
    assert cache["labels"] == ["second"]
    assert not list(tmp_path.glob("*.tmp.npy"))
    assert not (tmp_path / "train.cache.npy").exists()
