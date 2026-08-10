from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ultralytics.data.utils import check_det_dataset, verify_image_label


def test_urpc2019_yaml_declares_runtime_one_based_label_conversion(tmp_path: Path):
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"path: '{tmp_path.as_posix()}'\ntrain: images/train\nval: images/val\nnc: 4\nnames:\n  1: echinus\n  2: starfish\n"
        "  3: holothurian\n  4: scallop\n",
        encoding="utf-8",
    )

    data = check_det_dataset(str(data_yaml))
    assert data["label_index_offset"] == 1
    assert data["names"] == {0: "echinus", 1: "starfish", 2: "holothurian", 3: "scallop"}


def test_one_based_urpc_label_is_converted_before_training(tmp_path: Path):
    image_path, label_path = tmp_path / "sample.jpg", tmp_path / "sample.txt"
    Image.new("RGB", (32, 32)).save(image_path)
    label_path.write_text("1 0.5 0.5 0.2 0.2\n4 0.4 0.4 0.3 0.3\n", encoding="utf-8")

    _, labels, *_ = verify_image_label((str(image_path), str(label_path), "", False, 4, 0, 0, 1))
    np.testing.assert_array_equal(labels[:, 0], np.array([0.0, 3.0], dtype=np.float32))


def test_one_based_urpc_label_rejects_zero_class_id(tmp_path: Path):
    image_path, label_path = tmp_path / "sample.jpg", tmp_path / "sample.txt"
    Image.new("RGB", (32, 32)).save(image_path)
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    im_file, _, _, _, _, _, _, _, corrupt, message = verify_image_label(
        (str(image_path), str(label_path), "", False, 4, 0, 0, 1)
    )
    assert im_file is None and corrupt == 1
    assert "1-based range" in message
