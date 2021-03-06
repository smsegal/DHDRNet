import numpy as np
import pytest

from dhdrnet.Dataset import HDRDataset
from dhdrnet.util import (
    create_train_test_split,
    flatten,
    get_project_root,
    split_data,
)


DATA_DIR = get_project_root() / "data"

def test_sample_load():
    dataset = HDRDataset(gt_dir=DATA_DIR / "merged", raw_dir=DATA_DIR / "dngs")
    sample_size = 100
    for i in np.random.choice(len(dataset), size=sample_size):
        sample = dataset[i]
        exposures, gt, _name = sample.values()
        exposure_shapes = [e.shape for e in exposures]
        for e_shape in exposure_shapes:
            assert e_shape == gt.shape


def test_create_train_test_split():
    files, train, test, val = create_train_test_split(DATA_DIR, dry_run=True)
    assert len(files) == len(train) + len(test) + len(val)


# @pytest.mark.skip(reason="time consuming and not idempotent")
def test_split_dataset():
    # create_train_test_split(DATA_DIR)
    source_dest_map = split_data(root_dir=get_project_root(), data_dir=DATA_DIR)
    for source, dests in source_dest_map.items():
        source_files = source.iterdir()
        dest_files = flatten([d.iterdir() for d in dests])
        assert len(list(source_files)) == len(list(dest_files))
