import random
from collections import defaultdict
from collections.abc import Iterable as It
from pathlib import Path
from pprint import pprint
from subprocess import check_output
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterator,
    List,
    Mapping,
    Set,
    NewType,
    TypeVar,
    Generic,
    Collection,
)

T = TypeVar("T")


class Indexable(Mapping[int, T]):
    def __init__(self, *args, **kwargs):
        super(self, *args, **kwargs)


def get_project_root() -> Path:
    git_root = (
        check_output(["git", "rev-parse", "--show-toplevel"]).decode("utf-8").strip()
    )
    return Path(git_root).absolute()


ROOT_DIR: Path = get_project_root()
DATA_DIR: Path = ROOT_DIR / "data"


def get_train_test(data_dir: Path):
    pass


def create_train_test_split(
    data_dir: Path, train_split=0.9, val_split=0.2, dry_run=False
):
    files: Set = set((data_dir / "dngs").iterdir())

    train_size = round(train_split * len(files))
    train: Set = set(random.sample(files, k=train_size))

    test = files - train

    val_size = round(val_split * len(train))
    val = set(random.sample(train, k=val_size))

    train = train - val

    if not dry_run:
        for name, split in {"train": train, "test": test, "val": val}.items():
            split_file = ROOT_DIR / f"{name}.txt"
            split_file.write_text("\n".join([fname.stem for fname in split]))

    return files, train, test, val


def split_dataset(
    root_dir: Path = ROOT_DIR, data_dir: Path = DATA_DIR, dry_run: bool = False
) -> Dict[Path, List[Path]]:
    ds_splits = ["train", "test", "val"]
    data_subdirs = ["dngs", "merged", "processed"]
    source_dest_map = defaultdict(list)
    # Making sure that the dataset split directories exist
    for ds_dir in [(data_dir / f"{split}") for split in ds_splits]:
        for subdir, subname in [((ds_dir / subdir), subdir) for subdir in data_subdirs]:
            subdir.mkdir(parents=True, exist_ok=True)
            source_dest_map[data_dir / subname].append(subdir)
    pprint(source_dest_map)

    # getting each file that contains listings
    # and reading into dict keyed by train,test,val
    ds_files: Dict = {split: [] for split in ds_splits}
    for f, split in [(root_dir / f"{f}.txt", f) for f in ds_splits]:
        if f.is_file():
            ds_files[split] = set(f.read_text().splitlines())

    for source, dests in source_dest_map.items():
        source_files = list(source.iterdir())
        for split, names in ds_files.items():
            to_move = flatten(
                [[sf for name in names if name in sf.name] for sf in source_files]
            )
            print(f"{source}: {split}")
            # print(len(to_move))
            # print("\n")
            # pprint(to_move[:10])
            for orig in to_move:
                dest = data_dir / split / source.name
                print(f"{orig} --> {dest / orig.name}")
                # return
                try:
                    (dest / orig.name).symlink_to(orig)
                except FileExistsError as fee:
                    print(fee)

    return source_dest_map


def split_data(data_dir: Path, root_dir: Path) -> DefaultDict[Path, List[Path]]:
    """
    End file hierarchy should look like:
    DATA_DIR /
    {train, test, val} /
    {dngs, merged, processed}
    """
    splits = ["train", "test", "val"]
    data_splits = [
        set(f.read_text().splitlines())
        for f in [root_dir / f"{split}.txt" for split in splits]
    ]
    train, test, val = data_splits

    source_dest_map: DefaultDict[Path, List[Path]] = defaultdict(list)
    for source_dir in [data_dir / st for st in ["dngs", "merged", "processed"]]:
        for split_list, split_name in zip(data_splits, splits):
            target_dir = data_dir / split_name / source_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            source_dest_map[source_dir].append(target_dir)
            for name in split_list:
                source_files: Iterator[Path] = source_dir.glob(f"{name}.*")
                for source_file in source_files:
                    target = target_dir / source_file.name
                    # try:
                    target.symlink_to(source_file)
                    # except FileExistsError as fee:
                    #     continue
    return source_dest_map


def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, It) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x
