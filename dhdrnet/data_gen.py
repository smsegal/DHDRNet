import itertools
from pathlib import Path
from typing import Callable, Collection

import image_loader as il
from image_loader import DATA_DIR, FuseMethod


def main():
    gen(DATA_DIR/"dngs".iterdir(), DATA_DIR/"merged",FuseMethod.Debevec, is_script=True)
    pass


def gen(
    image_paths: Collection[Path],
    out_dir: Path,
    fuse_fun: Callable[[Collection[Path], FuseMethod], Collection[Path]],
    is_script: bool = False
) -> Collection[Path]:
    image_paths = list(image_paths)
    processed_dir = DATA_DIR / "processed"

    if not processed_dir.exists() or len(list(processed_dir.iterdir()))==0:
        processed_dir.mkdir(exists_ok=True)
        multi_exposure_paths = il.write_multi_exposure(image_paths)
    else:
        multi_exposure_paths = processed_dir.iterdir()

    if not out_dir.exists() or len(list(out_dir.iterdir()))== 0:
        if is_script:
            should_overwrite = input("merged out_dir is not empty, overwrite conflicting files?")
        else:
            should_overwrite = True
            

    grouped_ldr_paths: Mapping[str, Collection[Path]] = il.group_ldr_paths(
        multi_exposure_paths
    )
    
    for (name, exp_paths), method in itertools.product(
        multi_exposure_paths.items(), FuseMethod
    ):
        pass
        


if __name__ == "__main__":
    main()
