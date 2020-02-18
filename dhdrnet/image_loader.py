import logging
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Collection, Mapping

import colour as co
import colour_hdri as ch
import imageio as io
import numpy as np
from colour_hdri import (
    ImageStack,
    camera_response_functions_Debevec1997,
    filter_files,
    image_stack_to_radiance_image,
    weighting_function_Debevec1997,
)

DATA_DIR = Path("../../data/HDR+")


def main():
    pass


def ldr_image_fusion(ldr_files: Collection[Path], out_dir):
    image_stack = ldr_files_to_imagestack(ldr_files)
    merged = image_stack_to_radiance_image(
        image_stack, camera_response_functions=camera_response(image_stack)
    )
    return merged


def camera_response(image_stack: ImageStack):
    crfs = camera_response_functions_Debevec1997(image_stack)
    crfs = crfs[np.all(crfs, axis=1)]
    crfs = co.utilities.linear_conversion(crfs, (np.min(crfs), 1), (0, 1))
    return crfs


def ldr_files_to_imagestack(ldr_files: Collection[Path]) -> ImageStack:
    ldr_files = map(str, ldr_files)
    return ImageStack.from_files(ldr_files)  # sorts by lumincance for us


def group_ldr_paths(path: Path) -> Mapping[str, Collection[Path]]:
    """
takes a path containing the dir of all processed pngs, returns them
grouped by name and the paths to each different exposure
    """
    name_to_exposure = sorted(
        map(lambda p: tuple([*p.name.split(".")[:-1], p]), path.iterdir()),
        key=itemgetter(0),
    )
    name_exp = defaultdict(list)
    for name, exps in groupby(name_to_exposure, key=itemgetter(0)):
        for exp in exps:
            name_exp[name].append(exp[-1])

    return name_exp


def read_images(path: Path, limit: int = None) -> Mapping[str, np.ndarray]:
    file_paths = list(path.glob("*.dng"))
    if limit:
        sample = np.random.choice(file_paths, size=limit, replace=False)
    else:
        sample = file_paths

    return {p.name: ch.Image(str(p)) for p in sample}


def gen_multi_exposure(
    image_dict: Mapping[str, np.ndarray]
) -> Mapping[str, Collection[np.ndarray]]:
    multi_exposure = dict.fromkeys(image_dict, [])

    for k, img in image_dict.items():
        image_data = img.read_data()
        metadata = img.read_metadata()
        for exposure in np.linspace(-4, 4, 5):
            multi_exposure[k].append(
                (metadata, ch.adjust_exposure(image_data, exposure))
            )

    return multi_exposure


def write_multi_exposure(image_list):
    for img_path in image_list:
        img = co.io.read_image(img_path)
        for exposure in np.linspace(-4, 4, 5):
            io.imwrite(
                str(
                    (DATA_DIR / "processed")
                    / f"{img_path.name[:-4]}.{int(exposure)}.png"
                ),
                ch.adjust_exposure(img, exposure),
            )


if __name__ == "__main__":
    main()
