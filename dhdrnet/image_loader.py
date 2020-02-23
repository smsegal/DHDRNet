import logging
import os
import shlex
import subprocess
from collections import defaultdict
from functools import singledispatch
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Collection, Mapping

import colour as co
import colour_hdri as ch
import cv2 as cv
import imageio as io
import numpy as np
import rawpy
from colour_hdri import (
    Image,
    ImageStack,
    camera_response_functions_Debevec1997,
    filter_files,
    image_stack_to_radiance_image,
    weighting_function_Debevec1997,
)

DATA_DIR = Path("../../data/HDR+")


def main():
    pass


def ldr_image_fusion(ldr_files: Collection[Path]):
    image_stack = ldr_files_to_imagestack(ldr_files)
    merged = image_stack_to_radiance_image(image_stack)
    return merged


def camera_response(image_stack: ImageStack):
    crfs = camera_response_functions_Debevec1997(image_stack)
    crfs = crfs[np.all(crfs, axis=1)]
    crfs = co.utilities.linear_conversion(crfs, (np.min(crfs), 1), (0, 1))
    return crfs


def ldr_files_to_imagestack(ldr_files: Collection[Path]) -> ImageStack:
    ldr_files = map(str, ldr_files)
    image_stack = ImageStack()
    for f in ldr_files:
        image = Image(str(f))
        image.read_data()
        image.read_metadata()
        image_stack.append(image)
    return image_stack


def group_ldr_paths(path: Path) -> Mapping[str, Collection[Path]]:
    """
takes a path containing the dir of all processed pngs, returns them
grouped by name and the paths to each different exposure
each list is sorted by exposure
    """
    name_to_exposure = sorted(
        map(lambda p: tuple([*p.name.split(".")[:-1], p]), path.iterdir()),
        key=itemgetter(0),
    )
    name_exp = defaultdict(list)
    for name, exps in groupby(name_to_exposure, key=itemgetter(0)):
        for exp in sorted(exps, key=lambda e: int(e[1])):
            name_exp[name].append(exp)

    return name_exp


def read_images(path: Path, limit: int = None) -> Mapping[str, np.ndarray]:
    file_paths = list(path.glob("*.dng"))
    if limit:
        sample = np.random.choice(file_paths, size=limit, replace=False)
    else:
        sample = file_paths

    images = {p.name: ch.Image(str(p)) for p in sample}
    for name, image in images.items():
        image.read_data()
    return images, sample


def gen_multi_exposure(
    image_dict: Mapping[str, np.ndarray]
) -> Mapping[str, Collection[np.ndarray]]:
    multi_exposure = dict.fromkeys(image_dict, [])

    for k, img in image_dict.items():
        # image_data = img.read_data()
        # metadata = img.read_metadata()
        for exposure in np.linspace(-4, 4, 5):
            multi_exposure[k].append(
                (img.metadata, ch.adjust_exposure(img.data, exposure))
            )

    return multi_exposure


def norm_uint8(img: np.ndarray) -> np.ndarray:
    return cv.normalize(
        src=img, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U
    )


def write_multi_exposure(image_list: Collection[Path]):
    for img_path in image_list:
        img = ch.Image(str(img_path))
        img.read_metadata()
        img.read_data()
        for exposure in np.linspace(-4, 4, 5):
            img_name = f"{img_path.name[:-4]}.{int(exposure)}.png"
            co.write_image(
                norm_uint8(ch.adjust_exposure(img.data, exposure)),
                str((DATA_DIR / "processed") / img_name),
            )


def mertens_fuse(images: Collection[Path]):
    mertens_merger = cv.createMergeMertens()
    loaded_images = [cv.imread(str(img_path)) for img_path in images]

    res_name = f"{images[0].name.split('.')[0]}.mertens.png"
    out = DATA_DIR / "merged" / res_name
    cv.imwrite(str(out), clip_hdr(mertens_merger.process(loaded_images)))
    return out


def debevec_fuse(images: Collection[Path]):
    debevec_merger = cv.createMergeDebevec()
    return cv_fuse(images, debevec_merger.process, "debevec")


def robertson_fuse(images: Collection[Path]):
    robertson_merger = cv.createMergeRobertson()
    return cv_fuse(images, robertson_merger.process, "robertson")


def cv_fuse(images: Collection[Path], fuse_func, method):
    loaded_images = [cv.imread(str(img_path)) for img_path in images]
    exposure_levels = [int(image.name.split(".")[1]) for image in images]
    exp_min = np.min(exposure_levels)
    exp_max = np.max(exposure_levels)
    exp_normed_shift = (exposure_levels - exp_min + 1) / (exp_max - exp_min)
    tonemap = cv.createTonemap(gamma=2.2)
    hdr = fuse_func(loaded_images, times=exp_normed_shift.copy())
    result = tonemap.process(hdr.copy())

    res_name = f"{images[0].name.split('.')[0]}.{method}.png"
    out = DATA_DIR / "merged" / res_name
    cv.imwrite(str(out), clip_hdr(result))
    return out


def merge_all_test(merge_exp_paths):
    merged = []
    for name, exps in merge_exp_paths.items():
        exp_paths = [e[2] for e in exps]
        merged.append(
            fuse_func(exp_paths)
            for fuse_func in (debevec_fuse, robertson_fuse, mertens_fuse)
        )
    return merged


def clip_hdr(fused):
    return np.clip(fused * 255, 0, 255).astype("uint8")


def _write_tiff_with_metadata(in_file: Path, out_file: Path):
    logging.info(f"Converting {str(in_file)} to Tiff")
    subprocess.call(
        ["dcraw"] + shlex.split('-w -W -H 0 -q 3 -T "{0}"'.format(str(in_file)))
    )


if __name__ == "__main__":
    main()
