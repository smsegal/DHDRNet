import logging
import shlex
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial, singledispatch
from itertools import chain, groupby
from operator import itemgetter
from pathlib import Path
from typing import Callable, Collection, DefaultDict, Iterator, List, Mapping

import colour as co
import colour_hdri as ch
import cv2 as cv
import numpy as np
from colour_hdri import (
    Image,
    ImageStack,
    camera_response_functions_Debevec1997,
    image_stack_to_radiance_image,
)
from deprecated import deprecated

from cv_fuse import FuseMethod

DATA_DIR = Path("../data").resolve()


def ldr_image_fusion(ldr_files: Iterator[Path]):
    image_stack = ldr_files_to_imagestack(ldr_files)
    merged = image_stack_to_radiance_image(image_stack)
    return merged


def camera_response(image_stack: ImageStack):
    crfs = camera_response_functions_Debevec1997(image_stack)
    crfs = crfs[np.all(crfs, axis=1)]
    crfs = co.utilities.linear_conversion(crfs, (np.min(crfs), 1), (0, 1))
    return crfs


def ldr_files_to_imagestack(ldr_files: Iterator[Path]) -> ImageStack:
    image_stack = ImageStack()
    for f in ldr_files:
        image = Image(str(f))
        image.read_data()
        image.read_metadata()
        image_stack.append(image)
    return image_stack


@singledispatch
def group_ldr_paths(
    image_paths: Collection[Path],
) -> DefaultDict[str, Collection[Path]]:
    """
takes the path of the dir of all processed LDR pngs, returns them
grouped by name and the paths to each different exposure
each list is sorted by exposure
    """
    name_to_exposure = sorted(
        map(lambda p: tuple([*p.name.split(".")[:-1], p]), image_paths),
        key=itemgetter(0),
    )
    name_exp = defaultdict(list)
    for name, exps in groupby(name_to_exposure, key=itemgetter(0)):
        for exp in sorted(exps, key=lambda e: int(e[1])):
            name_exp[name].append(exp)

    return name_exp


@group_ldr_paths.register
def _(path: Path) -> Mapping[str, Collection[Path]]:
    image_paths = path.iterdir()
    return group_ldr_paths(image_paths)


@deprecated(reason="use get_exposures instead, works with dataset folder structure")
def multi_exp_paths(
    raw_paths: Collection[Path], processed_path: Path
) -> Iterator[Collection[Path]]:
    for path in raw_paths:
        yield multi_exp_path(path, processed_path)


@deprecated
def multi_exp_path(raw_path, processed_path):
    name = raw_path.name.split(".")[0]
    exp_paths = list(processed_path.glob(f"{name}*"))
    if len(exp_paths):
        return sorted(exp_paths, key=lambda p: int(p.name.split(".")[1]))


def get_exposures(exp_path: Path, img_path: Path):
    file_name_no_ext = img_path.stem
    exp_paths = exp_path.glob(f"{file_name_no_ext}*")
    return sorted(exp_paths, key=lambda p: int(p.name.split(".")[1]))


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


def read_hdr(path: Path) -> np.ndarray:
    image = ch.Image(str(path))
    return image.read_data()


def gen_multi_exposure(
    image_dict: Mapping[str, np.ndarray]
) -> Mapping[str, Collection[np.ndarray]]:
    multi_exposure: Mapping[str, List] = dict.fromkeys(image_dict, [])

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


def write_multi_exposure(image_list: Collection[Path], out_path: Path):
    for img_path in image_list:
        img = ch.Image(str(img_path))
        img.read_metadata()
        img.read_data()
        for exposure in np.linspace(-4, 4, 5):
            img_name = f"{img_path.name[:-4]}.{int(exposure)}.png"
            out = out_path / img_name
            if not out.exists():
                co.write_image(ch.adjust_exposure(img.data, exposure), str(out))
            yield out


def write_exp_image(image_path: Path, out_path: Path) -> List[Path]:
    exposures = []
    print(image_path)
    img = ch.Image(str(image_path))
    img.read_metadata()
    img.read_data()
    for exposure in np.linspace(-4, 4, 5):
        img_name = f"{image_path.name[:-4]}.{int(exposure)}.png"
        out = out_path / img_name
        if not out.exists():
            co.write_image(
                ch.adjust_exposure(img.data, exposure), str(out),
            )
        exposures.append(out)
    return exposures


def parallel_write_multi_exp(
    image_list: Collection[Path], out_path: Path
) -> Iterator[Path]:
    img_writer = partial(write_exp_image, out_path=out_path)
    with ProcessPoolExecutor(max_workers=25) as executor:
        return chain(*executor.map(img_writer, image_list))


def parallel_cv_fused(fuse_fun: Callable, grouped_paths: Mapping) -> List[Path]:
    with ProcessPoolExecutor(max_workers=25) as executor:
        return list(executor.map(fuse_fun, grouped_paths))


def cv_fuse(
    images: Collection[Path], method: FuseMethod, out_dir: Path
) -> Collection[Path]:
    if method == FuseMethod.All:
        return [
            _method_map[fm](images, out_dir)
            for fm in FuseMethod
            if fm != FuseMethod.All
        ]
    else:
        return [_method_map[method](images, out_dir)]


def merge_all_test(merge_exp_paths):
    merged = []
    for name, exps in merge_exp_paths.items():
        exp_paths = [e[2] for e in exps]
        merged.append(
            fuse_func(exp_paths)
            for fuse_func in (debevec_fuse, robertson_fuse, mertens_fuse)
        )
    return merged


def clip_hdr(fused: np.ndarray) -> np.ndarray:
    return np.clip(fused * 255, 0, 255).astype("uint8")


def _write_tiff_with_metadata(in_file: Path, out_file: Path):
    logging.info(f"Converting {str(in_file)} to Tiff")
    subprocess.call(
        ["dcraw"] + shlex.split('-w -W -H 0 -q 3 -T "{0}"'.format(str(in_file)))
    )

