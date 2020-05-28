from collections import defaultdict
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

DATA_DIR = Path("../data").resolve()

co.utilities.filter_warnings()


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


def gen_multi_exposures(
        img_path: Path, *exposure_values
) -> Collection[np.ndarray]:
    img_loaded = co.read_image(img_path)
    exposures = [
        ch.adjust_exposure(img_loaded, exposure)
        for exposure in exposure_values
    ]
    return exposures


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
