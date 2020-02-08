from typing import Mapping, Collection
import numpy as np
import colour as co
import colour_hdri as ch
from pathlib import Path

DATA_DIR = Path("../../data/HDR+")


def main():
    pass


def image_fusion(images):
    return


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
            co.io.write_image(
                ch.adjust_exposure(img, exposure),
                str((DATA_DIR / "processed") / f"{img_path.name}.{exposure}"),
            )


if __name__ == "__main__":
    main()
