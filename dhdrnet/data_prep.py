import sys
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Iterable, List, Optional, Union

import fire
from plumbum import ProcessExecutionError, local
from rich import print
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from tqdm.contrib.concurrent import thread_map
from tqdm.std import tqdm

from dhdrnet.data_generator import DataGenerator
from dhdrnet.dataset import CachingDataset

"""
Example URL:
https://storage.cloud.google.com/hdrplusdata/20171106/results_20171023/0006_20160721_181503_256/merged.dng
This is downloaded as data/0006_20160721_181503_256.dng
"""

hdrplus_bucket_base = "hdrplusdata/20171106/results_20171023"
default_download_dir = Path("./data")

gsutil = local["gsutil"]  # gsutil command
gslist = gsutil["ls"]  # gsutil ls
gsdl = gsutil["cp"]  # gsutil cp


def get_image_names(url: str) -> Iterable[str]:
    file_listing = (f.split("/") for f in gsutil["ls"](f"gs://{url}").split("\n"))
    image_names = (f[-2] for f in file_listing if len(f) > 1)
    return image_names


def download_file(fname: str, out_path: Path) -> Optional[Path]:
    dng_downloader = gsdl[f"gs://{hdrplus_bucket_base}/{fname}/merged.dng"]
    try:
        dng_downloader(f"{out_path}/{fname}.dng")
        dng_path = Path(out_path / f"{fname}.dng")
        if dng_path.exists():
            return dng_path
        else:
            return None
    except ProcessExecutionError as err:
        print(
            f"[bold red]Error with gsutil. Inner error:[/bold red] \n{err}",
            file=sys.stderr,
        )
        return None


def download(
    base_url: str = hdrplus_bucket_base,
    out: Union[str, Path] = default_download_dir,
    image_names: Optional[List[str]] = None,
    max_threads=10,
) -> List[Optional[Path]]:

    if image_names is None:
        image_names = list(get_image_names(base_url))

    out = coerce_path(out)
    if not out.exists():
        out.mkdir(parents=True)

    downloaded_files = thread_map(
        partial(download_file, out_path=out),
        image_names,
        max_workers=max_threads,
        chunksize=min(50, len(image_names)),
    )

    return downloaded_files


def coerce_path(maybe_path: Union[Path, str]) -> Path:
    if isinstance(maybe_path, str):
        return Path(maybe_path)
    else:
        return maybe_path


def gen_ev_pair_data(data_dir):
    exposure_vals = [-5, -3, -1, 0, 1, 3, 5]
    ds = CachingDataset(
        data_dir=Path(data_dir),
        exposure_values=exposure_vals,
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
            ]
        ),
    )
    dl = DataLoader(ds, batch_size=10, num_workers=cpu_count())
    # ev_pairs_dict = {v: k for (k, v) in evpairs_to_classes(exposure_vals).items()}
    for batch in tqdm(dl):
        images, ev_classes, scores = batch
        # ev_pairs = [ev_pairs_dict[ev_class.item()] for ev_class in ev_classes]
        # print(f"{images.shape=}")
        # print(f"{ev_pairs=}")
        # print(f"{scores=}")


def generate_data(
    download_dir: Union[str, Path] = default_download_dir,
    out: Union[str, Path] = Path("./generated_data"),
    multithreaded: bool = True,
    compute_stats: bool = True,
    stats_path: Union[str, Path] = Path("./generated_statistics.csv"),
    min_exposure: float = -6,
    max_exposure: float = 6,
    exposure_step: float = 0.25,
):

    download_dir = coerce_path(download_dir)
    out = coerce_path(out)
    stats_path = coerce_path(stats_path)

    generator = DataGenerator(
        raw_path=download_dir,
        out_path=out,
        multithreaded=multithreaded,
        exp_min=min_exposure,
        exp_max=max_exposure,
        exp_step=exposure_step,
        compute_scores=compute_stats,
        store_path=stats_path,
        image_names=[d.stem for d in download_dir.iterdir()],
    )
    generator()


if __name__ == "__main__":
    fire.Fire(
        {
            "download": download,
            "generate-data": generate_data,
            "generate-ev-data": gen_ev_pair_data,
        }
    )
