import sys
from functools import partial
from pathlib import Path
from typing import Iterable, List, Optional

from rich import print

import fire
from plumbum import ProcessExecutionError, local
from tqdm.contrib.concurrent import thread_map

"""
Example URL:
https://storage.cloud.google.com/hdrplusdata/20171106/results_20171023/0006_20160721_181503_256/merged.dng
"""

hdrplus_bucket_base = "hdrplusdata/20171106/results_20171023"
hdrp_bucket_name = "hdrplusdata"
hdrp_pre = "20171106"
default_download_dir = Path("./data")

gsutil = local["gsutil"]
gslist = gsutil["ls"]
gsdl = gsutil["cp"]


def get_image_names(url: str) -> Iterable[str]:
    file_listing = (f.split("/") for f in gsutil["ls"](f"gs://{url}").split("\n"))
    image_names = (f[-2] for f in file_listing if len(f) > 1)
    return image_names


def down_file(fname: str, out_path: Path) -> Optional[Path]:
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
    out: Path = default_download_dir,
    image_names: Optional[List[str]] = None,
    max_threads=10,
) -> List[Optional[Path]]:

    if image_names is None:
        image_names = list(get_image_names(base_url))

    if not isinstance(out, Path):
        out = Path(out)
    if not out.exists():
        out.mkdir(parents=True)

    downloaded_files = thread_map(
        partial(down_file, out_path=out),
        image_names,
        max_workers=max_threads,
        chunksize=min(50, len(image_names)),
    )

    return downloaded_files


if __name__ == "__main__":
    fire.Fire()
