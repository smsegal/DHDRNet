from collections import defaultdict
from more_itertools import flatten
from functools import partial
from random import sample
from tqdm.contrib.concurrent import thread_map


import pandas as pd
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

from dhdrnet.gen_pairs import GenAllPairs
from dhdrnet.util import DATA_DIR

default_metrics = {
    "psnr": peak_signal_noise_ratio,
    "mse": mean_squared_error,
    "sssim": partial(structural_similarity, multichannel=True),
}


def num_exposures_needed(runs, metrics=default_metrics):
    """
    Generate the approximate number of exposures needed for a low-enough
    reconstruction error when choosing with a given strategy.
    """
    data_gen = GenAllPairs(
        raw_path=DATA_DIR / "dngs",
        out_path=DATA_DIR / "correct_exposures",
        compute_scores=False,
    )
    exposures = list(data_gen.exposures)
    all_image_names = data_gen.image_names

    df = pd.DataFrame(columns=metrics.keys())

    # sample(all_image_names, k=runs):
    stats_fun = partial(
        compute_stats, data_gen=data_gen, exposures=exposures, metrics=metrics
    )
    err_list = flatten(
        thread_map(
            stats_fun,
            all_image_names,
        )
    )

    # for name in all_image_names[:runs]:
    #     ground_truth = data_gen.get_ground_truth(name)
    #     exp_groups = zip(
    #         (select_exposures(exposures, n) for n in exposure_steps), exposure_steps
    #     )
    #     for exp_group, n in exp_groups:
    #         fused = data_gen.get_fused(name, exp_group)
    #         for metric_name, metric_func in metrics.items():
    #             error = metric_func(ground_truth, fused)
    #             err_list.append(
    #                 {
    #                     "image_name": name,
    #                     "metric": metric_name,
    #                     "num_exposures": n,
    #                     "error": error,
    #                 }
    #             )

    df = pd.DataFrame.from_records(err_list, index="image_name")
    return df


def compute_stats(name, data_gen, exposures, metrics):
    exposure_steps = range(1, 7)  # total number of exposures to try
    ground_truth = data_gen.get_ground_truth(name)
    exp_groups = zip(
        (select_exposures(exposures, n) for n in exposure_steps), exposure_steps
    )
    err_list = []
    for exp_group, n in exp_groups:
        fused = data_gen.get_fused(name, exp_group)
        for metric_name, metric_func in metrics.items():
            error = metric_func(ground_truth, fused)
            err_list.append(
                {
                    "image_name": name,
                    "metric": metric_name,
                    "num_exposures": n,
                    "error": error,
                }
            )
    return err_list


def select_exposures(exposures, n=7):
    return sorted([0.0, *sample(exposures, k=n)])
