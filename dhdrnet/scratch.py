#!/usr/bin/env python3
from dhdrnet.vis_util import show_image_pair

for pred, img_name in zip(predictions, image_names[0]):
    ev = ev_options[pred]
    pred_img = Image.open(
        DATA_DIR / "correct_exposures" / "exposures" / f"{img_name}[{ev}].png"
    )
    baseline_img = Image.open(
        DATA_DIR / "correct_exposures" / "exposures" / f"{img_name}[0.0].png"
    )
    show_image_pair(baseline_img, pred_img, title=f"{img_name} (Input + Predicted)",
                   labels=["Input", "Predicted"])


    lower = np.minimum(ev, 0.0)
    upper = np.maximum(ev, 0.0)
    reconstructed = Image.open(
        DATA_DIR
        / "correct_exposures"
        / "reconstructions"
        / f"{img_name}[{lower}][{upper}].png"
    )
    ground_truth = Image.open(
        DATA_DIR / "correct_exposures" / "ground_truth" / f"{img_name}.png"
    )
    show_image_pair(
        ground_truth, reconstructed, title=f"{img_name} (Ground Truth + Reconstructed)", labels=["Ground Truth", "Reconstructed"]
    )
