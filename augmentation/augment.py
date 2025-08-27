from pathlib import Path
import random
from shutil import copy

import albumentations as A
import click
import numpy as np
from PIL import Image


# set seeds for reproducebility
random.seed(42)
np.random.seed(42)

transform = A.Compose([
    A.OneOf([
        A.MotionBlur(blur_limit=11, p=0.2),
        A.MedianBlur(blur_limit=11, p=0.2),
        A.Blur(blur_limit=11, p=0.1),
    ], p=1.0),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.3),
], seed=42)

transform_limits = A.Compose([
    A.OneOf([
        A.MotionBlur(blur_limit=(9, 9), p=0.2),
        A.MedianBlur(blur_limit=(9, 9), p=0.2),
        A.Blur(blur_limit=(9, 9), p=0.1),
    ], p=1.0),
    A.OneOf([
        A.OpticalDistortion(distort_limit=(0.05, 0.05), p=0.3),
        A.ElasticTransform(p=0.1),
    ], p=1.0),
    A.HueSaturationValue(hue_shift_limit=(15, 15), sat_shift_limit=(20, 20), val_shift_limit=(15, 15), p=1.0), 
], seed=42)


@click.command()
@click.argument(
    "images",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, resolve_path=True, path_type=Path),
    required=True,
    nargs=-1
)
@click.option(
    "-o", "--output",
    type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True, path_type=Path),
    required=True
)
@click.option(
    "-n",
    help="Create n random augmentations for each input image",
    type=click.INT,
    default=5,
    show_default=True
)
def main(images: list[Path], output: Path, n: int = 5):
    image_count = len(images)
    for i, image in enumerate(sorted(images), start=1):
        print(f"{i}/{image_count}: {image.as_posix()}")
        
        im = Image.open(image)
        icc_profile = im.info.get("icc_profile")
        im = np.array(im)
        
        for version in range(n):
            target_dir = output.joinpath(f"aug_{version+1}")
            if not target_dir.exists():
                target_dir.mkdir(exist_ok=True)
            transformed = transform(image=im)
            im_trans = Image.fromarray(transformed["image"])
            im_trans.save(target_dir.joinpath(image.name), icc_profile=icc_profile)
            copy(
                image.parent.joinpath(image.name.split('.')[0] + ".xml"), 
                target_dir.joinpath(image.name.split('.')[0] + ".xml")
            )
        

if __name__ == "__main__":
    main()
    