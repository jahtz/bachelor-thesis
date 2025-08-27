from collections import defaultdict
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Literal

import click
import cv2
import numpy as np
from PIL import Image
from pypxml import PageXML
from sklearn.metrics import jaccard_score


@dataclass()
class Result:
    """ Store result metrics """
    tPA: float  # total pixel accuracy
    tPA_nobg: float  # total pixel accuracy excluding background
    mIoU: float  # mean intersection over union
    mIoU_nobg: float  # mean intersection over union excluding background
    fwIoU: float  # frequency weighted intersection over union
    fwIoU_nobg: float  # frequency weighted intersection over union excluding background
    fgPA: float  # foreground pixel accuracy
    

class LayoutEvaluation:
    def __init__(
        self,
        ground_truth: PageXML,
        prediction: PageXML,
        normalize: bool = False,
        subtypes: bool = False
    ) -> None:
        self.width, self.height = ground_truth.imageWidth, ground_truth.imageHeight
        
        self.classes = {}
        self.gt_mask: np.ndarray = self.__parse_page(ground_truth, normalize, subtypes)
        self.pred_mask: np.ndarray = self.__parse_page(prediction, normalize, subtypes)
        assert self.gt_mask.shape == self.pred_mask.shape
        
    def __parse_page(self, pagexml: PageXML, normalize: bool = False, subtypes: bool = False) -> np.ndarray:
        polygons = defaultdict(list)
        for region in pagexml.regions:
            if (coords := region.find_coords()) is not None:
                if normalize:
                    cl = "content"
                elif subtypes:
                    cl = f"{region.pagetype.value}.{region['type']}" if 'type' in region else region.pagetype.value
                else:
                    cl = region.pagetype.value
                if cl not in self.classes:
                    self.classes[cl] = len(self.classes.keys()) + 1
                polygon = np.array([tuple(map(int, xy.split(','))) for xy in coords["points"].split()], dtype=np.int32)
                polygons[cl].append(polygon)
            else:
                print(f"No Coords element in region {region['id']}")

        label_mask = np.zeros((self.height, self.width), dtype=np.uint8)  # 0 = background
        for cl, polys in polygons.items():
            cl = self.classes[cl]
            for poly in polys:
                cv2.fillPoly(label_mask, [poly], color=cl)

        return label_mask
    
    def tPA(self, ignore_bg: bool = False) -> float:
        """
        Total Pixel Accuracy

        TPA = sum_x(c_x) / sum_x(1)
        c_x: 1 if pixel x was classified correctly, else 0,
        sum_x(1): total number of pixels
        """
        if ignore_bg:
            foreground = self.gt_mask != 0
            total = np.sum(foreground)
            
            # Edge case: no foreground in ground truth (blank page)
            if total == 0:
                if np.all(self.pred_mask == 0):
                    return 1.0
                else:
                    return 0.0
            
            correct = (self.gt_mask == self.pred_mask) & foreground
        
        else:
            correct = self.gt_mask == self.pred_mask
            total = self.gt_mask.size

        return np.sum(correct) / total if total > 0 else 0.0
    
    def fgPA(self, image: np.ndarray) -> float:
        """
        Foreground Pixel Accuracy (FgPA).
        
        FgPA = sum_x(b_x * c_x) / sum_x(b_x)
        x is a Pixel, 
        b_x: 1 if x is a foreground (value > 0) pixel else 0 ,
        c_x: 1 if the pixel x was correctly classified else 0.
        
        For Foreground Pixel Error (FgPE) calculate FgPE = 1 - FgPA
        
        Requires a binary image input
        """
        foreground = image == 0
        correct = (self.gt_mask == self.pred_mask) & foreground
        total = np.sum(foreground)
        return float(np.sum(correct) / total) if total > 0 else 0.0
    
    def IoU(self, mode: Literal["macro", "weighted"], ignore_bg: bool = False) -> float:
        """
        Intersection over Union
        
        Per class IoU: 
        IoU_c = TP_c / (TP_c + FP_c + FN_c)
        TP_c: True Positive for class c,
        FP_c: False Positive for class c,
        FN_c: False Negative for class c
        
        Macro/Mean/Unweighted IoU (mode = macro): 
        mIoU = (1/C) * sum^C_(c=1)(IoU_c)
        C: Total number of classes
        
        Frequency Weighted IoU (mode = weighted):
        fwIoU = sum^C_(c=1)((n_c / N) * IoU_c)
        n_c: Number of pixels in class c
        N: sum^C_(c=1)(n_c) (Total number of pixels)
        """
        gt_mask = self.gt_mask.flatten()
        pred_mask = self.pred_mask.flatten()
        if ignore_bg:
            labels = np.unique(np.concatenate([gt_mask, pred_mask]))
            labels = labels[labels != 0]
            
            # Edge case: no foreground in ground truth (blank page)
            if len(labels) == 0:
                if np.all(pred_mask == 0):
                    return 1.0
                else:
                    return 0.0
                
            return jaccard_score(gt_mask, pred_mask, labels=labels, average=mode, zero_division=0.0)
        
        else:
            return jaccard_score(gt_mask, pred_mask, average=mode, zero_division=0.0)


@click.command()
@click.argument(
    "gt_directory",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, resolve_path=True, path_type=Path),
    required=True
)
@click.argument(
    "pred_directory",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, resolve_path=True, path_type=Path),
    required=True
)
@click.option(
    "--pred-glob",
    type=click.STRING,
    default="*.xml",
    show_default=True
)
@click.option(
    "--image-ext",
    help="Specify the full extension of binary images located in gt_directory",
    type=click.STRING,
    default=".bin.png",
    show_default=True
)
@click.option(
    "-o", "--output", "output",
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    help="Output results to a JSON file.",
    metavar="JSON FILE"
)
@click.option(
    "-s", "--subtypes",
    type=click.BOOL,
    help="Split PageTypes (e.g. TextRegion) into its subclasses (e.g. paragraph, marginalia,...).",
    is_flag=True
)
def evaluate(gt_directory: Path, pred_directory: Path, pred_glob: str = "*.xml",
             image_ext: str = ".bin.png", output: Path | None = None, subtypes: bool = False) -> None:
    results: dict[str, Result] = {}  # {filename: Result(), ...}

    pred_files = sorted(pred_directory.glob(pred_glob))
    pred_filecount = len(pred_files)
    
    evaluated = 0
    for i, pred_fp in enumerate(pred_files, start=1):
        print(f"{i}/{pred_filecount} {pred_fp.stem}")
        
        gt_fp = gt_directory.joinpath(pred_fp.name.split('.')[0] + ".xml")
        image_fp = gt_directory.joinpath(pred_fp.name.split('.')[0] + image_ext)
        if not gt_fp.exists():
            print("GT file not found: ", gt_fp)
            continue
        if not image_fp.exists():
            print("Image file not found: ", image_fp)
            continue
            
        eval = LayoutEvaluation(
            ground_truth=PageXML.from_file(gt_fp),
            prediction=PageXML.from_file(pred_fp),
            subtypes=subtypes
        )
        res = Result(
            tPA=eval.tPA(),
            tPA_nobg=eval.tPA(ignore_bg=True),
            mIoU=eval.IoU(mode="macro"),
            mIoU_nobg=eval.IoU(mode="macro", ignore_bg=True),
            fwIoU=eval.IoU(mode="weighted"),
            fwIoU_nobg=eval.IoU(mode="weighted", ignore_bg=True),
            fgPA=eval.fgPA(image=np.array(Image.open(image_fp).convert("1")))
        )
        for key, value in asdict(res).items():
            print(f"{key:<10}: {round(value, 4)}".rstrip())
        print()
        results[pred_fp.stem] = res
        evaluated += 1
        
    total = Result(
        tPA=sum(r.tPA for _, r in results.items()) / evaluated,
        tPA_nobg=sum(r.tPA_nobg for _, r in results.items()) / evaluated,
        mIoU=sum(r.mIoU for _, r in results.items()) / evaluated,
        mIoU_nobg=sum(r.mIoU_nobg for _, r in results.items()) / evaluated,
        fwIoU=sum(r.fwIoU for _, r in results.items()) / evaluated,
        fwIoU_nobg=sum(r.fwIoU_nobg for _, r in results.items()) / evaluated,
        fgPA=sum(r.fgPA for _, r in results.items()) / evaluated
    )

    print("\nTotal:")
    for key, value in asdict(total).items():
            print(f"{key:<10}: {round(value, 4)}".rstrip())

    results["Total"] = total

    if output:
        with open(output, 'w') as f:
            json.dump({filename: asdict(result) for filename, result in results.items()}, f)
            

if __name__ == "__main__":
    evaluate()
