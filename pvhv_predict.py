"""OpenPVHV: portal vein / hepatic vein segmentation in CT (nnU-Net v2, trained on public data (HVM) only).

Input : a CT volume (NIfTI). Output: a label map on the input grid (uint8): 0 = background, 1 = portal vein, 2 = hepatic vein
        (intrahepatic veins; the inferior vena cava is not segmented).

The pipeline is TotalSegmentator 2.18's liver_vessels pipeline with the vessel model replaced; it calls TotalSegmentator's own
functions with the arguments TotalSegmentator uses for that task:
  1. liver for the crop: TotalSegmentator "total" 6 mm model (task 298; default) or, with --crop 1.5mm, the 1.5 mm organs model
     (task 291), no test-time augmentation; liver = label 5
  2. crop to the liver bounding box + 20 mm on each axis
  3. this model (Dataset950_HVM_PVHV, nnUNetTrainer__nnUNetPlansScratch__3d_fullres, fold "all", checkpoint_final.pth):
     sliding window step 0.5, Gaussian weighting, no mirroring, no postprocessing; back to the input grid.
If no liver is found in step 1, the output is empty (as in TotalSegmentator).

usage:
  python pvhv_predict.py -i ct.nii.gz -o pv_hv.nii.gz -m <...>/Dataset950_HVM_PVHV/nnUNetTrainer__nnUNetPlansScratch__3d_fullres
                         [--crop 6mm|1.5mm] [--device cuda|cpu|mps]
Requires TotalSegmentator 2.18.0 and nnunetv2 2.7.0. The TotalSegmentator crop model is downloaded by TotalSegmentator on
first use if it is not present (pass --no-download to fail instead).
Research use only. Not a medical device; not for diagnosis or treatment decisions.
"""
import argparse, json, sys, time
from pathlib import Path

SETTINGS = dict(
    crop_addon_mm=[20, 20, 20], step_size=0.5, tta=False, folds=["all"], checkpoint="checkpoint_final.pth",
    resample=None, remove_small_blobs=False, nr_thr_resamp=1, nr_thr_saving=6, liver_label=5,
    crop_models={"6mm": dict(task_id=298, trainer="nnUNetTrainer_4000epochs_NoMirroring", resample=6.0),
                 "1.5mm": dict(task_id=291, trainer="nnUNetTrainerNoMirroring", resample=1.5)},
)
LABELS = {0: "background", 1: "portal_vein", 2: "hepatic_vein"}
TASK_ID = 950
_tsn = None


def _setup(model_dir):
    """Import TotalSegmentator with its nnU-Net paths and point its model lookup for task 950 at model_dir."""
    global _tsn
    model_dir = Path(model_dir).resolve()
    assert (model_dir / "plans.json").exists() and (model_dir / "fold_all" / SETTINGS["checkpoint"]).exists(), \
        f"not an nnU-Net model folder with fold_all/{SETTINGS['checkpoint']}: {model_dir}"
    if _tsn is None:
        from totalsegmentator.config import setup_nnunet
        setup_nnunet()  # nnUNet_results -> the TotalSegmentator weights dir; must run before nnunetv2.paths is imported
        import totalsegmentator.nnunet as tsn
        from totalsegmentator.map_to_binary import class_map
        class_map["pvhv_950"] = {k: v for k, v in LABELS.items() if k}
        tsn._pvhv_orig_get_output_folder = tsn.get_output_folder
        _tsn = tsn
    orig = _tsn._pvhv_orig_get_output_folder
    _tsn.get_output_folder = lambda tid, tr, pl, mo: str(model_dir) if tid == TASK_ID else orig(tid, tr, pl, mo)
    return _tsn


def predict(img, model_dir, crop="6mm", device="cuda", allow_download=True, quiet=True):
    """img: nibabel Nifti1Image (CT, HU). Returns (label Nifti1Image on the input grid, info dict)."""
    import numpy as np
    import nibabel as nib
    tsn = _setup(model_dir)
    cm = SETTINGS["crop_models"][crop]
    crop_dir = Path(tsn.get_output_folder(cm["task_id"], cm["trainer"], "nnUNetPlans", "3d_fullres"))
    if not crop_dir.exists():
        if not allow_download:
            raise FileNotFoundError(f"TotalSegmentator crop model missing: {crop_dir}")
        from totalsegmentator.libs import download_pretrained_weights
        download_pretrained_weights(cm["task_id"])
    t0 = time.time()
    organ, _, _ = tsn.nnUNet_predict_image(img, None, cm["task_id"], model="3d_fullres", folds=[0], trainer=cm["trainer"],
                                           tta=False, multilabel_image=True, resample=cm["resample"], crop=None, crop_path=None,
                                           task_name="total", nora_tag="None", preview=False, save_binary=False,
                                           nr_threads_resampling=SETTINGS["nr_thr_resamp"], nr_threads_saving=1, crop_addon=None,
                                           output_type="nifti", statistics=False, quiet=quiet, verbose=False, test=0,
                                           skip_saving=False, device=device)
    t1 = time.time()
    liver = (np.asarray(organ.dataobj) == SETTINGS["liver_label"]).astype(np.uint8)
    if not liver.any():
        # no liver found: empty output, like the TotalSegmentator CLI. (TS 2.18 called in memory, file_out=None, would
        # treat an empty crop mask as no crop and run model 950 on the whole image.)
        print(f"WARNING: the {crop} crop model found no liver; returning an empty segmentation", file=sys.stderr)
        out = nib.Nifti1Image(np.zeros(img.shape[:3], np.uint8), img.affine, img.header.copy()); out.set_data_dtype(np.uint8)
        return out, {"crop": crop, "liver_voxels_crop_model": 0, "liver_ml_crop_model": 0.0, "liver_bbox_crop_grid": None,
                     "empty_crop": True, "pv_vox": 0, "hv_vox": 0, "labels": [0], "seconds_crop": round(t1 - t0, 2),
                     "seconds_model": 0.0}
    seg, _, _ = tsn.nnUNet_predict_image(img, None, TASK_ID, model="3d_fullres", folds=SETTINGS["folds"], trainer="nnUNetTrainer",
                                         tta=SETTINGS["tta"], multilabel_image=True, resample=SETTINGS["resample"],
                                         crop=nib.Nifti1Image(liver, organ.affine), crop_path=None, task_name="pvhv_950",
                                         nora_tag="None", preview=False, nr_threads_resampling=SETTINGS["nr_thr_resamp"],
                                         nr_threads_saving=SETTINGS["nr_thr_saving"], crop_addon=SETTINGS["crop_addon_mm"],
                                         output_type="nifti", statistics=False, quiet=quiet, verbose=False, test=0,
                                         skip_saving=False, device=device, remove_small_blobs=SETTINGS["remove_small_blobs"],
                                         plans="nnUNetPlansScratch")
    t2 = time.time()
    lab = np.asarray(seg.dataobj).astype(np.uint8)
    out = nib.Nifti1Image(lab, img.affine, img.header.copy()); out.set_data_dtype(np.uint8)
    nz = np.argwhere(liver)
    info = {"crop": crop, "liver_voxels_crop_model": int(liver.sum()),
            "liver_ml_crop_model": float(liver.sum() * np.prod(organ.header.get_zooms()[:3]) / 1000),
            "liver_bbox_crop_grid": [nz.min(0).tolist(), nz.max(0).tolist()] if len(nz) else None,
            "empty_crop": bool(liver.sum() == 0), "pv_vox": int((lab == 1).sum()), "hv_vox": int((lab == 2).sum()),
            "labels": np.unique(lab).tolist(), "seconds_crop": round(t1 - t0, 2), "seconds_model": round(t2 - t1, 2)}
    return out, info


def main():
    ap = argparse.ArgumentParser(description="Portal vein (1) / hepatic vein (2) segmentation in CT. Research use only.")
    ap.add_argument("-i", "--input", required=True, help="CT NIfTI (.nii or .nii.gz)")
    ap.add_argument("-o", "--output", required=True, help="output label map (.nii.gz)")
    ap.add_argument("-m", "--model-dir", required=True, help="…/Dataset950_HVM_PVHV/nnUNetTrainer__nnUNetPlansScratch__3d_fullres")
    ap.add_argument("--crop", choices=["6mm", "1.5mm"], default="6mm", help="model that finds the liver for the crop (default 6mm)")
    ap.add_argument("--device", default="cuda", help="cuda | cpu | mps")
    ap.add_argument("--no-download", action="store_true", help="do not download a missing TotalSegmentator crop model")
    ap.add_argument("--info", help="optional JSON file for crop / volume / timing information")
    a = ap.parse_args()
    import nibabel as nib
    out, info = predict(nib.load(a.input), a.model_dir, a.crop, a.device, allow_download=not a.no_download, quiet=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, a.output)
    if a.info:
        json.dump(info, open(a.info, "w"), indent=1)
    if info["empty_crop"]:
        print("WARNING: no liver found by the crop model; the output is empty.", file=sys.stderr)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
