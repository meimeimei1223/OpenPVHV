"""Run OpenPVHV on the paper's evaluation sets (HVM test, VEELA public train set, VEELA hidden test set), one case at a time.

The inference is pvhv_predict.predict() from this repository; this script only finds the input images, calls it per case and keeps a
record. Settings are those in pvhv_predict.SETTINGS (no options here change them). TotalSegmentator crop-model weights are never
downloaded by this script (allow_download=False): run TotalSegmentator once beforehand (`totalseg_download_weights -t total`) or
let pvhv_predict.py download them.

Inputs (not included in this repository; obtain them from the data providers):
  HVM   : the Hepatic Vessel Map dataset (Zenodo 10.5281/zenodo.19885789), folder with "Center 1" and "Center 2".
          The test cases and the case -> image file mapping are in evaluation/hvm_split.json and evaluation/hvm_pairs.json.
  VEELA : the VEELA 2025 challenge data, folder with "VEELA Train Set" and/or "VEELA Test Set" (setNN_norm.nii).

usage:
  python evaluation/predict_eval_sets.py hvm_test    6mm <out dir> --model-dir <OpenPVHV weights folder> --hvm-dir <HVM folder>
  python evaluation/predict_eval_sets.py veela_train 6mm <out dir> --model-dir <...> --veela-dir <VEELA folder>
  python evaluation/predict_eval_sets.py veela_test  6mm <out dir> --model-dir <...> --veela-dir <VEELA folder>
  options: --crop is the 2nd argument (6mm | 1.5mm), --cases c1_007,c1_010 (subset), --device cuda | cpu
Writes <out>/<case>.nii.gz (labels 0 background, 1 portal vein, 2 hepatic vein), <out>/<case>.json (crop, voxel counts, seconds)
and <out>/run.json (settings, versions, hashes). Cases already written are skipped (resume).
"""
import argparse, hashlib, importlib.metadata as md, json, platform, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def inputs(a):
    if a.set == "hvm_test":
        assert a.hvm_dir, "--hvm-dir is needed for hvm_test"
        test = json.load(open(HERE / "hvm_split.json"))["test"]
        pairs = {p["case"]: p for p in json.load(open(HERE / "hvm_pairs.json"))["pairs"]}
        return {c: Path(a.hvm_dir) / pairs[c]["center"] / "Image" / pairs[c]["image"] for c in test}
    assert a.veela_dir, "--veela-dir is needed for the VEELA sets"
    d = Path(a.veela_dir) / ("VEELA Train Set" if a.set == "veela_train" else "VEELA Test Set")
    return {f.name.replace("_norm.nii", ""): f for f in sorted(d.glob("set*_norm.nii"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=["hvm_test", "veela_train", "veela_test"]); ap.add_argument("crop", choices=["6mm", "1.5mm"])
    ap.add_argument("out"); ap.add_argument("--model-dir", required=True)
    ap.add_argument("--hvm-dir"); ap.add_argument("--veela-dir")
    ap.add_argument("--cases", default=""); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    sys.path.insert(0, str(REPO))
    import pvhv_predict as rel
    import nibabel as nib
    import torch

    src = inputs(a)
    cases = sorted(src) if a.set != "hvm_test" else list(src)
    if a.cases:
        cases = [c for c in a.cases.split(",") if c in src]
    assert cases and all(src[c].exists() for c in cases), "input images missing"
    model_dir = Path(a.model_dir)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    run = {"set": a.set, "crop": a.crop, "settings": rel.SETTINGS, "device": a.device, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "release_code_sha256": sha(REPO / "pvhv_predict.py"),
           "checkpoint_sha256": sha(model_dir / "fold_all" / rel.SETTINGS["checkpoint"]),
           "versions": {k: md.version(k) for k in ("TotalSegmentator", "nnunetv2", "torch", "numpy", "nibabel", "SimpleITK")},
           "python": sys.version.split()[0], "platform": platform.platform(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "cases": cases}
    json.dump(run, open(out / "run.json", "w"), indent=1)

    for c in cases:
        fo = out / f"{c}.nii.gz"
        if fo.exists() and (out / f"{c}.json").exists():
            continue
        t0 = time.time()
        img = nib.load(str(src[c]))
        lab, info = rel.predict(img, model_dir, crop=a.crop, device=a.device, allow_download=False, quiet=True)
        nib.save(lab, str(fo))
        rec = {"case": c, "input": src[c].name, "shape": list(img.shape[:3]), "zooms": [float(z) for z in img.header.get_zooms()[:3]],
               **info, "seconds_total": round(time.time() - t0, 2)}
        json.dump(rec, open(out / f"{c}.json", "w"), indent=1)
        print(json.dumps({k: rec[k] for k in ("case", "liver_ml_crop_model", "empty_crop", "pv_vox", "hv_vox", "seconds_total")}), flush=True)

    run["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(run, open(out / "run.json", "w"), indent=1)
    print(f"done: {len(cases)} cases -> {out}")


if __name__ == "__main__":   # nnU-Net workers use spawn on Windows: the module must not run on import
    main()
