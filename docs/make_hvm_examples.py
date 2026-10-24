"""Make docs/hvm_examples.png: coronal projections of the reference, TotalSegmentator liver_vessels and OpenPVHV on three held-out
HVM test cases (75th / 50th / 25th percentile of clDice in the paper). Masks only (HVM, CC BY 4.0).
usage: python make_hvm_examples.py <ref dir: <case>_gt.nii.gz, 1 PV / 2 HV> <OpenPVHV dir: <case>.nii.gz>
                                   <TS dir: <case>_liver_vessels.nii.gz> <out.png> [<per-case csv with clDice>]
"""
import csv, sys
import numpy as np, nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CASES = ["c1_096", "c2_159", "c1_069"]
PV, HV, TS = np.array([0.15, 0.45, 0.95]), np.array([0.90, 0.25, 0.25]), np.array([0.35, 0.35, 0.35])


def load(f):
    return nib.as_closest_canonical(nib.load(f))


def coronal(m):
    """project along anterior-posterior; superior up, patient right on the image left"""
    return np.flipud(np.flipud(m.any(1)).T)


def main():
    ref, pred, ts, out = sys.argv[1:5]
    cl = {r["case"]: r for r in csv.DictReader(open(sys.argv[5]))} if len(sys.argv) > 5 else {}
    fig, ax = plt.subplots(3, 3, figsize=(10.5, 10.5), facecolor="white")
    for i, c in enumerate(CASES):
        G = load(f"{ref}/{c}_gt.nii.gz"); g = np.asarray(G.dataobj)
        p = np.asarray(load(f"{pred}/{c}.nii.gz").dataobj)
        t = np.asarray(load(f"{ts}/{c}_liver_vessels.nii.gz").dataobj) == 1
        sp = G.header.get_zooms()[:3]
        anyz = lambda m: m.any((0, 1)); anyx = lambda m: m.any((1, 2))
        nz = np.where(anyz(g > 0) | anyz(p > 0) | anyz(t))[0]; nx = np.where(anyx(g > 0) | anyx(p > 0) | anyx(t))[0]
        sl = (slice(max(nx.min() - 5, 0), nx.max() + 5), slice(None), slice(max(nz.min() - 5, 0), nz.max() + 5))
        title = "OpenPVHV"
        if c in cl:
            title += f"  (clDice PV {float(cl[c]['openpvhv__official_portal_clDice']):.2f} / HV {float(cl[c]['openpvhv__official_hepatic_clDice']):.2f})"
        views = [("Reference (HVM)", [((g == 2)[sl], HV), ((g == 1)[sl], PV)]),
                 ("TotalSegmentator liver_vessels\n(one class)", [(t[sl], TS)]),
                 (title, [((p == 2)[sl], HV), ((p == 1)[sl], PV)])]
        for j, (tt, layers) in enumerate(views):
            L = [(coronal(m), col) for m, col in layers]
            img = np.ones(L[0][0].shape + (3,))
            for m, col in L:
                img[m] = col
            a = ax[i, j]; a.imshow(img, aspect=sp[2] / sp[0], interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            a.set_title(tt if i == 0 or j == 2 else "", fontsize=10)
            if j == 0:
                a.set_ylabel(f"HVM test {c}", fontsize=10)
    fig.legend(handles=[Patch(color=PV, label="portal vein"), Patch(color=HV, label="hepatic veins"),
                        Patch(color=TS, label="liver vessels (one class)")], loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.text(0.5, 0.005, "Coronal projections, held-out HVM test cases (75th / 50th / 25th percentile of clDice). HVM data: Xie T, "
             "Zenodo 2025, doi:10.5281/zenodo.19885789, CC BY 4.0.", ha="center", fontsize=7.5, color="0.3")
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(out, dpi=110)


if __name__ == "__main__":
    main()
