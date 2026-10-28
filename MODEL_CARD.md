# Model card: OpenPVHV (nnU-Net Dataset950_HVM_PVHV)

**OpenPVHV is an add-on task for [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) (v2.18) that replaces the
single-class `liver_vessels` output with separate portal vein and hepatic vein labels.** The vessel model was trained only on
HVM; the liver crop uses TotalSegmentator's `total` model (6 mm, Apache-2.0), exactly as TotalSegmentator's `liver_vessels`
task does. It is a drop-in replacement for TotalSegmentator's `liver_vessels` task, developed independently of the
TotalSegmentator authors (not an official TotalSegmentator task).

## Model
| | |
|---|---|
| Task | Semantic segmentation of the intrahepatic portal vein (1) and hepatic veins (2) in CT |
| Architecture | nnU-Net v2 3d_fullres, PlainConvUNet, 6 stages (32–320 channels), instance norm, leaky ReLU |
| Input | One CT channel (HU). Resampled by nnU-Net to 1.5 × 0.799 × 0.799 mm; CT normalisation clip [8, 268] HU, mean 128.7, SD 54.6 |
| Patch / batch | 64 × 192 × 192, batch size 2 |
| Initialisation | Random (no pretrained weights) |
| Training | 1000 epochs, fold `all` (224 cases), nnU-Net defaults (SGD with Nesterov momentum, poly learning rate from 0.01, Dice + cross-entropy, deep supervision, default augmentation) |
| Checkpoint | `checkpoint_final.pth` (epoch 1000), SHA-256 `ad977a08fde429a337b22cdfd54eb7bce87cf66c4e146688b4909740128f9ef5` |
| Software | nnunetv2 2.7.0, PyTorch 2.11.0 (CUDA 12.6), TotalSegmentator 2.18.0 |
| Hardware | 1 × NVIDIA GeForce RTX 4070 (12 GB), about 25 h |
| License | Apache License 2.0 (weights and code) |
| Author | Meidai Kasai (Independent Researcher) |
| Weights | Zenodo, https://doi.org/10.5281/zenodo.23619753 (`OpenPVHV_weights.zip`, SHA-256 `c775a4d054c553d0a40d3c3fff05942dc24561ac5698e8263c885cd31a914b1b`) |
| Paper | Separating portal and hepatic veins on CT with a TotalSegmentator-compatible model trained only on public data: external validation and release. Preprint, Research Square, 2026, https://doi.org/10.21203/rs.3.rs-11500369/v1 |

## Intended use
- Research and education: obtaining separate portal and hepatic venous trees from contrast-enhanced abdominal CT, for example as
  input to research tools for vascular territories or surgical-planning prototypes.
- **Not a medical device.** Not for diagnosis, treatment decisions or clinical surgical planning. Outputs must be reviewed by a
  qualified person before any use beyond research.

## Training data
- Hepatic Vessel Map (HVM). Paper: Xie T, et al. Sci Data. 2026;13:1259, https://doi.org/10.1038/s41597-026-07550-3.
  Data: Xie T. Hepatic Vessel Map (HVM) [Dataset]. Zenodo; 2025, https://doi.org/10.5281/zenodo.19885789. License CC BY 4.0.
- Two hospitals, portal venous phase CT, expert annotations of portal and hepatic veins.
- 280 cases paired one-to-one between annotation and image files (by NIfTI header and CT values inside the vessels); 2 excluded.
- Split: 224 training / 56 held-out test (20 % of each centre, seed 20261011). The test cases were not used for training or
  for any choice of settings.
- Slice thickness of the used cases: 0.7–1.35 mm. The distributed HVM images are intensity-clipped
  (Center 1 about −40 to 160 HU; Center 2 mostly −120 to 180 HU).
- No case overlap was found between HVM and MSD Task08 (the training data of TotalSegmentator's liver_vessels model).

## Inference settings (fixed before evaluation)
Liver crop by TotalSegmentator task 298 (6 mm) → bounding box + 20 mm → this model, step 0.5, Gaussian weighting, no mirroring,
no postprocessing. An empty liver crop gives an empty output.

## Evaluation
Metrics: VEELA 2025 challenge evaluation code (clDice, IoU, NSD; spacing (3, 2, 1) mm, NSD tolerance 1 mm, as fixed in that code),
the same code for every data set. HVM and VEELA public: prediction and reference restricted to the crop model's liver mask grown
by 10 mm; VEELA hidden: the organisers' evaluation of the submitted outputs.
An empty output (one HVM case, see Limitations) is scored 0 and included.

| Data | Metric | Portal vein | Hepatic vein |
|---|---|---|---|
| HVM held-out test (56) | mean clDice | 0.750 | 0.786 |
| VEELA public training set (20, external) | mean clDice | 0.723 | 0.662 |
| VEELA hidden test set (20, external, challenge evaluation) | mean clDice | 0.734 | 0.732 |

IoU, NSD, comparisons and statistics: see the paper.

## Limitations
- **Inferior vena cava is not segmented.** HVM has no IVC label; the hepatic veins end at the liver. Datasets that count the
  intrahepatic IVC as hepatic vein will show lower hepatic-vein overlap.
- **Liver crop failure.** The crop depends on TotalSegmentator's 6 mm model. In one held-out HVM case with severe fatty liver
  (liver about 23 HU) and intensity-clipped images it found no liver, and the output was empty. `--crop 1.5mm` uses the 1.5 mm
  model instead; the default stays 6 mm. The evaluation in the paper uses 6 mm.
- Thin peripheral branches may be missed.
- Trained on portal venous phase CT from two centres in one country; other phases (arterial, delayed, CTA), scanners, very thick
  slices, paediatric patients and post-surgical livers were not evaluated.
- Separation is learned from image appearance and context; anatomical variants may be mislabelled; this was not evaluated separately.

## Ethical considerations
Trained only on a public, de-identified dataset released under CC BY 4.0. No patient data are distributed with the model.
