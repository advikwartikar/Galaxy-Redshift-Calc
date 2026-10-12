# Photometric Redshift Estimation using Transfer Learning and Multi-Modal Deep Learning

A modular BE Computer Engineering project skeleton for estimating galaxy
redshift from SDSS-derived photometric measurements, galaxy images, or both.
It defines three comparable regressors: a photometric MLP, an ImageNet
pretrained ResNet50, and a fused multimodal model. **No dataset, checkpoint,
accuracy claim, or research result is included.**

## Architecture

- `src/data/`: schema validation, CSV loading, strict source-verified exact-ID
  matching, cleaning, train-only feature scaling, image resizing, and splits.
- `src/models/`: Keras MLP, ResNet50 transfer-learning, and multimodal builders.
- `src/training/`: reproducible training entry points and shared data workflow.
- `src/evaluation/`: MAE, RMSE, R², comparison CSV, and diagnostic plots.
- `src/inference/` and `app.py`: checkpoint-backed predictions and Streamlit UI.
- `configs/config.yaml`: data paths, schema, split, training, and checkpoint
  settings.
- `tests/`: small unit tests; neural-network tests use random, uninitialized
  ResNet50 weights and never download pretrained weights.

All three training commands use the same cleaned, exact-ID-matched sample and
the same deterministic split configuration. The feature imputer/scaler is fit
using training records only. Evaluation is on the held-out test subset. The
hybrid model is not assumed to outperform either baseline.

## Prerequisites

- Python 3.11 (recommended; `tensorflow==2.15.1` is pinned in the dependencies).
- Git, pip, and enough disk/RAM for TensorFlow and your chosen data.
- A GPU is optional. CPU training is possible but ResNet50 training can be slow.
- Internet access is needed during dependency installation and may be needed
  when a training run first initializes ResNet50's ImageNet weights. No catalog
  or image dataset is downloaded by this application.

## Setup

Run commands from the project root.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, either use an appropriately configured
execution policy or call the environment executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset preparation and matching

The initial configuration names example paths and assumes photometric feature
columns `u`, `g`, `r`, `i`, `z` and label `redshift`; **these are not
guaranteed to match your selected catalogue**. Check your CSV headers, update
`configs/config.yaml`, and document units, release, selection criteria, and
target-label provenance.

1. Obtain a suitable SDSS photometric catalogue and redshift labels from the
   relevant SDSS data release/catalogue products. Confirm that the redshift
   target and photometric measurements are appropriate for your research.
2. Obtain Galaxy Zoo image metadata and image products from an applicable
   Galaxy Zoo release. Place the prepared catalogue and images under
   `data/raw/`; the metadata CSV must identify each local image path.
3. Do not assume Galaxy Zoo identifiers and SDSS `objid` values refer to the
   same objects merely because their column names look similar. Verify survey,
   release, provenance, identifier semantics, and existing cross-match
   documentation. When IDs are not directly compatible, construct a
   documented coordinate cross-match with stated coordinate system, angular
   tolerance, ambiguity policy, and quality cuts; independently validate it.
4. This skeleton implements **exact one-to-one identifier matching**, not
   astronomical coordinate matching. The matching code deliberately refuses
   to join by default. Set
   `data.source_compatibility_verified: true` only after independently
   verifying that the identifiers are compatible (or after using a vetted
   cross-match catalogue). Training prints matched counts and rates. Inspect
   these diagnostics and the underlying records before training.
5. Update both ID column settings and `data.image_path_column`. Missing images
   are rejected by training. Resolve duplicate identifiers upstream according
   to a documented data-quality policy.

See [data/README.md](data/README.md) for further guidance. Consult the current
official SDSS and Galaxy Zoo release documentation for acquisition, schemas,
licenses, and supported cross-matches. No particular release pairing or
identifier mapping is certified by this code.

## Train the models

After configuring and verifying data, run each command from the project root:

```powershell
python -m src.training.train_photometric --config configs/config.yaml
python -m src.training.train_image --config configs/config.yaml
python -m src.training.train_multimodal --config configs/config.yaml
```

The commands create checkpoints under `models/` and a fitted photometric
imputer/scaler. The image-based commands load image arrays into memory and are
intended as a clear baseline, not a large-scale streaming pipeline. Configure
epochs, batch size, image size, learning rate, and backbone fine-tuning in
`configs/config.yaml`. ResNet50's application preprocessing is applied
consistently by training and inference. ImageNet weights may be fetched by
Keras when training starts and are never fetched just by starting the app or
running tests.

The optional notebook files are starting points, not completed experimental
analyses. Open them from the project root so imports and configured paths
resolve as expected.

## Evaluation

Once all three checkpoints and the scaler exist:

```powershell
python -m src.training.evaluate_models --config configs/config.yaml
```

The evaluator creates `reports/metrics/model_comparison.csv` and places one
actual-versus-predicted plot and one residual plot per model in
`reports/figures/`. MAE and RMSE are
lower-is-better; R² is generally higher-is-better. The output records measured
results only; no performance ordering is assumed. Ensure the test sample and
all three checkpoints correspond to the same dataset/configuration.

## Streamlit application

```powershell
streamlit run app.py
```

The application starts without downloaded data or trained checkpoints. It
explains when the selected checkpoint is missing rather than fabricating a
prediction. Photometric inputs are required for the photometric and hybrid
models; an image is required for image-only and hybrid models. Inputs must
follow the feature units and schema used during training.

## Tests

```powershell
python -m pytest -q
```

Tests use synthetic examples only. The neural model shape test uses
`weights=None`; it does not request ImageNet downloads. It does require a
working TensorFlow installation and may take longer than the data/metric
tests.

## Implemented, scaffolded, and not included

- **Implemented:** CSV schema checks, data cleaning, exact-ID merge guard and
  match report, train-only imputation/scaling, deterministic splitting,
  ResNet50-compatible image preprocessing, all three Keras model builders,
  training entry points, checkpoint/scaler persistence, regression metrics,
  comparison CSV/plots, inference validation, Streamlit interface, and tests.
- **Requires real data / user verification:** catalogue acquisition and
  preparation, scientifically valid SDSS/Galaxy Zoo cross-identification,
  schema and unit configuration, and actual model training/evaluation.
- **Not implemented:** coordinate-based astronomical cross-matching, a scalable
  streaming image pipeline, dataset download automation, hyperparameter
  search, uncertainty estimates, and scientific validation/claims.

## Known limitations and future improvements

The baseline trainer loads image samples into memory. For a production-scale
catalogue, replace it with a performant `tf.data` pipeline and cache only
appropriate derived data. Add explicit group-aware splitting when observations
can share a source, robust astronomical cross-match tooling, quality/outlier
analysis, calibration/uncertainty estimation, stronger experiment tracking,
and independent scientific validation. Keep datasets, trained weights,
credentials, and virtual environments out of version control.
