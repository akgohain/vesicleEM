# Vesicle Classification & Proofreading Pipeline

This repository contains **Tim Fan’s vesicle‑classification and visualization pipeline**. It lets you

1. **Train** a balanced 3‑class CNN on all available, labelled cells.
2. **Predict & visualise** vesicle classes for new cells.
3. **Evaluate** a saved model on one or many datasets.

> The code was last refactored for a tidy command‑line interface (CLI) with three mutually‑exclusive modes (`train`, `predict`, `eval`).

---

## 1  Quick start

```bash
# 1️⃣  Train a model from scratch on the balanced multi‑cell dataset
python main.py \
  --mode train \
  --checkpoint_path model_checkpoint_balanced_multi.pth \
  --num_epochs 50

# 2️⃣  Predict & generate HTML for every neuron placed in data/to_be_sorted
python main.py \
  --mode predict \
  --checkpoint_path model_checkpoint_balanced_multi.pth

# 3️⃣  Evaluate on a single neuron
python main.py \
  --mode eval \
  --checkpoint_path model_checkpoint_balanced_multi.pth \
  --eval_target KR4

# …or evaluate on every valid dataset automatically
python main.py --mode eval --checkpoint_path model_checkpoint_balanced_multi.pth
```

### 1.1  Prepare **BBS & patch** files (from **ves\_seg**)

When you have a **vesicle segmentation volume** and its **raw image stack** exported by the upstream *ves\_seg* stage, drop them in a dedicated cell folder (e.g. `data/11‑5/`) and rename as follows:

| Old filename (ves\_seg output) | ➜ New filename (required by patch task) |
| ------------------------------ | --------------------------------------- |
| `11-5_ves.h5`                  | `vesicle_big_11-5_30-8-8.h5`            |
| `11-5_clahe.h5`                | `vesicle_im_11-5_30-8-8.h5`             |

*Replace **``**.*

Run the patch task **from the project root** (script lives in `VesicleEM/data`):

```bash
python data/vesicle_mask.py -t neuron-vesicle-patch \
       -ir data/11-5 -n 11-5 -v big -cn 1
```

Outputs (written to the same folder):

- `vesicle_big-bbs_11-5_30-8-8.h5`  ← bounding‑box metadata
- `vesicle_big_11-5_30-8-8_patch.h5` ← image + mask patches (5×31×31, z‑y‑x)

Copy **both** files into `data/to_be_sorted/` before continuing with `main.py predict`.

## 2  Folder layout  Folder layout

```
project_root/
├── main.py                # entry‑point with CLI
├── scripts/               # helper modules (data_loader.py, vesicle_net.py, …)
├── data/
│   ├── to_be_sorted/      # DROP raw .h5 files here → auto‑sorted by predict mode
│   └── <cell>/            # e.g. KR4/, SHL55/, balanced_from_multi/
│       ├── vesicle_big_<cell>_30-8-8_patch.h5   # raw stack (images + masks)
│       ├── vesicle_big-bbs_<cell>_30-8-8.h5     # bounding‑boxes (IDs in col 0)
│       ├── im.h5          # created by split_h5_file() or placed manually
│       ├── mask.h5        #   "
│       └── label.h5       # 1‑based ground‑truth labels (CV=1, DV=2, DVH=3)
├── results/
│   └── <cell>/            # prediction PNGs, per‑class .h5, HTML & helper files
│       ├── CV/
│       ├── DV/
│       ├── DVH/
│       ├── js/
│       ├── saved_0/
│       └── test_0/        # main HTML pages live here
└── model_checkpoint_balanced_multi.pth
```

### HTML output location

The interactive proofreading pages are created in `results/<cell>/test_0/`. Open `test_0/test_0.html` in a browser to start labeling.

### Key files

| File                                 | Purpose                                                                                                |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `vesicle_big_<cell>_30-8-8_patch.h5` | Raw volume with **two datasets**: images & masks. `split_h5_file()` converts into `im.h5` + `mask.h5`. |
| `vesicle_big-bbs_<cell>_30-8-8.h5`   | Bounding boxes; first column is vesicle ID, reused for filenames.                                      |
| `label.h5`                           | 1‑based labels; **required only for training/eval**.                                                   |

---

## 3  CLI in detail

| Mode        | Required flags                      | Optional flags                        | What happens                                                                                                                                                                                                                                                         |
| ----------- | ----------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **train**   | `--checkpoint_path`, `--num_epochs` | (uses *all* labelled cells, balanced) | 1. Builds a balanced dataset under `data/balanced_from_multi/` 2. Trains `StandardNet` 3. Saves checkpoint 4. Immediately evaluates on every valid cell and prints a metrics table.                                                                                  |
| **predict** | `--checkpoint_path`                 | (none)                                | 1. Calls `sort_files_to_directories()` to auto‑organise **every** file in `data/to_be_sorted/` 2. Runs inference, saves PNG montages per vesicle, stores per‑class prediction IDs (`CV.h5`, `DV.h5`, `DVH.h5`) 3. Generates interactive HTML pages for proofreading. |
| **eval**    | `--checkpoint_path`                 | `--eval_target <folder>`              | If `eval_target` is given → evaluate that single folder; else iterate through **all** folders that have `im.h5`, `mask.h5`, `label.h5`. Generates `evaluation_summary.png` with macro & per‑class metrics (Rand, F1, precision, recall).                             |

---

## 4  Typical workflow

1. **Collect raw vesicle stacks** into `data/to_be_sorted/`.
2. **Run predict mode** to auto‑sort, classify, and create HTML.
3. **Proofread** the HTML in a browser & export edited `(vesicleID:label)` txt files (optional).
4. **Merge txt** annotations → `label.h5` via `scripts/html_txt_merge.py`.
5. **Train** on balanced data to update the model.
6. **Evaluate** new checkpoint and iterate.

---

## 5  Troubleshooting / FAQ

| Symptom                                                              | Likely cause & fix                                                                     |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `FileNotFoundError: Source folder does not exist: data/to_be_sorted` | Create the folder or pass the right path.                                              |
| CUDA‑related OOM                                                     | Reduce `--batch_size` in `main.py` (edit hard‑coded default).                          |
| No HTML pages generated                                              | Check that prediction finished without errors and that `results/<cell>/` is non‑empty. |

---

## 6  Citation & Credits

The pipeline uses [PyTorch](https://pytorch.org), `scikit‑image`, and HTML templates adapted from MIT proofreading tools. Original CNN architecture is a straightforward 3D ConvNet similar to **LeCun‑style ConvNets**, adapted for vesicle volumes.

Developed by **Yutian (Tim) Fan**, 2024‑2025.



---

