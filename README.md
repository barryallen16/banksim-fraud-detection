# Fraud detection on BankSim1 with DBSCAN

Unsupervised fraud detection on the BankSim1 transaction dataset. DBSCAN over a parameter
grid, tuned with 5-fold cross-validation.

| | |
|---|---|
| Winning config | `eps=0.75`, `min_samples=3`, `fraud_threshold=0.175` |
| Recall | 0.6617 |
| Precision | 0.6854 |
| F1 | 0.6732 |
| AUC | 0.8290 |

Those are the average test metrics across the 5 folds. The full sweep log is in
`results/nb-v5-cv.txt`.

## What is where

| | |
|---|---|
| Final model | `models/final-dbscan-eps075-ms3-thr0175.pkl` |
| Final model notebook | `notebooks/03-unsupervised-dbscan/notebook-v5-dbscan-cross-val.ipynb` |
| Final evaluation | `notebooks/05-misc-experiments/anol_eval.ipynb-banksim1-final-eval.ipynb` |
| Sweep log | `results/nb-v5-cv.txt` |
| Demo app | `app/app.py` |

## Dataset

`ealaxi/banksim1` from Kaggle, pulled with `kagglehub` as `bs140513_032310.csv`, about
1.48M transactions. Feature engineering covers timestamp decomposition, haversine distance
between customer and merchant, and frequency encoding on customer, merchant, category and
amount band.

`data/banksim_with_clusters.csv` is an older, pre-clustered copy of a related dataset. It
belongs to the app, not to the notebooks.

## Approach

Fraud labels only enter after clustering. DBSCAN groups the scaled feature space, each
cluster is assigned the fraud rate of its members, and any cluster above
`fraud_threshold` is flagged. Test points are labelled by nearest-neighbour lookup against
the fitted core points, so no refit is needed per fold.

The grid was 60 combinations of `eps`, `min_samples` and `fraud_threshold`, each scored on
5 folds. HDBSCAN was evaluated as an alternative and scored F1 0.1064
(`results/nb-v4-hdb.txt`), so DBSCAN won.

## Running the app

`app/app.py` is a Gradio interface over the trained model. It expects `preprocessor.pkl`
and `dbscan_fraud_model.pkl` in the same directory, both of which are committed.

```
cd app
python -m venv .venv
.venv/Scripts/pip install gradio pandas numpy joblib scikit-learn
.venv/Scripts/python app.py
```

The two pickle files were produced by scikit-learn's `joblib` against a specific version.
If unpickling fails on a version mismatch, match the scikit-learn version recorded in the
notebook that trained them.

Running the notebooks needs an NVIDIA GPU. They import `cudf`, `cuml` and `cupy`, so they
will not execute on CPU.

## Layout

```
notebooks/
  01-data-understanding/   EDA, 4 notebooks and 1 Rmd file
  02-supervised-baseline/  6 notebooks and 1 Rmd, RF, XGBoost, K-Means, isolation forest
  03-unsupervised-dbscan/  main lineage, 8 notebooks, notebook-v1 through v5
  04-cuml-branch/          GPU branch, 12 notebooks and 8 scripts, cuML DBSCAN and HDBSCAN
  05-misc-experiments/     autoencoder, testing-dbscan, the final evaluation
  06-unsorted-scratch/     28 notebooks kept under their original names
app/                       Gradio demo, app.py plus 2 model files
models/                    17 tracked artifacts, 9 pkl, 5 joblib, 3 json
results/                   logs from the parameter sweep
docs/                      writeups, PDF exports
media/                     plots
data/                      banksim_with_clusters.csv, y_labels.npy
_archive/                  three earlier snapshots of the project, extracted
```

## Three separate code paths

The three sets of code were written against different data and were never merged.

1. `notebooks/03-unsupervised-dbscan/` is the project. Cross-validated sweep on banksim1.
2. `notebooks/04-cuml-branch/` is a GPU branch against the older `X_scaled.csv` era data,
   abandoned once the sweep produced better numbers.
3. `app/` is a Gradio UI trained on `banksim_with_clusters.csv`. It runs, but it loads a
   different model from the one above.

## Things that will trip you up

- `notebook-v1` through `v5` each exist at two sizes. The small ones, 13 to 27 KB, are the
  cleaned set with outputs stripped. The large ones, 91 to 131 KB, are preserved in
  `_archive/`.
- Several files in `models/` share a name and a byte size but hold different models.
  Different training runs, same dimensions. Dedupe by hash, not by filename.
- Suffixes in the tree mark how a file relates to its siblings: `-full-outputs` is a richer
  run of the same notebook, `-rerun` is a repeat execution of a script, and `gradio-`
  marks artifacts consumed by `app/app.py`.