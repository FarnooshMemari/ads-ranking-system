# Criteo Dataset

**Chosen dataset/version**: Criteo Display Advertising Challenge, **sample variant**
(`dac_sample.txt`, "dac-v2", ~100K rows) — the same "sample" size used by Microsoft
Recommenders' `lightgbm_tinycriteo.ipynb` notebook.

Everything below was verified directly: by reading `recommenders/datasets/criteo.py` in the
Microsoft Recommenders repository, and by downloading and inspecting the actual sample archive
(including its bundled `readme.txt` and `license.txt`) — not inferred from column names or prior
assumptions.

## Source

- **Mirror used by this project**: `https://huggingface.co/datasets/Recommenders/Criteo/resolve/main/dac_sample.tar.gz`
  (~8.8 MB compressed) — this is the exact URL Microsoft Recommenders' own `criteo.py` loader
  downloads from, redistributed by the Recommenders team on Hugging Face Hub. No login,
  authentication, or click-through terms gate is required to download it.
- **Original dataset owner**: Criteo Labs, originally distributed for the "Display Advertising
  Challenge" hosted on Kaggle (`https://www.kaggle.com/c/criteo-display-ad-challenge/`).
  Assembled by Olivier Chapelle.
- **Terms of use** (from the archive's own `license.txt`, "CRITEO LABS DATA TERMS OF USE"):
  non-commercial use only; you may publish analyses/interpretations but not redistribute the
  data itself; you must not reverse-engineer it; Criteo retains all ownership. This project
  complies by: never committing the raw data to Git (`data/raw/` is `.gitignore`d) and using it
  only for personal, non-commercial, portfolio/learning purposes.

## Why "sample" and not "full"

Microsoft Recommenders' `criteo.py` supports exactly two variants, both verified from source:

| Variant | Source file | Compressed size | Extracted file | Rows |
|---|---|---|---|---|
| `sample` | `dac_sample.tar.gz` | ~8.8 MB | `dac_sample.txt` | **100,000** (verified by direct download + `wc -l`) |
| `full` | `dac.tar.gz` | ~4.58 GB | `train.txt` | ~45.8M (the full Kaggle Display Advertising Challenge training set; not downloaded or verified locally, but is a well-documented public figure for this dataset) |

**Recommendation: `sample`.** This isn't chosen merely for being smaller — the tradeoff was
weighed explicitly:

- **Relevance to CTR prediction**: identical schema and semantics to `full`; it's a genuine
  subsample of the same Criteo traffic, not a synthetic or toy dataset. Every model/feature
  technique that would apply to `full` applies here.
- **Compatibility with Microsoft Recommenders**: this is the exact size used by the `recommenders`
  library's own `lightgbm_tinycriteo.ipynb` example — using it keeps this project directly
  comparable to, and reusable within, the eventual open-source notebook contribution (see
  `recommenders-ads-extension/docs/`).
- **Computational feasibility**: `full` is ~4.6 GB compressed / ~11 GB+ uncompressed and 45.8M
  rows — impractical to download, store, and iterate on repeatedly on a laptop for a portfolio
  project. `sample` is 8.8 MB and loads in memory in seconds.
- **Reproducibility**: `sample` downloads in seconds on any connection and requires no special
  infrastructure, so anyone cloning this repo can reproduce the exact dataset used. `full` would
  make the project much harder for a reviewer (or the open-source maintainers) to reproduce
  quickly.
- **Cost of this choice**: `sample` is small enough that results (e.g. AUC) will be noisier and
  less representative of true model capacity than `full` would give — this is an accepted
  tradeoff for a portfolio/demo project, not a claim that `sample` is sufficient for a
  production-grade model.

## Schema (verified against the real file)

Confirmed by downloading `dac_sample.tar.gz`, extracting `dac_sample.txt`, and inspecting it
directly (`wc -l`, column counts, value distributions):

- **Rows**: exactly 100,000.
- **Columns**: exactly 40, tab-separated, **no header row** (column names are assigned by the
  loader, not present in the file).
- **Format** (from the archive's own `readme.txt`): `<label> <integer feature 1..13> <categorical feature 1..26>`.
- **Ordering**: rows are chronologically ordered (per `readme.txt`); there is no explicit
  timestamp column.

| Column(s) | Name(s) used in this project | Type | Notes |
|---|---|---|---|
| 1 | `label` | binary int (0/1) | Click indicator. Verified fully populated, no missing values, only values `{0, 1}` present (77,337 zeros / 22,663 ones in the sample file). |
| 2–14 | `int00`–`int12` (13 cols) | integer, nullable | "Mostly count features"; semantics undisclosed by Criteo. Missing values verified present (e.g. `int00` empty in 44,413/100,000 rows). |
| 15–40 | `cat00`–`cat25` (26 cols) | string (32-bit hashed) | Values are hashed for anonymization; semantics undisclosed. Missing values verified present (e.g. `cat25` empty in 41,471/100,000 rows). |

### Label definition

`label` = 1 if the ad was clicked, 0 otherwise. **Important verified caveat**: per the archive's
`readme.txt`, *"the positive (clicked) and negative (non-clicked) examples have both been
subsampled (but at different rates) in order to reduce the dataset size."* The ~22.7% positive
rate observed in this file is an artifact of that subsampling, **not** a real-world CTR — actual
display-ad CTR is typically well under 5%. Any CTR computed directly on this dataset should be
reported as an artifact of the sample, not treated as a realistic base rate.

### Missing-value convention

Verified directly from the raw file: a missing value is an **empty field** between two tab
characters (confirmed by counting empty fields per column: e.g. 44,413 empty values in `int00`,
13,708 in a later numeric column, 41,471 in the last categorical column). `pandas.read_csv` parses
these as `NaN` with no additional handling required at load time; no sentinel value (like `-1` or
`"unknown"`) is used in the raw file.

## Identifiers — verified absent

The task explicitly required verifying identifiers from the dataset's actual documentation/schema
rather than inferring from feature names. Confirmed from both the loader's own column
specification (`recommenders/datasets/criteo.py`: `DEFAULT_HEADER = [label] + int00..int12 +
cat00..cat25`) and the archive's `readme.txt` ("The semantic of these features is undisclosed"):

| Identifier | Present? |
|---|---|
| User ID | **No** — no column exists for this. |
| Ad ID | **No** — no column exists for this. |
| Request/session ID | **No** — no column exists for this; rows are only chronologically ordered, with no grouping key. |
| Conversion label | **No** — only a single binary click label exists; there is no downstream conversion event in this dataset. |
| Advertiser/campaign ID | **No** — no column exists for this. |

The 26 categorical columns are anonymized 32-bit hashes with **undisclosed semantics** — it is not
possible to determine from the data itself whether any of them happens to correlate with a
user, ad, or campaign concept; none is documented as such, and none should be assumed to be one.

## Known limitations

- **No candidate generation is possible on this dataset.** Candidate generation requires a
  grouping key (e.g. multiple ads competing for the same user/request) to narrow a candidate set.
  This dataset has no such key — each row is an independent impression with no linkage to any
  other row. `src/retrieval/` remains unimplemented placeholders for this reason; this dataset
  cannot exercise that stage.
- **No CVR (conversion rate) prediction is possible on this dataset.** There is no conversion
  label, synthetic or otherwise, in the raw data. `src/evaluation/ads_metrics.py`'s `compute_cvr`
  remains a placeholder for this reason.
- **Anonymized/undisclosed features** limit interpretability: no feature importance or ablation
  analysis can be tied to real-world semantics (e.g. "device type" or "ad category") since none of
  the columns are labeled as such.
- **Subsampled label distribution** (see Label definition above) means observed CTR on this
  dataset is not representative of a real platform's CTR.
- **Chronological order without timestamps** allows a time-respecting split (train on earlier
  rows, evaluate on later ones) but not calendar-based features (day-of-week, hour-of-day, etc.).

## Why this dataset is appropriate for this project

It is the same, real, industry-standard CTR benchmark already used by Microsoft Recommenders'
own example notebooks, so results and code built here are directly comparable to and reusable in
the open-source contribution being prepared separately. It's large enough to train and evaluate a
real CTR model (LightGBM, per the Phase 2 plan) and small enough to download, store, and iterate
on entirely on a laptop, with no manual terms-of-service click-through blocking automated
acquisition. Its explicitly undisclosed, anonymized feature semantics are also an honest match for
this project's scope: a CTR *ranking* prototype, not a claim of building a full advertiser/campaign
data platform.

## Acquisition instructions

Automated (recommended):

```bash
make download-data
# or:
python3 scripts/download_criteo.py
```

This downloads `dac_sample.tar.gz` from the Hugging Face mirror above into
`data/raw/criteo/` (path from `configs/config.yaml`: `criteo.raw_dir`), extracts it, and leaves
`dac_sample.txt` (plus the archive's own `readme.txt`/`license.txt`) in place. Re-running the
script is a no-op if the extracted file already exists (pass `--force` to re-download).

Manual fallback (if automated download fails, e.g. no network access in your environment): download
`dac_sample.tar.gz` from the URL above in a browser, then extract it so that
`dac_sample.txt` ends up at `data/raw/criteo/dac_sample.txt`. No account or terms click-through is
required on the Hugging Face mirror; see the **Terms of use** section above for the underlying
Criteo Labs license that applies to the data itself regardless of how it's obtained.

The raw file is never committed to Git — `data/raw/` is `.gitignore`d, consistent with both the
non-redistribution term in the dataset's license and this project's existing convention for all
raw data.
