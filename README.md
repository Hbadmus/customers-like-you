# Customers Like You (GiftLane)
Truncated SVD recommender ("customers who bought this also bought...") and customer map, built on UCI Online Retail II.

## Success targets (J1)
| # | Target | Result |
|---|--------|--------|
| 1 | Stored factors at least 10x smaller than the dense matrix | 161.4 MB -> 7.3 MB (22x) |
| 2 | Nightly job (fit + score every product and customer) under 1 minute | about 1.5 s |
| 3 | Hit rate@10 on held-out purchases beats the 10 best sellers | 38.9% vs 26.9% |
| 4 | 3 to 6 customer groups marketing can name | 5 named segments |

## Data
Download `online_retail_II.xlsx` from https://archive.ics.uci.edu/dataset/502/online+retail+ii into `data/raw/`.
Rows loaded: 1,067,371 (two sheets, Dec 2009 to Dec 2011). `data/` is not committed.

## How to run
```
pip install -r requirements.txt
python build_matrix.py                                                    # DE1-DE5, about 1 min
python -m nbconvert --to notebook --execute --inplace analysis.ipynb     # ML1-ML7
```
All random seeds are 42.

## Files
| File | What |
|------|------|
| `build_matrix.py` | Profile, clean, time split, sparse matrix, matrix report |
| `analysis.ipynb` | Method choice, k, recommendations, evaluation, PCA vs SVD, customer map, sensitivity |
| `data/processed/` | Created by `build_matrix.py`: `profile.csv`, `cleaning_log.csv`, `matrix_report.json`, `A.npz`, index maps, train/test |
| `figures/` | Created by the notebook: `k_curve.png`, `customer_map.png`, `pca_vs_svd.csv` |

## Team roles
| Role | Owner | Tasks |
|------|-------|-------|
| Data Engineer | _name_ | DE1-DE6 |
| ML Engineer | _name_ | ML1-ML7 |

## Decision log (R1)
| Decision | Choice | Reason |
|----------|--------|--------|
| Missing customer ID | Drop (22.8% of lines) | Cannot be placed in a customer row |
| Cancellations (`C` invoices) | Drop | They are not purchases |
| Quantity or price <= 0 | Drop | Returns, adjustments, bad debt |
| Non-product codes | Keep only codes starting with 5 digits | Removes POST, M, D, DOT, BANK CHARGES, AMAZONFEE, etc. |
| Exact duplicate lines | Drop | Logging duplicates, not real repeat buys |
| Test split | Last 2 months (from 2011-10-09) | Matches the brief; time-based, so there is no future leak |
| Held-out purchases | Test products the customer never bought in training | Measures new discoveries, not repeat orders |
| Product filter | Bought by >= 10 training customers (3,687 products) | Rare items have too little signal; computed on training data only |
| Purchase value | log(1 + total quantity) | Damps huge wholesale quantities; ML7 confirms raw counts do much worse |
| Method | Truncated SVD | Works on sparse data without centring; gives customer and product factors |
| Customer scoring | Dot product V V^T (not cosine) | Cosine lost to the baseline (23%); dot product keeps popularity (39%) |
| k | 100 | Smallest k within 1 point of the best hit rate; k=200 doubles storage for +0.9 pts |
| Segments | k-means, 5 clusters on unit-length customer factors | Normalising removes volume, so the groups reflect what customers buy |

## Contribution log (C2)
| Date | Task | Owner | Evidence | Status |
|------|------|-------|----------|--------|
| Day 1 | DE1 data profile | | `data/processed/profile.csv` | |
| Day 1 | J1 success targets | | README | |
| Day 2 | DE2 cleaning | | `cleaning_log.csv` | |
| Day 3 | DE3 split, DE4 matrix | | `build_matrix.py` | |
| Day 4 | DE5 report, DE6 script | | `matrix_report.json` | |
| Day 1-5 | ML1-ML7 | | `analysis.ipynb` | |
| Day 6 | J2 memo, J3 slides | | `memo.pdf`, `slides.pptx` | |

Signed: ____________  ____________

Data: Chen, D. (2012). Online Retail II. UCI ML Repository. https://doi.org/10.24432/C5CG6D (CC BY 4.0)
