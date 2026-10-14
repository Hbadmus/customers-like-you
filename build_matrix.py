"""DE1-DE5: rebuild the customer x product matrix from the raw Excel file.
Usage: python build_matrix.py
"""
import json
import numpy as np, pandas as pd, scipy.sparse as sp

RAW = "data/raw/online_retail_II.xlsx"
OUT = "data/processed/"
MIN_CUSTOMERS = 10   # keep products bought by at least 10 training customers
TEST_MONTHS = 2


# load both sheets and unify column names
def load():
    sheets = pd.read_excel(RAW, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    return df.rename(columns={"InvoiceNo": "Invoice", "UnitPrice": "Price", "CustomerID": "Customer ID"})


def profile(df):
    # DE1: one row per column
    p = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                      "missing_%": (df.isna().mean() * 100).round(2)})
    p.to_csv(OUT + "profile.csv")
    print(f"Rows loaded: {len(df):,}\n{p}\n")


def clean(df):
    # DE2: each rule logs how many rows it removed
    rules = [
        ("missing customer ID", lambda d: d["Customer ID"].notna()),
        ("cancellation (invoice starts with C)", lambda d: ~d["Invoice"].str.startswith("C")),
        ("quantity <= 0", lambda d: d["Quantity"] > 0),
        ("price <= 0", lambda d: d["Price"] > 0),
        ("non-product code (POST, M, D, ...)", lambda d: d["StockCode"].str.match(r"^\d{5}")),
        ("exact duplicate line", lambda d: ~d.duplicated()),
    ]
    log = [("raw", 0, len(df))]
    for name, keep in rules:
        before = len(df)
        df = df[keep(df)]
        log.append((name, before - len(df), len(df)))
    log = pd.DataFrame(log, columns=["rule", "rows_removed", "rows_left"])
    log.to_csv(OUT + "cleaning_log.csv", index=False)
    print(log, "\n")
    df = df.copy()
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["StockCode"] = df["StockCode"].str.upper()
    return df


def split(df):
    # DE3: last 2 months = test. Hold out only products the customer never bought in train.
    cutoff = df["InvoiceDate"].max() - pd.DateOffset(months=TEST_MONTHS)
    train, test = df[df["InvoiceDate"] < cutoff], df[df["InvoiceDate"] >= cutoff]

    # product filter uses training data only
    n_cust = train.groupby("StockCode")["Customer ID"].nunique()
    keep = n_cust[n_cust >= MIN_CUSTOMERS].index
    train = train[train["StockCode"].isin(keep)]

    train_agg = train.groupby(["Customer ID", "StockCode"], as_index=False)["Quantity"].sum()
    test_pairs = test[test["StockCode"].isin(keep) & test["Customer ID"].isin(train_agg["Customer ID"])]
    test_pairs = test_pairs[["Customer ID", "StockCode"]].drop_duplicates()
    seen = set(zip(train_agg["Customer ID"], train_agg["StockCode"]))
    test_pairs = test_pairs[[p not in seen for p in zip(test_pairs["Customer ID"], test_pairs["StockCode"])]]

    train_agg.to_csv(OUT + "train.csv", index=False)
    test_pairs.to_csv(OUT + "test.csv", index=False)
    print(f"Cutoff {cutoff.date()} | train lines {len(train):,} | kept products {len(keep):,} | "
          f"test customers {test_pairs['Customer ID'].nunique():,} | held-out pairs {len(test_pairs):,}\n")
    return train, train_agg, cutoff


def build(train, train_agg, cutoff):
    # DE4: purchase value = log(1 + total quantity), damps wholesale bulk orders
    cust = sorted(train_agg["Customer ID"].unique())
    prod = sorted(train_agg["StockCode"].unique())
    ci = {c: i for i, c in enumerate(cust)}
    pj = {p: j for j, p in enumerate(prod)}
    A = sp.csr_matrix((np.log1p(train_agg["Quantity"].values),
                       (train_agg["Customer ID"].map(ci), train_agg["StockCode"].map(pj))),
                      shape=(len(cust), len(prod)))
    sp.save_npz(OUT + "A.npz", A)

    # index maps (customer IDs stay inside the team, R4)
    pd.DataFrame({"row": range(len(cust)), "customer_id": cust}).to_csv(OUT + "customers.csv", index=False)
    country = train.groupby("Customer ID")["Country"].agg(lambda s: s.mode()[0])
    pd.DataFrame({"row": range(len(cust)), "country": country.loc[cust].values}).to_csv(
        OUT + "customer_country.csv", index=False)
    desc = train.groupby("StockCode")["Description"].agg(lambda s: s.mode()[0] if s.notna().any() else "")
    pd.DataFrame({"col": range(len(prod)), "stock_code": prod, "description": desc.loc[prod].values}).to_csv(
        OUT + "products.csv", index=False)

    # DE5: matrix report
    dense_mb = A.shape[0] * A.shape[1] * 8 / 1e6
    sparse_mb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e6
    report = {"customers": A.shape[0], "products": A.shape[1], "non_zeros": int(A.nnz),
              "sparsity_%": round(100 * (1 - A.nnz / (A.shape[0] * A.shape[1])), 2),
              "dense_MB": round(dense_mb, 1), "sparse_MB": round(sparse_mb, 1),
              "train_cutoff": str(cutoff.date()), "purchase_value": "log1p(quantity)",
              "min_customers_per_product": MIN_CUSTOMERS}
    json.dump(report, open(OUT + "matrix_report.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    df = load()
    profile(df)
    df = clean(df)
    build(*split(df))
