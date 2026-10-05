import numpy as np
import pandas as pd   # only used to read the CSV

THRESHOLD = 0.5       # starting value; replaced by the CV-tuned threshold in section 7b
STEPS = 6000          # enough steps for the loss to fully settle

# load data
df = pd.read_csv('train.csv').drop(columns='row_id')
y = (df['y'] == 'yes').astype(int).values
df = df.drop(columns='y')

# preprocessing
# Goal: return X (all numbers, one row per client) and a list of column names.

def one_hot(col):
    cats = sorted(col.unique())
    return np.column_stack([(col == c).astype(float) for c in cats]), [f"{col.name}_{c}" for c in cats]

def preprocess(df, include_duration=False):
    d = df.copy()
    cols, names = [], []

    # duration: only known after the call, so it would leak the answer.
    # include_duration=True is ONLY for the benchmark at the bottom.
    if not include_duration:
        d = d.drop(columns='duration')

    # pdays: 999 = never contacted; I only care whether they were contacted
    d['contacted_before'] = (d['pdays'] != 999).astype(int)
    d = d.drop(columns='pdays')

    # campaign/previous: these have a few extreme values (campaign goes up to 56, but
    # about 98% of clients had 10 or fewer calls). A log transform would have shrunk
    # those extremes so they don't dominate. Tested log vs raw with 5-fold CV: no
    # meaningful difference (~0.001 vs ~0.01 noise), so left them raw for simplicity.

    # contact: only two values, so cellular = 1, telephone = 0
    d['contact'] = (d['contact'] == 'cellular').astype(int)

    # numeric columns (grab these AFTER the fixes above)
    numeric = ['age', 'campaign', 'previous', 'contacted_before', 'contact',
               'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
               'euribor3m', 'nr.employed']
    if include_duration:
        numeric.append('duration')
    cols.append(d[numeric].values.astype(float))
    names += numeric

    # job, marital, month, day_of_week, poutcome: no natural order, so one-hot
    # (one 0/1 column per category). "unknown" kept as its own category so
    # nothing is thrown out or guessed.
    # education: has a rough order, but professional.course vs university.degree
    # isn't clearly ranked and "unknown" has no place in an order. One-hot makes
    # fewer assumptions; tested vs ordinal with CV, no meaningful difference.
    # default, housing, loan: kept "unknown" as its own category.
    # default: unknowns subscribe at 5.2% vs 12.9% for "no", so unknown carries
    #   real signal; imputing would erase it, deleting would lose 21% of rows.
    #   (default "yes" has only 1 row, so the model can't learn much from it.)
    # housing/loan: all values incl. unknown subscribe at ~11%, so kept unknown
    #   as its own category for consistency; little impact either way.
    for c in ['job', 'marital', 'education', 'default', 'housing', 'loan',
              'month', 'day_of_week', 'poutcome']:
        m, n = one_hot(d[c])
        cols.append(m)
        names += n

    return np.hstack(cols), names

X, feature_names = preprocess(df)
print(f"X shape: {X.shape}  ({X.shape[1]} features)\n")

# ---------- 3. split helpers (stratified, by hand) ----------
rng = np.random.default_rng(42)

def stratified_split(y, test_frac=0.2):
    test = []
    for cls in [0, 1]:
        idx = rng.permutation(np.where(y == cls)[0])
        test.extend(idx[:int(len(idx) * test_frac)])
    test = np.array(test)
    train = np.setdiff1d(np.arange(len(y)), test)
    return train, test

def stratified_kfold(y, k=5):
    folds = [[] for _ in range(k)]
    for cls in [0, 1]:
        idx = rng.permutation(np.where(y == cls)[0])
        for i, chunk in enumerate(np.array_split(idx, k)):
            folds[i].extend(chunk)
    all_idx = np.arange(len(y))
    for i in range(k):
        va = np.array(folds[i])
        yield np.setdiff1d(all_idx, va), va

# logistic regression model
# The model learns a weight for each of these, then combines them into one
# probability: "this client has a 35% chance of saying yes."
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def fit(X, y, lr=0.5, steps=STEPS):
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(steps):
        p = sigmoid(X @ w)
        w -= lr * (X.T @ (p - y)) / n       # cross-entropy gradient step
    return w

def scale_and_bias(X_tr, X_te):
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd[sd == 0] = 1
    X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd      # train stats only
    add1 = lambda A: np.hstack([np.ones((len(A), 1)), A])
    return add1(X_tr), add1(X_te)

# ---------- 5. metrics, by hand ----------
def roc_auc(y_true, p):
    # chance a random "yes" scores higher than a random "no" (ties count half)
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    for v in np.unique(p):                       # average ranks for ties
        m = p == v
        if m.sum() > 1: ranks[m] = ranks[m].mean()
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def scores(y_true, p, threshold=THRESHOLD):
    pred = (p >= threshold).astype(int)
    TP = np.sum((pred == 1) & (y_true == 1)); TN = np.sum((pred == 0) & (y_true == 0))
    FP = np.sum((pred == 1) & (y_true == 0)); FN = np.sum((pred == 0) & (y_true == 1))
    acc  = (TP + TN) / len(y_true)
    prec = TP / (TP + FP) if TP + FP else 0
    rec  = TP / (TP + FN) if TP + FN else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=roc_auc(y_true, p)), (TN, FP, FN, TP)

# 80/20 split 
tr_idx, te_idx = stratified_split(y)
X_train, y_train, X_test, y_test = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]

# ---------- 7. 5-fold CV on the 80% ----------
# Save each fold's predicted probabilities ("out-of-fold"), so every training
# row gets a prediction from a model that never saw it.
oof = np.zeros(len(y_train))
fold_ids = []
for k, (tr, va) in enumerate(stratified_kfold(y_train), 1):
    Xtr, Xva = scale_and_bias(X_train[tr], X_train[va])
    w = fit(Xtr, y_train[tr])
    oof[va] = sigmoid(Xva @ w)
    fold_ids.append(va)

# ---------- 7b. threshold sweep (CV predictions only, never the test set) ----------
# 0.5 is too strict when only ~11% say yes: the model rarely gets that confident.
# Try thresholds from 0.05 to 0.50 and keep the one with the best F1.
print("threshold sweep on CV predictions:")
best_t, best_f1 = 0.5, -1
for t in np.round(np.arange(0.05, 0.51, 0.01), 2):
    s_t, _ = scores(y_train, oof, threshold=t)
    if t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        print(f"  t={t:.2f}  precision {s_t['precision']:.3f}  recall {s_t['recall']:.3f}  f1 {s_t['f1']:.3f}")
    if s_t['f1'] > best_f1:
        best_t, best_f1 = t, s_t['f1']
THRESHOLD = best_t
print(f"\nbest threshold: {THRESHOLD:.2f}  (CV F1 {best_f1:.3f})\n")

# per-fold scores at the chosen threshold
fold_scores = []
for k, va in enumerate(fold_ids, 1):
    s_k, _ = scores(y_train[va], oof[va], threshold=THRESHOLD)
    fold_scores.append(s_k)
    print(f"fold {k}: " + "  ".join(f"{m} {v:.3f}" for m, v in s_k.items()))

metrics = fold_scores[0].keys()
print(f"\nCV mean ± std (threshold {THRESHOLD:.2f}):")
for m in metrics:
    vals = np.array([f[m] for f in fold_scores])
    print(f"  {m:<10}{vals.mean():.3f} ± {vals.std():.3f}")

# ---------- 8. final: train on 80%, test on 20% ----------
Xtr, Xte = scale_and_bias(X_train, X_test)
w = fit(Xtr, y_train)
s, (TN, FP, FN, TP) = scores(y_test, sigmoid(Xte @ w), threshold=THRESHOLD)

print(f"\nTEST scores (run once, threshold {THRESHOLD:.2f}):")
for m, v in s.items():
    print(f"  {m:<10}{v:.4f}")
print("\nConfusion matrix (test):")
print("              pred no   pred yes")
print(f"  actual no   {TN:>7}   {FP:>8}")
print(f"  actual yes  {FN:>7}   {TP:>8}")


# benchmark: WITH duration 
# The data dictionary says duration should only be used as a benchmark:
# it's unknown before the call, and a long call basically means "yes".
# Same splits, same CV, same threshold sweep, just with duration added.
X_dur, _ = preprocess(df, include_duration=True)
Xd_train = X_dur[tr_idx]
oof_d = np.zeros(len(y_train))
for va in fold_ids:
    tr = np.setdiff1d(np.arange(len(y_train)), va)
    a, b = scale_and_bias(Xd_train[tr], Xd_train[va])
    oof_d[va] = sigmoid(b @ fit(a, y_train[tr]))

best_td, best_fd = 0.5, -1
for t in np.round(np.arange(0.05, 0.51, 0.01), 2):
    f = scores(y_train, oof_d, threshold=t)[0]['f1']
    if f > best_fd:
        best_td, best_fd = t, f

bench_folds = [scores(y_train[va], oof_d[va], threshold=best_td)[0] for va in fold_ids]
real_folds = fold_scores
print(f"\nBENCHMARK (CV mean, each at its own best threshold):")
print(f"  {'metric':<10}{'no duration':>14}{'WITH duration':>16}")
for m in metrics:
    r = np.mean([f[m] for f in real_folds]); b = np.mean([f[m] for f in bench_folds])
    print(f"  {m:<10}{r:>14.3f}{b:>16.3f}")
print(f"  {'threshold':<10}{THRESHOLD:>14.2f}{best_td:>16.2f}")
print("  -> duration looks great, but it's leakage: not usable before the call.")




"""



"""





