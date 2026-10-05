# Bank Marketing: Logistic Regression

Predicting which clients will subscribe to a term deposit, using a plain logistic regression written from scratch in numpy.

## The goal
A bank calls people to sell a savings account (term deposit). Only about 11% say yes. This model predicts who is likely to say yes, so the bank can call those people first.

## How to run
```
python bank_logreg_final.py
```
Requires `numpy` and `pandas` (pandas is only used to read the CSV). Put `train.csv` in the same folder.

## How the model learns
- Each feature gets a weight (how much it matters).
- Weighted features are added up into a score, and sigmoid squashes it into a probability between 0 and 1.
- The model compares predictions to the real answers, measures the error (cross-entropy), and nudges the weights to reduce it. This repeats 6,000 times (gradient descent).

No ML libraries are used for the model, splits, cross-validation, encoding, or metrics. All are written by hand.

## Preprocessing decisions
Models only read numbers, so every column needed a decision. Two rules guided each one:
1. **Would I know this before calling the person?** If not, it can't be used (data leakage).
2. **Keep the real meaning.** Don't invent or erase differences, and use the fewest assumptions possible.

| Column | Decision | Why |
|---|---|---|
| duration | Dropped | Only known after the call ends, so it leaks the answer |
| pdays | Replaced with `contacted_before` (0/1), pdays dropped | 999 means "never contacted," not 999 days. Whether they were contacted is what matters |
| campaign, previous | Left raw | A few extreme values (max 56 calls), but a log transform made no measurable CV difference |
| contact | cellular = 1, telephone = 0 | Only two values |
| age, economic indicators | Kept as numbers | Already numeric. Adding age² showed no gain |
| job, marital, month, day_of_week, poutcome | One-hot | No natural order, so each category gets its own 0/1 column |
| education | One-hot | Has a rough order, but professional.course vs university.degree isn't clearly ranked. Tested ordinal, no difference |
| default, housing, loan | One-hot, "unknown" kept | For default, unknowns subscribe at 5.2% vs 12.9% for "no," so unknown carries real signal. Imputing would erase it, and deleting would lose 21% of rows |
| row_id | Dropped | Just an index |

All "unknown" values are kept as their own category, so nothing is thrown out or guessed.

## Evaluation
- **80/20 stratified split:** train on 80%, test once on the 20% the model never saw. Both parts keep the 11% yes rate.
- **5-fold cross-validation** on the training 80%, averaging the 5 scores so one lucky split can't fool us.
- **Threshold tuning:** the default 0.5 cutoff missed most customers because only 11% say yes. Thresholds from 0.05 to 0.50 were swept on the out-of-fold CV predictions (never the test set), and **0.21** gave the best F1.

## Results (threshold 0.21)

| Metric | CV mean ± std | Test |
|---|---|---|
| Accuracy | 0.870 ± 0.004 | 0.870 |
| Precision | 0.439 ± 0.014 | 0.439 |
| Recall | 0.552 ± 0.017 | 0.549 |
| F1 | 0.489 ± 0.013 | 0.488 |
| ROC-AUC | 0.793 ± 0.010 | 0.791 |

Confusion matrix (test):
```
              pred no   pred yes
  actual no      4993        489
  actual yes      314        382
```

**What it means:** the model flags 14% of clients and reaches 55% of the subscribers. When it says "yes," it's right 44% of the time, vs 11% when calling at random, which is about **4x better**.

**Why accuracy is misleading:** saying "no" to everyone scores 88.7% accuracy but finds zero customers. F1, recall, and ROC-AUC are the more useful scores here.

## Duration benchmark (comparison only)

| Metric | No duration (my model) | With duration |
|---|---|---|
| Accuracy | 0.870 | 0.904 |
| Precision | 0.439 | 0.561 |
| Recall | 0.552 | 0.672 |
| F1 | 0.489 | 0.612 |
| ROC-AUC | 0.793 | 0.934 |

Duration raises ROC-AUC to 0.934, but it's only known after the call, so it's data leakage. The dataset documentation says to use it only as a benchmark.

## Assumptions and limitations
- Plain logistic regression: each feature adds its own effect, and features don't interact.
- campaign, previous, and age are treated as straight-line effects.
- Unknowns may behave differently from known values (true for default).
- The 5 economic indicators are correlated, so their individual weights shouldn't be over-interpreted.
- Future clients are assumed to look like these past ones.
- Preprocessing variations all moved ROC-AUC less than the CV noise (about ±0.01), so plain logistic regression tops out around 0.79 AUC on this data. The threshold mattered most.

## Main lessons
- Most encoding choices barely changed the score. The threshold mattered most.
- Spotting leakage matters more than chasing a higher number.
- Pick metrics that fit the problem, not just accuracy.

## AI use
I used Claude to learn the concepts (gradient descent, sigmoid, cross-validation, metrics), help write the code, and test preprocessing options. An early version was AI-built. I then reset the preprocessing section and went column by column making and justifying the decisions myself. Full conversation: #https://claude.ai/share/b785dfb6-d3c2-41d9-a965-3330d4b3d998
