# Part 2 — Supervised ML: Regression + Classification

Loads `cleaned_data.csv` from Part 1 (53,794 rows, diamonds dataset).

## Label definitions
- **Feature matrix X**: all columns except `price` — 9 columns before
  encoding (`carat, cut, color, clarity, depth, table, x, y, z`).
- **Regression label `y_reg`**: `price` (continuous, USD).
- **Classification label `y_clf`**: `1` if `price` > median price, else `0`
  ("Expensive" vs "Cheap"). Class balance came out almost exactly even:
  21,557 vs 21,478 in the training split (50.09% / 49.91%).

Confirmed shapes after defining these: `X` = (53,794, 9), `y_reg` =
(53,794,), `y_clf` = (53,794,).

## Encoding categorical columns (Task 2)
Three categorical columns: `cut`, `color`, `clarity`.
- **`cut`** has a genuine quality order (Fair < Good < Very Good < Premium
  < Ideal), so it was **label-encoded** by explicitly mapping each category
  to an integer that preserves that order
  (`{"Fair":0,"Good":1,"Very Good":2,"Premium":3,"Ideal":4}`).
- **`color`** and **`clarity`** have no such single agreed ordering usable
  directly as an integer scale in this notebook, so both were **one-hot
  encoded** with `pd.get_dummies(..., drop_first=True)`, dropping the first
  dummy of each to avoid multicollinearity. One-hot encoding avoids forcing
  a false ordinal relationship onto these categories — label-encoding them
  instead would make the model treat e.g. `color_G` as "between" `color_F`
  and `color_H` in a specific numeric sense that isn't necessarily how the
  model should weigh them.

After encoding, `X` grew from 9 to **20 columns** (confirmed via `X.shape`
→ `(53794, 20)`): `carat, cut, depth, table, x, y, z` plus 6 `color_*` and
7 `clarity_*` dummy columns.

## Leak-free split and scaling (Task 3)
`train_test_split(X, y, test_size=0.2, random_state=42)` was run separately
for the regression and classification labels (same `X`, same
`random_state=42`, so both splits select the identical rows for
train/test — done as two calls purely for workflow clarity, not two
different partitions). Resulting shapes: train (43,035, 20), test
(10,759, 20) for both.

A **separate `StandardScaler`** was fit only on each split's *training*
features (`scaler.fit_transform(X_train)`), then used to transform the
matching test set (`scaler.transform(X_test)`) — the scaler never sees the
test rows during fitting. Fitting a scaler on the full dataset (train+test
combined) would be data leakage: the scaler's mean/std would then encode
information about the test set's distribution into what is supposed to be
a "test of generalization to unseen data," making the reported metrics
overly optimistic.

## Regression — Linear Regression vs Ridge (Task 4)
| Model | MSE | R² |
|---|---|---|
| Linear Regression | 1,221,432.94 | 0.9199 |
| Ridge (alpha=1.0) | 1,221,478.17 | 0.9199 |

Top 3 features by \|coefficient\|: **carat** (5372.42), **clarity_VS2**
(1843.89), **clarity_VS1** (1686.67). A large **positive** coefficient
(e.g. `carat`) means a one-standard-deviation increase in that scaled
feature is associated with a predicted price increase of roughly that many
dollars, holding other features fixed. `x` has a large **negative**
coefficient (-1130.86) despite being strongly, positively correlated with
price on its own (Part 1's heat map showed `carat`/`x` correlation ≈0.975)
— this is a multicollinearity artifact: because `carat`, `x`, `y`, and `z`
all measure the diamond's size and move together almost perfectly, OLS
distributes credit for "bigger diamond → higher price" unevenly across
them, sometimes assigning a compensating negative weight to one of the
correlated features even though it isn't truly driving prices down.

Ridge produced an almost identical MSE/R² here (1,221,478 vs 1,221,433;
0.91986 vs 0.91986) because `alpha=1.0` is a fairly mild penalty relative
to the scale of these coefficients. In general, Ridge would be expected to
shrink the highly-correlated size features (`carat`, `x`, `y`, `z`) toward
each other more than plain OLS does as `alpha` increases — `alpha` controls
the strength of the L2 penalty on the coefficients: higher alpha shrinks
coefficients more (reducing variance from multicollinearity at some cost
in bias), lower alpha lets Ridge behave more like plain OLS.

## Classification — Logistic Regression (Task 5)
`y_clf_train.value_counts()`: 21,557 (class 0) vs 21,478 (class 1) —
**50.09% / 49.91%**, i.e. already balanced well above the 35% imbalance
threshold, so **no SMOTE/`class_weight` resampling was applied** — this was
checked directly rather than assumed.

- Confusion matrix: `[[5230, 115], [133, 5281]]`
- Accuracy **0.9769**, precision/recall/F1 **0.98** for both classes
- **AUC = 0.9979**

**Precision** = TP / (TP + FP). **Recall** = TP / (TP + FN). For an
"is this diamond above-median price" task, a false negative (calling an
expensive stone "cheap") is arguably the costlier mistake for a jeweler
pricing inventory, so **recall** is the more important metric here — though
with both precision and recall already at ~0.98, this model doesn't need
to trade one off against the other in practice. An AUC of 0.998 means the
model separates the two price classes almost perfectly: a randomly chosen
above-median diamond is ranked higher than a randomly chosen below-median
diamond in about 99.8% of comparisons.

### Threshold sensitivity (0.30–0.70)
| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.9593 | 0.9874 | 0.9732 |
| 0.40 | 0.9703 | 0.9825 | 0.9763 |
| **0.50** | **0.9787** | 0.9754 | **0.9771** |
| 0.60 | 0.9851 | 0.9669 | 0.9760 |
| 0.70 | 0.9895 | 0.9538 | 0.9713 |

**F1-maximizing threshold: 0.50** (the default). Since recall matters
slightly more for this task, lowering the threshold from 0.50 toward 0.30
or 0.40 would trade a small amount of precision for a small gain in recall
(catching a few more truly-expensive stones at the cost of a few more
false alarms) — a reasonable choice if the business cost of underpricing
a diamond is meaningfully higher than the cost of over-flagging one, but
given the default threshold already achieves the best F1 here, there's no
strong reason to move off 0.50 unless that asymmetric cost is explicit.

## Regularization experiment (C=1.0 vs C=0.01)
| Model | Precision | Recall | AUC |
|---|---|---|---|
| C=1.0 (baseline) | 0.9787 | 0.9754 | 0.9979 |
| C=0.01 (strong reg.) | 0.9763 | 0.9745 | 0.9970 |

`C` is the **inverse** of the regularization strength in scikit-learn's
`LogisticRegression` — a smaller `C` means a stronger L2 penalty, pulling
coefficients toward zero and reducing variance at the cost of some bias.
Reducing `C` from 1.0 to 0.01 slightly **worsened** performance across all
three metrics here (small drops in precision, recall, and AUC), suggesting
the baseline model was not overfitting, and the extra regularization mostly
discarded genuinely useful signal rather than noise.

### Bootstrap confidence interval for the AUC difference
500 bootstrap resamples of the test set (`np.random.choice`, with
replacement, `np.random.seed(42)`) were used to compute
`AUC(C=1.0) - AUC(C=0.01)` per sample:
- **Mean AUC difference: 0.000917**
- **95% CI: [0.000571, 0.001330]**
- The interval **excludes zero**, so the C=1.0 model's small AUC advantage
  over the C=0.01 model is statistically consistent across resamples of
  the test data rather than just noise — though, as with Part 2's own
  numbers show, the absolute size of the advantage (~0.0009 AUC) is small
  in practical terms.

## Files
- `part2_models.ipynb` — full notebook (all tasks, all outputs, ROC curve
  plot rendered inline). Expects `cleaned_data.csv` from `part1/` to be in
  the same working directory (or update the path in the first code cell).
