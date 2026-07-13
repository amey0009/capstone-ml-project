# Part 3 — Ensembles, Tuning, and Full ML Pipeline

`part3_ensembles.ipynb` starts by recreating Part 2's exact classification
preprocessing (same `cut` label-encoding, same `color`/`clarity` one-hot
encoding, same `train_test_split(random_state=42)`, same `StandardScaler`
fit only on train) so every model here is trained/evaluated on the
identical `X_train_clf_scaled` / `X_test_clf_scaled` / `y_clf_train` /
`y_clf_test` used in Part 2.

## Decision Tree — unconstrained vs controlled (Tasks 1–2)
| Model | Train acc | Test acc | Gap |
|---|---|---|---|
| Unconstrained (`max_depth=None`) | 0.9999 | 0.9662 | **0.0338** |
| Controlled (`max_depth=5, min_samples_split=20`) | 0.9637 | 0.9640 | **-0.0003** |

The unconstrained tree hits essentially **100% training accuracy** (0.9999)
but loses ~3.4 points of accuracy on the test set — classic overfitting.
Decision trees are described as **high-variance** models because they
build splits greedily, one node at a time, choosing whatever split locally
minimizes impurity at that moment, and never revisit or reconsider an
earlier split once made — so a fully-grown tree can end up carving out a
leaf for individual training points, including noise, rather than
generalizable structure.

`max_depth` limits how many sequential levels of splits the tree is
allowed to use, trading a little bias for much lower variance;
`min_samples_split=20` refuses to split any node with fewer than 20
samples, which prevents the tree from creating splits that respond to
noise in tiny subsets of the data. Together, these two constraints don't
just shrink the train/test gap — they essentially **eliminate** it: the
controlled tree's gap is -0.0003 (test accuracy is actually marginally
*higher* than train accuracy), compared to 0.0338 for the unconstrained
tree.

## Gini vs Entropy (Task 3)
Both trained at `max_depth=5`:
- Gini test accuracy: **0.9640**
- Entropy test accuracy: **0.9628**

Gini impurity: `1 - Σ pᵢ²`. Entropy: `-Σ pᵢ log₂(pᵢ)`. A node with
**Gini = 0** means every sample reaching that node belongs to the same
class — the node is perfectly pure and does not need to be split further.

## Random Forest (Task 4)
`n_estimators=100, max_depth=10`: Train acc **0.9800**, Test acc **0.9737**,
**Test AUC 0.9973**.

Top 5 features by importance:

| Feature | Importance |
|---|---|
| y | 0.3631 |
| carat | 0.2117 |
| x | 0.1808 |
| z | 0.1800 |
| clarity_SI2 | 0.0141 |

Random Forest computes feature importance as the **average reduction in
Gini impurity contributed by a feature, across every split that uses it,
averaged over every tree in the forest**. This is fundamentally different
from a linear regression coefficient: a coefficient measures a linear,
additive effect on the raw prediction (and can be interpreted directly as
"one unit of X changes the prediction by this much"), whereas importance
measures how useful a feature was for *separating classes* anywhere in any
tree — capturing non-linear and interaction effects a linear coefficient
can't, and (unlike a coefficient) always non-negative and directionless.

**Bagging**: each of the 100 trees is trained on an independent bootstrap
sample — drawn with replacement from the training set, same size as the
original — and at every split, only a random subset of √(20)≈4–5 features
is considered as candidates, which decorrelates the trees from one
another. Averaging/voting across many such decorrelated trees cancels out
each individual tree's idiosyncratic overfitting, sharply reducing
variance compared to a single deep decision tree while keeping bias low —
visible here in Random Forest's much smaller train/test gap (0.9800 vs
0.9737, a 0.0063 gap) compared to the unconstrained single tree's 0.0338
gap.

## Gradient Boosting (Task 4a)
`n_estimators=100, learning_rate=0.1, max_depth=3`: Train acc **0.9752**,
Test acc **0.9738**, **Test AUC 0.9974** — the highest single-model test
AUC among the three tree-based ensembles trained so far.

## Feature ablation study (Task 4b)
The 5 lowest-importance features from the Random Forest were:
**clarity_IF, clarity_VS1, clarity_VS2, color_G, color_F**.

| Model | Test AUC |
|---|---|
| Full model (all 20 features) | 0.9973 |
| Reduced model (5 lowest-importance features removed) | 0.9967 |

AUC drops only slightly, from 0.9973 to 0.9967 (**-0.0006**), when these
five features are removed — this small drop suggests these particular
`clarity`/`color` dummy columns were **close to genuinely uninformative
for this Random Forest**, contributing very little beyond what `carat`,
`x`, `y`, `z` (and the remaining clarity/color dummies) already capture.
In production, this supports deploying a simpler, lower-dimensional model:
dropping these 5 columns reduces inference cost and maintenance burden
(one-hot columns to maintain, feature-engineering code to keep in sync)
at essentially no cost to ranking quality — an AUC drop of 0.0006 is well
within a tolerable threshold for most business use cases, so removing
them would be a reasonable simplification if runtime/maintenance cost
mattered.

## Cross-validated comparison (Task 5)
5-fold `StratifiedKFold`, `scoring='roc_auc'`:

| Model | Mean AUC | Std AUC |
|---|---|---|
| LogisticRegression | **0.9982** | 0.0002 |
| DecisionTree(depth=5) | 0.9928 | 0.0005 |
| RandomForest | 0.9972 | 0.0002 |
| GradientBoosting | 0.9969 | 0.0003 |

Cross-validation gives a more reliable estimate of generalization than a
single train/test split because it averages performance across 5
different held-out folds — a single split's score can look optimistic or
pessimistic purely by chance depending on which rows happened to land in
the test set; averaging (and reporting the std) shows both the expected
performance and how sensitive that performance is to the particular split
chosen. Interestingly, plain Logistic Regression narrowly leads all three
tree-based ensembles here — this dataset's classification task (price
above/below median) turns out to be close to linearly separable once the
strongly-correlated size features are scaled, so the extra flexibility of
trees/boosting doesn't add much.

## GridSearchCV (Task 6)
Pipeline: `make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
RandomForestClassifier(random_state=42))`.

Grid: `n_estimators` ∈ {50,100,200} × `max_depth` ∈ {5,10,None} ×
`min_samples_leaf` ∈ {1,5} = **18 configurations × 5 folds = 90 fits**.

**Best params**: `max_depth=None, min_samples_leaf=1, n_estimators=200`.
**Best CV score**: 0.9978. **Best pipeline test AUC**: **0.9980** — the
highest test AUC of any model in this project so far.

Grid Search is exhaustive — it evaluates every combination in the grid, so
it's guaranteed to find the best point *within that grid*, but its cost
grows multiplicatively with the number of parameters and values tried (90
fits here for a fairly small 3-parameter grid). Randomized Search instead
samples a fixed number of random combinations from the same space, trading
a small chance of missing the single best combination for a cost that no
longer grows combinatorially — preferable when the grid is large or when
some parameters have diminishing returns beyond a coarse search.

## Manual learning curve (Task 7)
Best pipeline re-fit on 20/40/60/80/100% of `X_train_clf`:

| Training fraction | Training AUC | Test AUC |
|---|---|---|
| 0.2 | 1.0000 | 0.9967 |
| 0.4 | 1.0000 | 0.9975 |
| 0.6 | 1.0000 | 0.9977 |
| 0.8 | 1.0000 | 0.9977 |
| 1.0 | 1.0000 | 0.9980 |

(i) Training AUC is **flat at 1.0000** across every fraction, not
decreasing — this Random Forest (with `min_samples_leaf=1`, i.e. no
constraint against fitting individual points) can perfectly rank its own
training rows regardless of how much data it sees.
(ii) Test AUC **does increase, slightly but consistently**, from 0.9967 at
20% of the data up to 0.9980 at 100% — so collecting more rows of this
same kind of data would likely still help a little, though the gains are
getting smaller each step (0.0008, 0.0002, 0.0000, 0.0003).
(iii) **Conclusion: the model is closer to the data-limited end, but with
diminishing returns** — test AUC is still very slowly rising at 100% of
the data rather than having fully plateaued, but the increments are
already small (well under 0.001 per doubling of data), so while more data
would likely help marginally, the model is not leaving large amounts of
performance on the table purely for lack of data.

## Serialization (Task 8)
`best_model.pkl` (the GridSearchCV best pipeline, including its own
`SimpleImputer` + `StandardScaler` + `RandomForestClassifier` steps) was
saved with `joblib.dump(..., compress=6)`. Compression was necessary
because GitHub's web-based drag-and-drop uploader caps individual files at
25MB, and the uncompressed pipeline (200 fully-grown trees) was ~42MB;
`compress=6` shrinks it to **~7MB with byte-for-byte identical predictions**
(verified directly: `predict_proba()` on the same rows before and after
compression returns identical values). The reload-and-predict block
(`joblib.load('best_model.pkl')` → `.predict()`/`.predict_proba()` on two
held-out test rows) ran without error, correctly predicting class 0
(probability 0.755 vs 0.245) for one row and class 1 (probability 1.000)
for the other.

## Summary comparison table and recommendation
| Model | CV Mean AUC | CV Std AUC | Test AUC |
|---|---|---|---|
| LogisticRegression | 0.9982 | 0.0002 | 0.9979 |
| DecisionTree(depth=5) | 0.9928 | 0.0005 | 0.9931 |
| RandomForest | 0.9972 | 0.0002 | 0.9973 |
| GradientBoosting | 0.9969 | 0.0003 | 0.9974 |
| RandomForest (GridSearch best) | 0.9978 | — | **0.9980** |

**Recommendation: the tuned Random Forest pipeline (`best_model.pkl`).**
All five models are extremely close (within ~0.005 AUC of each other) —
effectively tied given the cross-validation standard deviations of
0.0002–0.0005. The tuned Random Forest pipeline is preferred over plain
Logistic Regression (which had a marginally higher plain-CV mean before
tuning) because it (a) achieved the single highest test-set AUC of any
model built across Parts 2–3 (0.9980), (b) is packaged as one serialized
`Pipeline` object that already includes imputation and scaling, so it can
be deployed and re-run on raw, unscaled feature rows with no external
preprocessing code required, and (c) provides interpretable feature
importances (Task 4) that are directly useful for the client's business
questions about which diamond attributes drive price, which a plain
logistic regression's coefficients only partially capture given the
correlated size features.

## Files
- `part3_ensembles.ipynb` — full notebook (all tasks, all real outputs
  already run and saved in the file).
- `cleaned_data.csv` — copy of Part 1's cleaned dataset (needed to run this
  notebook standalone).
- `best_model.pkl` — final serialized pipeline, compressed (`compress=6`,
  ~7MB, down from ~42MB uncompressed — needed to stay under GitHub's 25MB
  web-upload limit).
