# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis

## Dataset choice and justification
**Diamonds dataset**, loaded via `seaborn.load_dataset("diamonds")`.
53,940 rows, 7 numeric columns (`carat, depth, table, price, x, y, z`) and
3 categorical columns (`cut, color, clarity`) — comfortably exceeds the
brief's minimum of 500 rows / 5 numeric columns / 2 categorical columns.
`price` is a natural continuous target for regression, and is binarized at
its median for the classification task carried out in Part 2.

## Steps taken (`part1_eda.ipynb`)
1. Loaded the dataset, printed `.head()`, `.dtypes`, `.shape` — **53,940 rows
   × 10 columns**: 7 numeric, 3 category dtype.
2. Null analysis: `isnull().sum()` and the percentage table.
3. Duplicate detection and removal.
4. Data type check.
5. Descriptive statistics (`describe()`) and skewness per numeric column.
6. IQR-based outlier detection on `carat` and `price`.
7. Five visualizations: line plot, bar chart, histogram, scatter plot, box
   plot.
8. Correlation heat map + highest-correlation pair.
9. (a) Mean/median comparison for the two highest-skew columns, (b)
   Spearman vs Pearson correlation comparison, (c) grouped aggregation.
10. Saved `cleaned_data.csv` (53,794 rows after de-duplication).

## Findings

### Null values (Task 2)
Every column showed **0.0% nulls** — `(df.isnull().sum() / df.shape[0]) * 100`
returned 0 for all 10 columns, so no column exceeded the 20% threshold and
no median-fill was necessary. Had nulls been present in numeric columns,
**median** (not mean) would have been the right choice, because — as the
skewness results below show — several numeric columns in this dataset
(`price`, `y`) are meaningfully right-skewed: the mean of a skewed column is
pulled toward the long tail by a minority of extreme values, while the
median reflects the typical row and stays robust to that skew, so
median-imputation avoids biasing the filled values toward outliers.

### Duplicates (Task 3)
`df.duplicated().sum()` found **146 duplicate rows**. These were removed
with `drop_duplicates(inplace=True)`, taking the dataset from 53,940 to
**53,794 rows**. Re-running `duplicated().sum()` confirmed 0 remaining
duplicates. Since the removed rows were exact copies of rows already
counted in the null check (and the dataset had 0 nulls throughout), removing
them did not change any column's null percentage — it stayed at 0.0%
everywhere before and after.

### Data type correction (Task 4)
On inspection, **no column had an incorrect dtype**: `carat, depth, table,
price, x, y, z` were already numeric (`float64`/`int64`), and `cut, color,
clarity` were already `category` dtype on load (seaborn's diamonds dataset
ships with these pre-typed). Because there was nothing to correct, the
notebook explicitly demonstrates the *method* that would be used if a
numeric column had arrived as strings (`pd.to_numeric(..., errors='coerce')`
to coerce it, with any resulting nulls filled by median), and re-applies
`astype('category')` to `cut/color/clarity` for completeness. Memory usage
before and after this step was **identical (3,606,010 bytes → 3,606,010
bytes, 0 bytes saved)**, confirming these columns were already stored
efficiently as `category` and no further reduction was possible.

### Descriptive statistics and skewness (Task 5)
Skew per numeric column:

| Column | Skew |
|---|---|
| carat | 1.1137 |
| depth | -0.1143 |
| table | 0.7922 |
| price | 1.6182 |
| x | 0.3796 |
| **y** | **2.4458** |
| z | 1.5290 |

**`y`** has the highest absolute skewness (2.4458), followed by `price`
(1.6182). Values between -1 and 1 are treated as not meaningfully skewed;
`carat`, `price`, `y`, and `z` all exceed 1, meaning their distributions are
stretched substantially to one side. `y`'s skew is **positive**: the bulk of
diamonds have a similar, modest width, but a small number of unusually wide
stones stretch the right tail out, pulling the mean above the median. The
consequence for imputation: filling missing `y` values with the **mean**
would systematically over-estimate a "typical" diamond's width, since the
mean here is inflated by a handful of outliers — median is the safer choice.

### IQR outliers (Task 6)
| Column | Q1 | Q3 | IQR | Lower bound | Upper bound | Outliers |
|---|---|---|---|---|---|---|
| carat | 0.40 | 1.04 | 0.64 | -0.56 | 2.00 | **1,873** |
| price | 951.00 | 5326.75 | 4375.75 | -5612.62 | 11890.38 | **3,523** |

These outliers were **not dropped**. Both columns' outliers sit entirely on
the high side (a diamond can't have negative carat or price), representing
genuinely large/expensive stones rather than data-entry errors. **Handling
in Part 2**: these rows were **retained as-is** — no capping or removal was
applied before modeling. This is a reasonable choice because (a) removing
~3,500+ rows (6.5% of the data) purely for being "expensive" would bias the
model against ever learning to price high-end diamonds, and (b) the
regression/classification models trained in Part 2 (Linear/Ridge/Logistic
Regression) were evaluated on standardized features, which reduces (though
doesn't eliminate) the leverage a small number of large values can exert;
the very strong R² (0.92) and AUC (0.998) achieved in Part 2 confirm the
outliers did not need special handling to get a well-fitting model on this
dataset.

### Visualizations (Task 7)
1. **Line plot** — `price` across the row index for the full dataset: no
   trend (row order is not chronological here), just the natural
   row-to-row variability of the raw file.
2. **Bar chart** — mean `price` by `cut`: **Premium** has the highest mean
   price (4,583.50) despite not being the "best" cut label, followed by
   Fair (4,341.95) — a known quirk of this dataset, since cut quality and
   carat size are not strongly linked, and carat dominates price.
3. **Histogram** — `price` (bins=20): a strongly **right-skewed**,
   unimodal distribution — most diamonds are clustered at the lower end of
   the price range, with a long thin tail stretching toward the most
   expensive stones, visually matching the skew=1.618 finding.
4. **Scatter plot** — `carat` vs `price`: a clear, strong **positive**
   relationship — price rises steeply as carat increases, consistent with
   the Task 8 Pearson correlation of 0.9215 between the two.
5. **Box plot** — `price` by `cut`: medians are broadly similar across cut
   categories, but **Premium** shows both a higher median and visibly
   larger spread/more high-price outliers than the other cuts, consistent
   with it having both the highest mean (4,583.50) and the numbers
   reported in Task 9c below.

### Correlation heat map (Task 8)
Full Pearson correlation matrix was computed and visualized with
`sns.heatmap(annot=True)`. The **highest absolute correlation pair is
`carat` and `x` (r ≈ 0.9754)**. This is not a meaningfully causal
relationship between two independent measurements — both `carat` (weight)
and `x` (length in mm) are driven by the same underlying **physical size**
of the diamond, so they move together almost perfectly by construction; a
plausible alternative explanation is simply that carat weight and every
physical dimension (`x`, `y`, `z`) are all different ways of measuring the
same underlying "how big is this stone" property, not that one causes the
other.

### Task 9a — mean vs median for the two most-skewed columns
| Column | Mean | Median |
|---|---|---|
| y | 5.734653 | 5.710000 |
| price | 3933.065082 | 2401.000000 |

(These are the two columns identified in Task 5 as having the highest
absolute skewness: y=2.4458, price=1.6182.) Both columns are positively
skewed, and in both cases the mean sits above the median — most visibly for
`price`, where the mean ($3,933) is over $1,500 higher than the median
($2,401), confirming that a minority of expensive diamonds pull the average
upward. Since this dataset had 0 nulls to begin with, no `fillna()` was
actually required on these columns — but had nulls existed, median would
be the justified choice for the same reason given in Task 5.

### Task 9b — Spearman vs Pearson
Top 3 pairs by |Spearman − Pearson| (from the full difference table,
computed across all 21 numeric column pairs):

| Pair | Pearson | Spearman | \|diff\| |
|---|---|---|---|
| price vs y | 0.8654 | 0.9628 | **0.0974** |
| price vs z | 0.8612 | 0.9573 | 0.0961 |
| price vs x | 0.8845 | 0.9633 | 0.0788 |

In all three of the top pairs, **|Spearman| > |Pearson|**, indicating the
relationship between `price` and each physical dimension (`x`, `y`, `z`) is
**monotonic but non-linear**: price consistently rises as size increases,
but not in direct proportion — consistent with how diamond pricing formulas
scale super-linearly with size/carat rather than linearly. For **Part 2
feature selection**, this means Spearman is the more informative measure
for judging how strongly these size-related features relate to price,
since it captures the true strength of the monotonic relationship that
Pearson understates; Pearson remains adequate for pairs where the two
measures are already close (e.g. `carat` vs `y`: 0.9519 vs 0.9956, still a
non-trivial gap, but smaller in absolute terms than the price pairs above).

### Task 9c — grouped aggregation (`price` by `cut`)
| cut | mean | std | count |
|---|---|---|---|
| Ideal | 3462.75 | 3810.93 | 21,488 |
| **Premium** | **4583.50** | **4348.05** | 13,748 |
| Very Good | 3981.02 | 3934.81 | 12,069 |
| Good | 3919.12 | 3671.07 | 4,891 |
| Fair | 4341.95 | 3540.12 | 1,598 |

- Highest mean group: **Premium**. Highest std group: **Premium** (also the
  highest mean group here — the same cut has both the highest average price
  and the widest spread of prices).
- Within-group standard deviation is large relative to the between-group
  mean differences for every cut category (e.g. Ideal's std of 3810.93 is
  larger than its own mean of 3462.75) — this **is a concern**: knowing
  only a diamond's cut still leaves a very wide range of plausible prices,
  so `cut` alone is not sufficient to predict price reliably; it needs to
  be combined with `carat` and the other size/clarity features.
- Ratio of highest to lowest group mean = 4583.50 / 3462.75 = **1.3237**.
  This is a modest (~32%) gap, suggesting `cut` carries some real
  predictive signal but far less than `carat`/size (which correlate with
  price at r≈0.88–0.92).

## Files
- `part1_eda.ipynb` — full notebook (all tasks, all outputs, all 5 plots +
  heat map rendered inline).
- `cleaned_data.csv` — the cleaned dataset (53,794 rows), used by Part 2.
