# Applied AI & ML Capstone Project — End-to-End Data Intelligence System

## Repository structure
**Single repository, one folder per Part** (stated choice, as permitted by
the brief). Parts 3 and 4 will be added as `part3/` and `part4/` folders
following the same pattern once completed.

```
├── part1/
│   ├── part1_eda.ipynb
│   ├── cleaned_data.csv
│   └── README.md
├── part2/
│   ├── part2_models.ipynb
│   └── README.md
└── README.md   <- this file
```

## Dataset
**Diamonds** — loaded directly via `seaborn.load_dataset("diamonds")`
(a well-known public diamond pricing dataset). 53,940 rows before cleaning
(53,794 after duplicate removal — see `part1/README.md`), 7 numeric columns
(`carat, depth, table, price, x, y, z`) and 3 categorical columns
(`cut, color, clarity`), well above the brief's minimum of 500 rows / 5
numeric / 2 categorical columns. `price` is the natural continuous
regression target and is binarized at its median for the classification
task in Part 2.

## Part index
| Part | Folder | Status | Summary |
|---|---|---|---|
| 1 | `part1/` | Done | Load, null/duplicate check, dtype check, EDA, 5 plots + heat map, correlation analysis, `cleaned_data.csv` |
| 2 | `part2/` | Done | Categorical encoding, leak-free split/scaling, Linear/Ridge regression, Logistic Regression, threshold + regularization + bootstrap analysis |
| 3 | `part3/` | Not yet submitted | Ensembles, tuning, pipeline |
| 4 | `part4/` | Not yet submitted | LLM-powered feature |

## How to reproduce
Open `part1/part1_eda.ipynb` in Jupyter/Colab and run all cells top-to-bottom
first (this writes `cleaned_data.csv`). Then open `part2/part2_models.ipynb`
in the same folder as that `cleaned_data.csv` (or update the path in its
first cell) and run all cells top-to-bottom.

Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
