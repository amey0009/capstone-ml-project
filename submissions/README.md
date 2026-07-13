# Applied AI & ML Capstone Project — End-to-End Data Intelligence System

## Repository structure
**Single repository, one folder per Part** (stated choice, as permitted by
the brief).

```
├── part1/
│   ├── part1_eda.ipynb
│   ├── cleaned_data.csv
│   └── README.md
├── part2/
│   ├── part2_models.ipynb
│   └── README.md
├── part3/
│   ├── part3_ensembles.ipynb
│   ├── cleaned_data.csv
│   ├── best_model.pkl
│   └── README.md
├── part4/
│   ├── part4_llm_explain.ipynb
│   ├── best_model.pkl
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
| 3 | `part3/` | Done | Decision Trees, Random Forest, Gradient Boosting, GridSearchCV pipeline, feature ablation, learning curve, serialized `best_model.pkl` |
| 4 | `part4/` | Done | LLM-powered feature — Track C: model prediction explanation pipeline, JSON schema validation, PII guardrail, temperature A/B comparison |

**All 4 parts are complete.**

## How to reproduce
Open `part1/part1_eda.ipynb` in Jupyter/Colab and run all cells top-to-bottom
first (this writes `cleaned_data.csv`). Then open `part2/part2_models.ipynb`
in the same folder as that `cleaned_data.csv` (or update the path in its
first cell) and run all cells top-to-bottom. `part3/part3_ensembles.ipynb`
also needs `cleaned_data.csv` in its working directory (a copy is included
in `part3/`) and recreates Part 2's preprocessing internally before training
the ensemble models and saving `best_model.pkl`. `part4/part4_llm_explain.ipynb`
needs `best_model.pkl` in its working directory (a copy is included in
`part4/`) and runs standalone from there — it does not need `cleaned_data.csv`.

Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`,
`joblib`, `jsonschema`, `requests`.

## Note on Part 4 and network access
Part 4's `call_llm()` is written exactly to the OpenAI/OpenRouter-compatible
spec required by the brief and reads its API key from an environment
variable only (never hardcoded). It defaults to a documented local mock
mode (`LLM_MOCK=1`) because the environment used to build/run this
submission has no outbound network access to third-party LLM providers.
Setting `LLM_MOCK=0` plus real `LLM_API_URL`/`LLM_API_KEY` env vars runs the
identical pipeline against a live model with no other code changes. See
`part4/README.md` for full detail.
