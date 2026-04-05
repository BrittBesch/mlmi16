# mlmi16 — Advanced HCI (CW3)

Coursework materials for the **personalisation & inspiration** user study: a between-subjects comparison of **personalised** vs **generic** LLM suggestions across **travel** and **cooking** scenarios, with survey and behavioural outcomes analysed in Python.

## Project structure

```
mlmi16/
├── README.md
├── experiment/
│   └── index.html                    # Study web app (HTML/CSS/JS)
├── data/
│   └── user_study_MLMI16.csv         # Raw participant exports (condition, Likert, save choice, demographics)
└── analyses/
    ├── analyse.py                    # Cleaning, composites, assumption checks, hypothesis tests, plots
    ├── processed_data.csv            # Generated — cleaned dataframe with composites (run `analyse.py`)
    └── plots/                        # Generated — result figures (PDF)
        ├── fig1_dvs_by_condition_scenario.pdf
        └── fig2_manipulation_check.pdf
```

## Running the analysis

From the repository root (this directory):

```bash
python3 analyses/analyse.py
```

**Dependencies:** `pandas`, `numpy`, `scipy`, `matplotlib`. For full output (logistic regression, ANCOVA, mixed ANOVA), also install `statsmodels` and `pingouin`:

```bash
pip install pandas numpy scipy matplotlib statsmodels pingouin
```

The script reads `data/user_study_MLMI16.csv`, prints results to the terminal, saves `analyses/processed_data.csv`, and writes plots under `analyses/plots/`.