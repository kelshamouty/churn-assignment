# Customer Churn Prediction

Reproducible end-to-end pipeline for the take-home exercise.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run end-to-end

```bash
python -m src.eda              # 1) EDA figures → figures/eda/
python -m src.modeling         # 2) Model figures + fitted pipeline → figures/modeling/, artifacts/
python -m src.generate_reports # 3) HTML reports → reports/
```

All outputs (figures and reports) are already included in the repository — re-running is only needed if you want to reproduce from scratch.

Total runtime ≈ 4 minutes. Steps 1 and 2 can run independently; step 3 requires both.

## File map

<pre>
.
├── ecommerce_customer_churn_dataset.csv   raw data
├── Instructions.md                        the brief
├── requirements.txt                       pinned dependencies
├── README.md                              this file
│
├── src/
│   ├── eda.py                             EDA figures
│   ├── eda_supplement.py                  dirty values · missingness signal · feature relevance
│   ├── preprocess.py                      cleaning · feature engineering · transformers
│   ├── modeling.py                        CV · test set eval · model figures
│   └── generate_reports.py               assembles HTML reports (no intermediate files)
│
├── figures/
│   ├── eda/                               EDA charts (PNG)
│   └── modeling/                          modelling charts (PNG)
│
└── reports/
    ├── eda_report.html                    self-contained EDA report (figures embedded)
    └── modelling_report.html             self-contained modelling report (figures embedded)
</pre>
