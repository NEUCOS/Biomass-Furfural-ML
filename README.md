# Biomass-Furfural-ML

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Data-Driven Optimization and Mechanistic Modeling of Biomass Valorization to Furfural: Machine Learning Evaluation of Catalytic Systems and Operational Conditions for Humin Minimization**

---

## 📌 Overview

This repository contains the open-source code, curated experimental dataset, machine learning pipeline, and optimization scripts supporting the research paper:

> **Data-Driven Optimization and Mechanistic Modeling of Biomass Valorization to Furfural: Machine Learning Evaluation of Catalytic Systems and Operational Conditions for Humin Minimization**  
> *Mei Yang, Hongbin Sun, Guangwen Xu*  
> Key Laboratory of Resources Chemicals and Materials (Shenyang University of Chemical Technology), Ministry of Education, Shenyang, 110142, China.

---

## 📂 Repository Structure

`	ext
Biomass-Furfural-ML/
│
├── README.md               # Project documentation and reproduction guide
├── requirements.txt        # Python library dependencies
├── data/
│   ├── Furfural_Biomass_ML_Dataset.csv   # Structured reaction dataset (N=270, 11 features)
│   ├── Furfural_Biomass_ML_Dataset.xlsx  # Excel format with metadata
│   └── model_evaluation_metrics.csv      # 10-fold CV performance metrics across 8 algorithms
├── src/
│   └── run_ml_pipeline.py  # End-to-end ML training, 10-fold CV, feature importance & 2D/3D plotting
└── figures/
    ├── Fig1_Model_Evaluation_Comparison.png  # Algorithm performance comparison (R² & RMSE)
    ├── Fig2_Parity_Plot_Predicted_vs_Actual.png # GBDT Parity plot with ±10% error envelope
    ├── Fig3_Feature_Importance_Analysis.png    # Permutation feature importance ranking
    ├── Fig4_Response_Surface_Optimization.png # 2D isocontour and 3D response surface maps
    └── Fig5_Pareto_Frontier_Yield_vs_Humin.png # Acid ratio & solvent synergy analysis
`

---

## 🔬 Key Research Findings

1. **Machine Learning Benchmarking**: Eight supervised regression algorithms were systematically evaluated under 10-fold cross-validation. **Gradient Boosting Regressor (GBDT)** demonstrated superior accuracy (**10-Fold CV ^2 = 0.8594 \pm 0.0388$, Test ^2 = 0.8677$, RMSE = 8.785%**), outperforming linear models (^2 = 0.6099$) by capturing complex non-linear reaction kinetic trade-offs.
2. **Feature Importance**: Permutation importance revealed that **Reaction Temperature**, **Residence Time**, and **Brønsted-to-Lewis (B:L) Acid Molar Ratio** dominate furfural yield and humin byproduct suppression.
3. **Optimal Operating Condition Window**:
   - **Temperature**: 175 – 185 °C
   - **Residence Time**: 60 – 90 min (Batch) / 5 – 15 min (Flow)
   - **Catalyst System**: Brønsted-Lewis Mixed Acid ( = 1.8 : 1$)
   - **Solvent System**: Biphasic Water / MIBK (/O = 0.33$, 15 wt% NaCl)
   - **Projected Furfural Yield**: $> 80.0\%$ with minimal humin formation ($< 12.0\%$).

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

`ash
git clone https://github.com/NEUCOS/Biomass-Furfural-ML.git
cd Biomass-Furfural-ML
pip install -r requirements.txt
`

### 2. Reproduce Models & Figures

Run the end-to-end machine learning pipeline:

`ash
python src/run_ml_pipeline.py
`

This script will:
- Load the structured reaction dataset from data/Furfural_Biomass_ML_Dataset.csv;
- Perform 10-fold cross-validation across all 8 machine learning models;
- Evaluate feature importances via permutation tests;
- Execute 2D/3D response surface optimization;
- Export high-resolution figures to igures/ and metrics to data/model_evaluation_metrics.csv.

---

## 📋 Citation

If you find this repository, dataset, or code useful for your research, please cite our work:

`ibtex
@article{yang2026datadriven,
  title={Data-Driven Optimization and Mechanistic Modeling of Biomass Valorization to Furfural: Machine Learning Evaluation of Catalytic Systems and Operational Conditions for Humin Minimization},
  author={Yang, Mei and Sun, Hongbin and Xu, Guangwen},
  journal={Chemical Engineering Journal / Green Chemistry},
  year={2026}
}
`

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
