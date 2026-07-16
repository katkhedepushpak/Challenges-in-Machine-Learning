# Challenges in Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A curated collection of machine learning coursework challenges exploring core concepts in classification, model evaluation, and distribution shift adaptation. Each assignment tackles a distinct real-world ML problem — from handling missing astronomical data to cross-validating activity recognition models and correcting for label shift — providing hands-on implementations that go beyond textbook theory.

---

## Highlights

- **Assignment 1 — Swiss Cheese in Space:** Trains an SVC classifier on astronomical survey data (`cfhtlens.csv`) and benchmarks five strategies for handling missing values: abstaining, majority-class imputation, feature dropping, mean imputation, and median imputation.
- **Assignment 2 — Give Your Models a Grade:** Evaluates four classifiers (Decision Tree, Random Forest, KNN, MLP) on an activity recognition dataset across five cross-validation strategies, including stratified and groupwise splits, to surface overfitting and data-leakage risks.
- **Assignment 3 — Label Shift Adaptation:** Implements the label shift correction method of Lipton et al. (2018) and Shrikumar & Kundaje (2020), reweighting test posteriors by inverting the confusion matrix to handle train/test distribution mismatch.

---

## Tech Stack

- **Language:** Python 3
- **ML / Data:** scikit-learn, NumPy, pandas, SciPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Environment:** Conda (conda 22.11.1+)

---

## Getting Started

### Prerequisites

- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.x

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/katkhedepushpak/Challenges-in-Machine-Learning.git
   cd Challenges-in-Machine-Learning
   ```

2. Create and activate a Conda environment using the provided requirements file:
   ```bash
   conda create --name ml-challenges --file Assignment1/requirements.txt
   conda activate ml-challenges
   ```

   > **Note:** The requirements file reflects a full Anaconda environment snapshot. For a lighter install, you can instead run:
   > ```bash
   > pip install scikit-learn pandas numpy matplotlib seaborn scipy plotly
   > ```

---

## Usage

Each assignment is self-contained. Navigate to the relevant directory and run the script:

### Assignment 1 — Missing Value Strategies
```bash
cd Assignment1
python train_eval.py
```
Expects `cfhtlens.csv` to be present in the `Assignment1/` directory.

### Assignment 2 — Classifier Cross-Validation
```bash
cd "Assignment 2"
python activity_eval.py
```

### Assignment 3 — Label Shift Adaptation
```bash
cd "Assignment 3"
python label_shift_adaptation.py
```

---

## Repository Structure

```
Challenges-in-Machine-Learning/
├── Assignment1/
│   ├── train_eval.py          # SVC + missing value strategies on cfhtlens.csv
│   └── requirements.txt       # Conda environment snapshot
├── Assignment 2/
│   └── activity_eval.py       # Multi-classifier cross-validation on activity data
└── Assignment 3/
    └── label_shift_adaptation.py  # Label shift correction (Lipton et al. 2018)
```

---

## References

- Lipton, Z., Wang, Y-X., & Smola, A. (2018). *Detecting and Correcting for Label Shift with Black Box Predictors.* ICML.
- Shrikumar, A., & Kundaje, A. (2020). *Relating label shift correction to black box prediction.*

---

## Author

Built by [Pushpak Vijay Katkhede](https://katkhedepushpak.github.io)
