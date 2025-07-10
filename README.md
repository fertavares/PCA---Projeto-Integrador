# 🏙️ PCA Analysis of School Accessibility by Neighborhood

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge\&logo=pandas\&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge\&logo=numpy\&logoColor=white)](https://numpy.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge\&logo=seaborn\&logoColor=white)](https://seaborn.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge\&logo=matplotlib\&logoColor=white)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)

---

## 🔍 Project Overview

This Python project performs Principal Component Analysis (PCA) on school accessibility data by neighborhood using a dataset loaded from a CSV file. It cleans the data, encodes categorical neighborhoods numerically, applies PCA with scaling, and visualizes the results with a scatter plot including PCA component vectors.

---

## ⚙️ Features

* Reads and cleans dataset from CSV with automatic encoding handling (UTF-8 / ISO-8859-1)
* Selects relevant columns: Neighborhood name, Number of accessible rooms, Number of special education classes
* Encodes neighborhood names into numeric codes for PCA
* Standardizes data before PCA
* Calculates covariance matrix, eigenvalues, and eigenvectors
* Performs PCA with 2 principal components
* Visualizes PCA results with neighborhood coloring and component vectors on scatter plot

---

## 📦 Requirements

* Python 3.7+
* pandas
* numpy
* seaborn
* matplotlib
* scikit-learn

Install dependencies with:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

## 🚀 Usage

1. Set the path to your dataset CSV in the `caminho` variable in the script or main block.
2. Run the script.
3. The program will:

   * Load and clean the dataset
   * Perform PCA on selected features
   * Plot the PCA scatter plot with components

---

## 📁 Project Structure

```
PCAAcessibilidade.py            # Main Python script/class performing PCA and visualization
DataFrame/
└── DFmicrodados2021.csv        # Example input dataset (not included)
```

---

## 📝 Notes

* The script automatically handles CSV encoding issues by trying UTF-8 first, then ISO-8859-1.
* PCA components are plotted as arrows on the scatter plot for interpretability.
* Neighborhood names are displayed as color-coded groups.

---
