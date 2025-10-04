# Fetal Health Classification using Cardiotocography Data

**Project for Datathon 2025**

## 1. Objective

The goal of this project is to develop a machine learning model that accurately classifies fetal health status as **Normal**, **Suspect**, or **Pathologic** based on features extracted from Cardiotocography (CTG) examinations. This addresses the clinical need for reliable, automated monitoring to support timely, life-saving interventions during labor, as outlined in the Datathon 2025 booklet.

## 2. Dataset

This project utilizes the **UCI Cardiotocography Dataset**, a widely studied collection of over 2,000 real CTG traces that have been classified by expert obstetricians.

## 3. Methodology Overview

Our workflow is designed to build a robust and trustworthy predictive model, focusing on data quality and rigorous evaluation.

* **Data Cleaning & Preprocessing:** We performed several steps to ensure data quality, including removing duplicate rows, correcting skewed features using a Yeo-Johnson power transform, and implementing a selective outlier capping strategy to preserve critical clinical signals while reducing noise.

* **Leakage Prevention:** The `CLASS` feature was identified as a source of data leakage and was removed to ensure a realistic evaluation of our models' performance on raw CTG features.

* **Feature Engineering:** New, medically relevant features (e.g., `Deceleration_Severity`, `FHR_Range`) were created to provide the models with more insightful "clues."

* **Modeling:** We are exploring and comparing the performance of three different machine learning models to identify the most effective approach.

* **Evaluation:** Models are evaluated based on the official datathon metrics: **Balanced Accuracy** and **Macro F1-Score**.

## 4. Models Explored

To find the optimal balance between performance and interpretability, we are training and evaluating the following models:

1. **Logistic Regression:** A highly interpretable linear model that provides a strong, explainable baseline.

2. **Random Forest:** A powerful ensemble model known for its high accuracy and robustness to outliers.

3. **XGBoost:** An advanced gradient boosting model renowned for its state-of-the-art performance in data science competitions.

## 6. How to Run This Project

### Prerequisites

* Python 3.9+

* `pip` package manager

### Installation

1. Clone this repository to your local machine.

2. Open your terminal and navigate to the project's root directory (`datathon-2025/`).

3. Install the required dependencies by running:
pip install -r requirements.txt

### Running the Notebooks

To reproduce the results, run the Jupyter Notebooks in the directory in numerical order:

1. `01_log_regres.ipynb`: To perform initial data cleaning and save the processed data.

2. `02_log_regres.ipynb`: To train and evaluate the Logistic Regression model.

## 7. Team Members

LI QIYUE

MADHU

KELLY