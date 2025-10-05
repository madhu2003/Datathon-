# Fetal Health Classification using Cardiotocography Data

## Team Members (TM-213)

Li Qiyue

Madhumita Jambulingam

Soon Si Qi, Kelly

## 1. Objective

The goal of this project is to develop a machine learning model that accurately classifies fetal health status as **Normal**, **Suspect**, or **Pathologic** based on features extracted from Cardiotocography (CTG) examinations. This addresses the clinical need for reliable, automated monitoring to support timely, life-saving interventions during labor.

## Link to video demo
https://drive.google.com/file/d/17iwvstL6cRKekfgkgy2hdFx56t40_CLT/view?usp=share_link


## 2. Models Explored

To find the optimal balance between performance and interpretability, we are training and evaluating the following models:

1. **Logistic Regression**

2. **Random Forest:**
 
3. **XGBoost:** 

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

#### logistic regression
the EDA is in (data_exploration/logistic_regress/data_analysis.ipynb)
this shows how we handled the data for this model 

For test and train, it is in (log_reg_traintest/testing.ipynb)
This gets us the capabilities and scores for this model.

#### randomForest
The EDA is in (data_exploration/random forest/exploratory data analysis.ipynb)

For test and train of the model, it is in 

#### XGBoost:
The EDA is in the data_exploration folder > XGBoost > EDA. This contains the data loading, cleaning and preprocessing steps. 
The model training code is in the Train_XGBoost file. 
The inference for this model is the Inference.py file. 
These have to be run in order to get the proper results. 








