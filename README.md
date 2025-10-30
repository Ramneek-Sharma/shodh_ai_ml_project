# shodh_ai_ml_project

This project builds and compares two machine learning models for a fintech loan approval problem. The objective is to move beyond simple risk prediction and develop an intelligent policy that maximizes financial return.

The core of the project compares:
1.  **Model 1 (Task 2): A Deep Learning Classifier** trained to predict the *probability of default*.
2.  **Model 2 (Task 3): A Deep Learning Regressor (as an Offline RL Agent)** trained to predict the *exact profit or loss in dollars*.

## 🚀 Core Finding

While the standard **Risk Model (Task 2)** was a good predictor (74.5% AUC), it proved to be a poor business tool. It approved **4,902 loans that defaulted**, which would lead to massive financial losses.

The **Profit Model (Task 3)**, which was trained to maximize reward, learned a much safer and more profitable policy. It became extremely conservative, **avoiding 93% of all defaults** (only approving 339) and generating an **estimated total profit of $1.62 million** on the test set.

This demonstrates that a model trained to optimize for **profit** creates a far superior business policy than a model trained only to optimize for **risk**.

## 📊 Model Results Comparison

| Metric | Model 1: "Risk Predictor" (Task 2) | Model 2: "Profit Maximizer" (Task 3) |
| :--- | :--- | :--- |
| **Business Goal** | Minimize Risk (Predict Default) | Maximize Profit (Predict Reward) |
| **Key Metric** | AUC = 0.7450 | Est. Policy Value = **$45.83** (avg. per loan) |
| **Total Profit** | *(Not Optimized)* | **$1,622,439.09** |
| | | |
| **Confusion Matrix** | **(Predicted)** | **(Predicted)** |
| **(Actual)** | **Deny (0)** | **Approve (1)** | **Deny (0)** | **Approve (1)** |
| **Paid (0)** | 19,085 | 9,114 | **22,019** | 6,180 |
| **Default (1)** | 2,303 | **4,902** | **6,866** | **339** |

## 📂 Project Structure

This repository contains three main notebooks, which are designed to be run in order.

1.  **`01_Preprocessing.ipynb`**
    * Loads the raw `accepted_2007_to_2018.csv` data (using a sample for speed).
    * Performs all data cleaning, feature engineering, and removal of "leaky" features.
    * Creates the binary target `is_default` and the `reward` (profit/loss) target.
    * Splits, scales, and saves the final clean datasets (`X_train_scaled.csv`, `y_train.csv`, etc.) to be used by the other notebooks.

2.  **`02_DL_Model.ipynb`**
    * Implements **Task 2**.
    * Loads the clean, scaled data from Notebook 1.
    * Builds and trains a TensorFlow/Keras deep learning classifier to predict the probability of default.
    * Evaluates the model and reports the **AUC** and **F1-Score**.

3.  **`03_Profit_Model.ipynb`**
    * Implements **Task 3** (Offline RL).
    * Loads the clean, scaled data from Notebook 1.
    * Builds and trains a TensorFlow/Keras deep learning regressor to predict the exact reward (profit/loss) for approving a loan.
    * Derives a policy (Approve if `predicted_profit > 0`) and evaluates it.
    * Reports the **Estimated Policy Value**.


## Requirements

The main libraries used for this project are:
