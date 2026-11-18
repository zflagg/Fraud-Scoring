# Real-Time Fraud Scoring

A real-time fraud detection system: models trained on the IEEE-CIS dataset,
served behind a low-latency API (target: p99 under 50 ms).

## Results

| Model | PR-AUC | Recall @ 1% FPR | Data |
|---|---|---|---|
| Logistic regression | 0.169 | 0.154 | 100k slice |
| XGBoost | 0.457 | 0.436 | 100k slice |
| FraudNet (MLP + embeddings) | 0.xx | 0.xx | 100k slice |

## Data

Download `train_transaction.csv` from the
[IEEE-CIS Fraud Detection competition](https://www.kaggle.com/c/ieee-fraud-detection/data)
on Kaggle and place it in the project root.

## Setup

    conda env create -f environment.yml
    conda activate fraudscore

## Usage

    python models.py

## Project structure

    config.py          column lists and settings
    preprocessing.py   time split, metrics, Preprocessor
    models.py          trains and evaluates the models

## Notes

Currently in the process of splitting up certain functions and classes into their own files