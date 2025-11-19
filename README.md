# DEEP-LEARNING-PROJECT

Project Overview

This project implements an advanced multivariate time series forecasting system using deep learning models enriched with attention mechanisms. Unlike traditional LSTMs or ARIMA models, the attention-based model (Transformer Encoder-Decoder) provides both higher accuracy and interpretability by highlighting which time steps and features the model focuses on during prediction.

The project includes:

A synthetic-but-realistic multivariate dataset (5 features + target; 1200 timesteps).

A full Transformer-based forecasting model with multi-head self-attention.

Rolling-origin time series cross-validation for reliable evaluation.

Baseline model comparison using LSTM.

Attention-weight visualization for model interpretability.

Clean, modular, production-quality Python code.

📂 Project Structure
project/
│
├── time_series_dataset.csv        # Generated dataset (1200 rows × 7 columns)
├── model_training.py              # Deep learning models (Transformer + LSTM)
├── attention_analysis.py          # Attention weight extraction & visualization
├── utils.py                       # Helper functions (scaling, data loaders, CV)
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies

📊 Dataset Description

The dataset (time_series_dataset.csv) contains:

1200 timestamped rows (synthetic but realistic)

5 correlated features

Seasonal + trend patterns

Heteroscedastic Gaussian noise

A target variable derived using:

weighted feature mixture

seasonal effect

noise injection

Columns:

timestamp

feat_1, feat_2, feat_3, feat_4, feat_5

target

🧠 Model Summary
1. Transformer Encoder-Decoder with Multi-Head Attention

Positional Encoding

Multi-Head Self-Attention

Feedforward Blocks

Sequence-to-Sequence Forecasting

Adam optimizer + MSE loss

2. Baseline LSTM Model

2-layer LSTM

Dense output layer

Serves as benchmark

🧪 Training & Validation
Rolling-Origin Cross-Validation

The split moves forward step-by-step:

Train → Validate
Train → Validate
Train → Validate (rolling forward)


This ensures:

No leakage

True forecasting behavior

Reliable hyperparameter selection

📈 Evaluation Metrics

Both models are evaluated using:

RMSE – Root Mean Squared Error

MAE – Mean Absolute Error

SMAPE – Symmetric Mean Absolute Percentage Error

A full text-based comparison report is generated after training.

🎯 Deliverables Included

✔ Full Python code (dataset generation, models, training, evaluation)
✔ Transformer + LSTM models
✔ Rolling CV implementation
✔ Dataset file (.csv)
✔ Attention visualization code
✔ Final comparison report text
✔ Attention interpretability summary

📥 How to Run the Project
1. Install Dependencies
pip install -r requirements.txt

2. Run Training
python model_training.py

3. Generate Attention Visualizations
python attention_analysis.py

🔍 Attention Interpretation

The project includes visualizations and textual explanations for:

Which features matter most

Which time steps influence predictions

How attention changes across forecast horizons

Example insights:

Feature 3 & Feature 5 strongly influence longer-horizon forecasts.

The model pays high attention to seasonal peaks around 30–40 timesteps prior.

Sudden noise spikes reduce attention weights on recent points.

📌 Requirements
numpy
pandas
matplotlib
torch
scikit-learn
seaborn
