# Advanced Time Series Forecasting with Attention based Transformers

## Overview

This project focuses on developing a complete forecasting pipeline using a decoder-only Transformer model for long-term, multivariate time series prediction. The goal is to explore the benefits of attention-based architectures compared to traditional methods such as LSTM and baseline statistical models. The project includes dataset generation, model implementation, hyperparameter tuning, walk-forward validation, and a final interpretability analysis.


## Objectives

1. Generate a synthetic multivariate time series dataset with realistic trend, seasonality, and noise patterns.
2. Implement a causal Transformer-based forecasting model tailored for sequence prediction tasks.
3. Perform hyperparameter tuning focusing on the attention mechanism.
4. Compare forecasting performance against an LSTM baseline using RMSE and MAE.
5. Use a rolling-window backtesting strategy for robust evaluation.
6. Provide a conceptual interpretability summary describing insights from attention weights.

## Project Structure

* Synthetic data generation script
* Transformer model implementation
* LSTM baseline model
* Hyperparameter tuning workflow
* Rolling-window backtesting module
* Evaluation and metrics reporting
* Final interpretability analysis

## Dataset

The dataset is programmatically generated using NumPy and SciPy.
It consists of five years of daily observations with at least three features exhibiting:

* Long-term trend
* Seasonal components
* Irregular noise patterns
* Correlation across features

This controlled dataset ensures that long-range dependencies are present for the Transformer to capture.

## Models

### Transformer (Decoder-Only)

* Custom positional encoding
* Causal mask for autoregressive forecasting
* Multi-head self-attention
* Tunable depth, width, and attention parameters

### LSTM Baseline

* Standard sequence-to-sequence architecture
* Serves as the benchmark for comparison

## Training and Tuning

Hyperparameters tuned include:

* Number of layers
* Number of attention heads
* Embedding dimensions
* Dropout rates

The tuning process evaluates multiple configurations to identify the most effective architecture. Both Transformer and LSTM were trained under comparable conditions to ensure fairness.

## Backtesting and Evaluation

A rolling-window (walk-forward) validation procedure is used. After each window shift, the models are retrained and evaluated on the next horizon.
Metrics reported:

* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)

This method simulates real-world deployment conditions and ensures reliability of the forecasting performance.

## Performance Summary

The Transformer model demonstrated superior performance over the LSTM baseline, particularly in scenarios requiring long-term dependency modeling. The attention mechanism enabled the model to dynamically weigh important time steps, improving predictive accuracy during seasonal shifts and trend changes.

# Final Interpretability Summary (Expected Deliverable)

The Transformer’s attention mechanism provided valuable insights into how the model prioritizes historical information during forecasting. Unlike recurrent models, which store information in hidden states that are difficult to analyze directly, attention weights reveal which time points contribute most to each prediction.

Key observations from attention weight patterns:

* The model consistently assigned higher attention to recurring seasonal positions, indicating its ability to recognize cyclical behavior in the synthetic dataset.
* Long-range dependencies, such as annual trends, influenced predictions far more strongly than immediate past values. This demonstrates the advantage of self-attention over LSTM architectures, which typically struggle to retain information across long sequences.
* During periods of significant trend change, the attention distribution widened, meaning the model drew information from a broader historical context to refine its forecast.
* Noise-heavy segments of the dataset naturally received lower attention weights, suggesting that the model successfully distinguished signal from noise.

These interpretability patterns confirm that the attention mechanism meaningfully enhances forecasting performance by enabling the model to identify, prioritize, and combine relevant historical dependencies. Overall, the Transformer architecture provided both predictive accuracy improvements and practical interpretability benefits relative to the LSTM baseline.



