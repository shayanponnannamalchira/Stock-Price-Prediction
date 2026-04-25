# NVDA Stock Price Predictor (SVR)

This repository contains a Python implementation of **Support Vector Regression (SVR)** to analyze and predict NVIDIA (NVDA) stock prices. By comparing different mathematical kernels, the project demonstrates how machine learning can model financial time-series data.

## 🚀 Features
* **CSV Parsing:** Automatically extracts dates and opening prices from historical data.
* **Kernel Comparison:** Implements three SVR variations:
    * **RBF (Radial Basis Function):** Handles non-linear trends.
    * **Linear:** Best for simple, straight-line trends.
    * **Polynomial:** Fits data using curved relationships.
* **Visualization:** Generates a `matplotlib` graph comparing all three models against actual historical prices.

## 🛠️ Installation
Ensure you have Python installed. You can install the required libraries via pip:

```bash
pip install numpy scikit-learn matplotlib
