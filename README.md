# Explainable Fourier Graph Neural Network Framework for Inter-well Connectivity Identification in Mature Shale Gas Reservoirs

This repository contains the PyTorch implementation of **Explainable Fourier Graph Neural Network Framework**.

---

## Project Structure

* `main.py`: The main entry point for training, validating, and testing the model.
* `model/FourierGNN.py`: The core implementation of the FourierGNN architecture.
* `data/data_loader.py`: Custom dataset classes for loading and preprocessing various time series datasets .
* `utils/utils.py`: Utility functions including metrics calculation (MAPE, RMSE, MAE), and model save/load helpers.
* `extract_weights.py`: A standalone script to extract and save the learned frequency-domain graph convolution weights and embeddings for downstream analysis.

---

## Requirements

Ensure you have the following dependencies installed:

* Python 3.8+
* PyTorch (>= 1.8.0, with CUDA support recommended)
* NumPy
* Pandas

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base:
FourierGNN：https://github.com/aikunyi/FourierGNN

## Getting Started

1. Data Preparation
By default, the script looks for datasets in the data/directory. Ensure your data files are placed correctly 
as defined in main.py.

2. Training the Model
To train the model from scratch, run main.py.

3. Extracting Learned Weights
To extract these weights and token embeddings after training, edit the configuration section in extract_weights.py
