# Citation Network

This repository contains code for collecting, analyzing, and modeling citation networks using data from the OpenAlex API.

## Project Structure

- `download_dataset.py`: Script for collecting citation data from OpenAlex API
- `data_utils/`: Utilities for data preprocessing and handling
- `models/`: Neural network models for citation network analysis
- `train_autoregressive_model.py`: Training script for autoregressive model
- `train_cvae_model.py`: Training script for conditional variational autoencoder model

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/7pocheR/citation_network.git
   cd citation_network
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the data collection script:
   ```
   python download_dataset.py
   ```

4. Train the models:
   ```
   python train_autoregressive_model.py
   python train_cvae_model.py
   ```

## Requirements

See `requirements.txt` for the list of dependencies. 