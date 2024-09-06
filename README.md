# GSMC: Metric Learning for AMP Activity Prediction

## Overview
This project develops a deep learning approach using a metric learning-guided Siamese neural network model for predicting antimicrobial activity differences among antimicrobial peptides (AMPs). Unlike traditional methods that focus on individual AMP activity, this model quantifies the activity differences between paired AMPs. This enhances prediction performance, especially in scenarios involving few-shot learning and data imbalance.

Our model is further integrated into the GSCM pipeline, which supports deep mutational scanning to discover potent AMPs. Validation experiments confirm that our approach can identify peptides with significantly higher antimicrobial activity and lower toxicity compared to standard treatments.

## Features
- Implementation of Siamese neural network for metric learning on AMPs.
- Evaluation scripts for model performance and feature visualization.
- Cascading pipeline for automated deep mutational scanning.
- Scripts for training and inference on both individual AMPs and their graph representations.

## Usage

### Data Preparation
Prepare your dataset using the provided scripts:
- `dataset.py`: General dataset preparation.
- `dataset_single.py`: Prepare dataset for single AMP predictions.
- `dataset_graph.py`: Prepare graph-based dataset representations.

### Training Models
Train the models using the following scripts:
- `train.py`: Train models on individual AMP data.
- `train_graph.py`: Train models on graph-based data representations.

Shell scripts are provided for convenience:
- `train.sh`: Bash script for training models on individual data.
- `train_graph.sh`: Bash script for training on graph data.

### Evaluation
Evaluate the models and visualize their performance:
- `eval.py`: Evaluate the models and print metrics.
- `eval_gradcam.py`: Visualize model decisions using Grad-CAM.
- `eval_graphgrad.py`: Grad-CAM for graph-based models.

### Inference
Run inference on new data:
- `infer_graph.py`: Perform inference on graph-based models.

## Project Structure
```plaintext
.
├── dataset.py
├── dataset_graph.py
├── dataset_graph2.py
├── dataset_single.py
├── eval.py
├── eval_gradcam.py
├── eval_graphgrad.py
├── get_experiments.py
├── infer_graph.py
├── loss.py
├── main.py
├── model.py
├── model_single.py
├── network.py
├── train.py
├── train.sh
├── train_graph.py
├── train_graph.sh
├── train_graph_single.py
├── train_single.py
└── utils.py
