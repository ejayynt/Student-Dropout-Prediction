# Student Dropout Prediction using Graph Neural Networks

This repository contains a research-based implementation for predicting student dropout utilising **Graph Neural Networks (GNNs)**. The model constructs a student-factor graph based on causal relationships identified via **Fuzzy DEMATEL** and employs a hybrid architecture combining **Graph Attention Networks (GAT)** and **Graph Convolutional Networks (GCN)** for accurate classification.

## Project Overview

Traditional dropout prediction models often treat student data as independent tabular records. This project focuses on the *relationships* between various dropout factors (e.g., financial constraints, teaching quality, academic performance). By modelling these factors as a graph, we capture the complex interdependencies that lead to student attrition.

### Key Features
*   **Graph-Based Modelling**: Converts student data into graph structures where nodes represent dropout factors.
*   **Causal Weighting**: Uses **Fuzzy DEMATEL** (Decision-Making Trial and Evaluation Laboratory) scores to define the edges and weights between factors, prioritising significant causal relationships.
*   **Hybrid Architecture**: Combines the attention mechanism of GAT (to focus on important neighbours) with the local aggregation of GCN.
*   **Personalised Graphs**: Each student is represented as a unique graph based on their specific feature values.

## Methodology & Architecture

The system follows a pipeline that transforms raw data into a graph format suitable for GNN processing.

<p>
<img alt="YOLOv8 Image Detection-2026-02-03-075215" src="https://github.com/user-attachments/assets/dc61d9c1-63cb-428d-84ef-d47ba1c42ace" width="450"/>
</p>

1.  **Data Preprocessing**: Student features are normalized.
2.  **Graph Construction**: 
    *   **Nodes**: Correspond to selected factors (e.g., "Financial and domestic constraints", "Poor teaching quality").
    *   **Edges**: Defined by a pre-computed DEMATEL matrix. An edge exists if the causal influence between two factors exceeds a certain threshold (e.g., > 0.2).
    *   **Node Features**: Each node contains the student's specific value for that factor plus a positional encoding.
3.  **Model (ImprovedGATModel)**:
    *   **3x GAT Layers**: Apply multi-head attention to learn the importance of neighboring factors.
    *   **1x GCN Layer**: Refines local structural information.
    *   **Pooling & Classification**: Aggregates node features into a graph-level representation to predict the binary outcome (Success vs. Dropout).

## Getting Started

### Prerequisites

Ensure you have Python installed. You will need the following libraries:

*   `torch` (PyTorch)
*   `torch_geometric` (PyG)
*   `pandas`
*   `numpy`
*   `sklearn`
*   `openpyxl` (for reading Excel files)

### Installation

```bash
pip install torch torch-geometric pandas numpy scikit-learn openpyxl
```


## Project Structure

*   `student_dropout.py`: The main script containing the model definition (`ImprovedGATModel`), data preprocessing, graph construction, training loop, and evaluation.
*   `dataset.csv` / `final.xlsx`: The dataset containing student feedback and dropout labels.
*   `README.md`: Project documentation.

## Usage

To train and evaluate the model, simply run the main script:

```bash
python student_dropout.py
```

The script will:
1.  Load the dataset.
2.  Construct graphs for each student.
3.  Split data into stratified train, validation, and test sets.
4.  Train the Hybrid GAT model (with early stopping).
5.  Output the final test accuracy.
