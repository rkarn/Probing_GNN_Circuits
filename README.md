# Black-Box Probing of Graph Neural Networks in Circuits (ISCAS'85 + EPFL)

This repository contains Jupyter notebooks implementing a black-box probing framework to evaluate the robustness of Graph Neural Networks (GNNs) in circuit analysis. The approach leverages sensitivity metrics to assess model reliability in a black-box MLaaS environment. These metrics include:
- Jacobian Norm
- Lipschitz Constant
- Hessian-based Curvature
- Prediction Margin
- Adversarial Robustness Radius
- Stability under Input Noise

## Notebooks Overview

#### `Trojan_Injection_Parsing.ipynb` in `Trojan_Injection_Detection`
This notebook reads the ISCAS+EPFL dataset (from `https://github.com/jpsety/verilog_benchmark_circuits`) and injects Trojans. Three types of Trojans are inserted using Trust-Hub templates, including countermux, fsmor, and andxor. It then parses the Trojanized and clean samples and generates CSV files for node-level, subgraph-level, and graph-level classification.

#### `Trojan_Detection.ipynb` in `Trojan_Injection_Detection`
This notebook reads the CSV files and performs three types of training and evaluation for Trojan vs. non-Trojan classification. It uses different GNN architectures for node-level, subgraph-level, and graph-level tasks. The notebook is well-annotated, detailing the different architectural explorations. Further, it shows the evaluation performed for one architecture and the reasoning for moving to a better one.

#### Robustness Evaluation in `Trojan_Injection_Detection` folder
- `Trojan_Detection_Robustness_node-level.ipynb`: Performs node-level Trojan detection and computes several robustness metrics.
- `Trojan_Detection_Robustness_subgraph_level.ipynb`: Performs subgraph-level Trojan detection and computes several robustness metrics.
- `Trojan_Detection_Robustness_graph_level.ipynb`: Performs graph-level Trojan detection and computes several robustness metrics.

### `Probing_GCNN.ipynb` in `General_Node_Classification` folder
This notebook introduces the probing framework for evaluating GCNN robustness. It systematically computes sensitivity metrics across different circuit gate types using black-box techniques, providing insights into model behavior under adversarial conditions and random perturbations. It also measures the relative error of each metric and the ML evaluation metrics under perturbed features.

### `ISCAS85+EPFL_Parsing.ipynb` in `General_Node_Classification` folder
This notebook parses raw circuit netlist data from the ISCAS'85 and EPFL benchmarks, extracting nodes, edges, and circuit features. The processed data is saved as a CSV file used to build a DGL graph for the GCNN model.

### `Comparison with Other Architectures` in `General_Node_Classification` folder
This directory contains notebooks for the following architectures:
- GraphSAINT
- GraphSAGE
- Graph Isomorphism Network (GIN)
- Graph Attention Network (GAT)
They perform the same operations as `Probing_GCNN.ipynb`. The objective is to compare the robustness of GCNN with state-of-the-art GNN architectures for node classification.

## Installation & Dependencies

To run these notebooks, install the required dependencies:
- Python 3.8+
- torch, torch_geometric, dgl, numpy, pandas, matplotlib
- scikit-learn, seaborn, networkx

`Detailed instructions for generating a Docker image containing our trained GCNN model and deploying it on an AWS Container platform will be provided after the paper's acceptance. Stay tuned for the deployment code, Dockerfile, and step-by-step guidance.`
