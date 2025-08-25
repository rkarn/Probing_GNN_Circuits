# Black‑Box Probing of Graph Convolution Neural Networks in Circuit Analysis (ISCAS'86 + EPFL)

This repository contains Jupyter notebooks implementing a black-box probing framework to evaluate the robustness of Graph Convolutional Neural Networks (GCNNs) in circuit analysis. The approach leverages sensitivity metrics such as Jacobian norm, Lipschitz constant, Hessian-based curvature, prediction margin, adversarial robustness radius, and stability under input noise to assess model reliability in a black-box MLaaS environment.

## Notebooks Overview

### `Probing_GCNN.ipynb` in `General_Node_Classification` folder
This notebook introduces the probing framework for evaluating GCNN robustness. It systematically computes sensitivity metrics across different circuit gate types using black-box techniques, providing insights into model behavior under adversarial conditions and random perturbations. It also measures the relative-error of each metric and the ML evaluation metric under pruturbed features.

### `ISCAS85+EPFL_Parsing.ipynb` in `General_Node_Classification` folder
This notebook parses raw circuit netlist data from the ISCAS'85 and EPFL benchmarks, extracting nodes, edges, and circuit features. The processed data is saved as a CSV file, which is later used to build a graph with DGL as input for the GCNN model.

### `Comparison with Other Architectures` in `General_Node_Classification` folder
There are several notebooks under this directory. It uses following architectures:
- GraphSAINT
- GraphSAGE
- Graph isomorphism Network (GIN)
- Graph Attention Network (GAT)
They perform the same operations as in `Probing_GCNN.ipynb`. The objective is to compare the GCNN with state-of-the-arts GNN architecrure for robusness in node classification.

## Installation & Dependencies

To run these notebooks, install the required dependencies using:
- Python 3.8+
- torch, torch_geometric, dgl, numpy, pandas, matplotlib
- scikit-learn, seaborn, networkx

`Detailed instructions for generating a Docker image containing our trained GCNN model and deploying it on AWS Container platform will be provided after paper acceptance. Stay tuned for the deployment code, Dockerfile, and step-by-step guidance`.
