# Black‑Box Probing of Graph Convolution Neural Networks in Circuit Analysis (ISCAS'86 + EPFL)

This repository contains Jupyter notebooks implementing a black-box probing framework to evaluate the robustness of Graph Convolutional Neural Networks (GCNNs) in circuit analysis. The approach leverages sensitivity metrics such as Jacobian norm, Lipschitz constant, Hessian-based curvature, prediction margin, adversarial robustness radius, and stability under input noise to assess model reliability in a black-box MLaaS environment.

## Notebooks Overview

### `Probing_GCNN.ipynb`
This notebook introduces the probing framework for evaluating GCNN robustness. It systematically computes sensitivity metrics across different circuit gate types using black-box techniques, providing insights into model behavior under adversarial conditions and random perturbations.

### `ISCAS85+EPFL_Parsing.ipynb`
This notebook parses raw circuit netlist data from the ISCAS'85 and EPFL benchmarks, extracting nodes, edges, and circuit features. The processed data is saved as a CSV file, which is later used to build a graph with DGL as input for the GCNN model.

## Installation & Dependencies

To run these notebooks, install the required dependencies using:
- Python 3.8+
- torch, dgl, numpy, pandas, matplotlib
- scikit-learn, seaborn, networkx

