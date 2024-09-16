# Assignment 2 Report

To install dependencies, use 
```bash
pip install -r requirements.txt
```

To run any python3 file, run it from the base directory using,
```bash
python3 -m path.to.code.file # without .py extension
```
To run the code for this assignment,
```bash
python3 -m assignments.2.a2 # (without .py extension)
```
Website references are mentioned in the code.

# K-Means Clustering

## Elbow Plot for seed 43

![Elbow Plot](./figures/elbow_kmeans_8.png)

Optimal number of clusters (k_kmeans1): 8
Cluster labels: [4 4 6 2 2 7 4 4 7 6 2 4 2 1 1 0 1 4 4 3 4 4 6 4 6 0 6 4 7 0 6 5 2 5 2 7 7
 1 4 6 7 1 7 4 0 0 4 4 4 4 4 4 5 4 0 0 0 4 2 7 2 0 6 4 0 7 5 2 7 4 2 4 4 7
 5 2 0 1 0 3 4 0 0 2 1 4 7 0 0 1 3 2 4 6 4 1 7 1 0 4 1 1 0 4 7 0 3 0 1 0 7
 7 1 7 1 6 7 6 6 4 4 4 0 6 2 7 5 7 7 4 7 7 0 7 6 5 3 1 5 7 1 4 4 4 4 1 5 4
 4 4 6 0 4 4 0 0 4 4 0 0 4 2 6 6 6 0 1 5 7 6 1 7 6 7 2 0 4 1 4 5 3 0 0 6 4
 7 1 2 5 7 6 0 4 0 1 4 0 0 0 5]
WCSS cost for optimal k: 3800.3406663064675


## Elbow Plot for seed 33

![Elbow Plot](./figures/elbow_kmeans_5.png)

Optimal number of clusters (k_kmeans1): 5
Cluster labels: [1 1 3 3 3 4 3 1 4 3 4 4 3 2 2 0 3 1 1 3 1 3 3 1 3 3 3 4 4 3 3 2 0 2 4 4 4
 2 1 3 4 3 4 1 0 3 1 1 1 1 3 1 0 1 0 0 3 1 3 4 2 0 3 4 3 4 0 4 4 0 4 4 4 1
 2 0 0 2 0 1 1 0 0 3 0 3 4 0 3 3 1 4 1 3 3 3 1 2 3 1 2 3 0 0 4 0 2 0 3 0 4
 4 2 4 0 3 4 0 3 3 0 0 0 3 3 4 0 4 4 1 1 4 3 1 3 0 1 2 2 4 2 3 1 1 2 2 2 0
 3 1 3 0 3 4 0 0 1 1 0 0 1 3 3 3 3 0 2 0 4 0 0 4 3 4 3 0 3 2 1 2 1 0 3 3 1
 4 0 3 2 4 3 0 1 0 2 1 0 0 0 2]
WCSS cost for optimal k: 3966.973318266988


## Elbow Plot for seed 35

![Elbow Plot](./figures/elbow_kmeans_4.png)

Optimal number of clusters (k_kmeans1): 4
Cluster labels: [2 1 0 0 0 1 2 2 1 0 2 1 0 3 3 2 0 1 2 2 2 0 0 2 0 2 0 1 1 2 0 3 3 3 1 1 1
 3 1 0 1 2 1 1 2 2 2 2 2 2 0 2 2 1 2 3 2 1 0 1 3 3 0 1 2 1 3 1 1 2 1 1 1 1
 3 3 3 2 3 2 2 3 3 0 3 2 1 2 2 0 2 1 1 0 2 0 1 3 0 2 3 0 2 2 1 2 3 3 0 3 1
 1 2 1 3 0 1 3 0 2 2 2 2 0 0 1 3 1 1 1 1 1 0 1 0 3 2 2 3 1 3 2 2 2 2 3 3 2
 2 2 2 2 2 1 2 3 1 2 3 2 1 0 0 3 0 2 2 3 1 3 2 1 0 1 2 2 2 2 1 3 2 2 2 0 2
 1 2 2 3 1 0 2 2 2 3 2 2 2 2 3]
WCSS cost for optimal k: 4057.293281791949

**We settle at $K_{kmeans1}$ = 5 due to the uniform nature of the WCSS graph and the single indisputable elbow point.**

# Gaussian Mixture Models

**Error on AIC BIC > dataset**

For this we get WCSS cost as 3966.973318266988

Optimal number of dimensions based on 85% explained variance: 9
Reconstruction Error: 0.13712198288339003
Explained Variance Ratio: 0.8629
Classification Task Scores
Accuracy: 0.2561
  Precision (macro): 0.2340
  Recall (macro): 0.2472
  F1-Score (macro): 0.2405
  Precision (micro): 0.2561
  Recall (micro): 0.2561
  F1-Score (micro): 0.2561
time_taken=65.20476651191711

Classification Task Scores
Accuracy: 0.2167
  Precision (macro): 0.1884
  Recall (macro): 0.2089
  F1-Score (macro): 0.1981
  Precision (micro): 0.2167
  Recall (micro): 0.2167
  F1-Score (micro): 0.2167
time_taken=40.58926296234131