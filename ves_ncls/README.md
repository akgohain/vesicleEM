# Cluster Analysis of Vesicle Morphology

This project performs a cluster analysis on morphological data of vesicles to identify distinct structural groups. The analysis uses a dataset of 20 vesicle samples, characterized by a combination of numerical (e.g., 'TotalVol', 'NucVol', 'TotalLen') and categorical (e.g., 'Branch' type) features.

The primary methods employed include:
* Log transformation for numerical features.
* Gower's distance to handle mixed data types.
* Hierarchical agglomerative clustering (complete linkage method).
* Cluster identification based on a distance threshold of 0.4.
* Visualization of results using a dendrogram and a 2D Multidimensional Scaling (MDS) plot.

This repository contains the Google Colab notebook used for the analysis.

The analysis is performed in a Python environment. The key dependencies are:
* Python 3.x
* pandas
* numpy
* scipy (for spatial distance, cluster hierarchy)
* gower
* matplotlib (for plotting)
* scikit-learn (for MDS)

The notebook includes a command to install the `gower` library:
```bash
!pip3 install gower
