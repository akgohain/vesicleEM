## Clustering Methodology

This project employs a **Hierarchical Agglomerative Clustering** algorithm to group the data points. The specific choices within this methodology are detailed below:

### 1. What Clustering Algorithm is Used?

* **Algorithm:** Hierarchical Agglomerative Clustering.
    * **Hierarchical:** The algorithm builds a tree-like structure (a dendrogram) representing nested clusters. It starts with each data point as its own cluster and progressively merges the closest pairs of clusters.
    * **Agglomerative:** This describes the "bottom-up" approach of the algorithm, where individual data points are successively merged to form larger clusters.

* **Implementation Details:**
    * The `scipy.cluster.hierarchy.linkage` function is used to perform the clustering.
    * The `scipy.cluster.hierarchy.fcluster` function is then used to "flatten" the hierarchy and extract a specific set of clusters based on a distance criterion.

### 2. Why Was This Clustering Algorithm Chosen?

The choice of Hierarchical Agglomerative Clustering, along with specific components, was made for the following reasons:

* **Handling Mixed Data Types (Gower Distance):**
    * The dataset contains both numerical (e.g., `TotalVol`, `NucVol`) and categorical (e.g., `Branch`) features. Standard distance metrics (like Euclidean distance) are not suitable for such mixed data.
    * To address this, **Gower distance** (calculated using `gower.gower_matrix()`) is employed. Gower distance is specifically designed to compute a meaningful dissimilarity score between observations that have a combination of data types. It handles numerical features by typically using a range-normalized difference and categorical features by using a simple matching dissimilarity (0 if categories are the same, 1 if different).

* **No Pre-specification of Cluster Numbers:**
    * Unlike some other clustering algorithms (e.g., K-Means), hierarchical clustering does not require you to specify the number of clusters beforehand. The full hierarchy is built, and then a desired number of clusters or a cut-off point can be chosen.
    * In this script, the clusters are extracted using `fcluster` with a `distance_threshold`. This allows the data itself (via the linkage distances) to suggest the groupings up to a certain dissimilarity level, rather than forcing it into a fixed `k` number of clusters.

* **Understanding Data Structure (Dendrogram):**
    * The hierarchical nature allows for the generation of a **dendrogram**. This visualization is highly informative as it shows the structure of how data points and clusters are merged, revealing potential relationships at various levels of similarity. The script visualizes this, coloring the branches based on the extracted flat clusters.

* **Linkage Method (`method='complete'`):**
    * The `linkage` function requires a method to define the distance between clusters. This script uses `method='complete'` (complete linkage).

* **Extracting Flat Clusters (`fcluster` with a distance threshold):**
    * While the dendrogram shows the full hierarchy, for many practical applications, a single set of distinct, non-overlapping clusters is needed.
    * `fcluster(linked, t=distance_threshold, criterion='distance')` achieves this by cutting the dendrogram at the specified `distance_threshold`. All merges (and thus the data points within them) that occur below this distance value in the hierarchy are considered part of the same flat cluster.

