# Threshold-based Correlation Clustering (TBCC) Algorithm
This repository contains the research paper titled "Optimizing Document Clustering through Correlation-Driven Cluster Formation". The paper presents an innovative approach to document clustering optimization using correlation-driven techniques. This repository also contains the script which is used to test the accuracy of the clustering algorithm.

## Introduction
This repository contains the implementation of the Threshold-based Correlation Clustering (TBCC) algorithm for document clustering. The algorithm optimizes clusters dynamically based on correlation thresholds, enhancing clustering accuracy and efficiency.

## Methodology
1. **Data Preprocessing**: Converts raw text data into TF-IDF vectors, including tokenization, lemmatization, and stop word removal.
2. **Correlation Matrix Calculation**: Uses Spearman's Rank Correlation Coefficient to analyze semantic relationships.
3. **Cluster Formation**: Identifies cohesive semantic groups based on correlation values.
4. **Cluster Optimization**: Merges clusters iteratively using the Jaccard Coefficient to improve cohesion.
5. **Cluster Refinement**: Refines clusters by reassigning unassigned columns.

## Experiments
- **Dataset**: 200 questions on Biotechnology, DBMS, Networking, and Climate Change (arbitary datasets).
- **Tools**: Python, Scikit-learn, NumPy, Pandas, NLTK.
- **Parameter Settings**: Threshold values varied from 0.2 to 0.3 in increments of 0.01. (This hyper-paramter must be set for efficient clustering)
<table>
    <tr>
        <td>No of clusters vs Threshold value</td>
        <td><img src="https://github.com/imsuraj675/Clustering-Algorithm/blob/main/Results/no_of_clusters.png" width="500px" /></td>
    </tr>
</table>
      
- Identification of Optimum Threshold Value

![Identification of Optimum Threshold Value](https://github.com/imsuraj675/Clustering-Algorithm/blob/main/Results/optimum_val.png)

From this we have identified that for the given Input files, the optimum threshold value is 0.23 or 0.25.

## Results
1. **Comparison Algorithms and Indices**:
   - Compared Algorithms: K-means, Affinity Propagation , Gaussian Mixture Model and Agglomerative Clustering.
   - Comparison Indices: Silhouette Score, Calinski Harabasz Score, Davies-Bouldin Index, Adjusted Rand Score, and Normalized Mutual Information Score.

2. **Accuracy Analysis**:
   - Our TBCC algorithm achieved an accuracy of:
     - Silhouette Score: 0.2445
     - Calinski Harabasz Score: 122.0241
     - Davies-Bouldin Index: 0.8524 
     - Adjusted Rand Score: 0.8721
     - Normalized Mutual Information Score: 0.8694

3. **Experiment Results**:
   For better evaluation of the algorithm, two comparisons are done on the clustering algorithms. The comparsions are as as follows:
   - Comparative analysis when number of clusters is set to numbers of clusters corresponding to the optimum the Threshold value (T=25)
     
     ![Comparative Analysis when N = 12](https://github.com/imsuraj675/Clustering-Algorithm/blob/main/Results/Comparative%20Analysis%20when%20N%3D12.png)
   - Comparative analysis when number of clusters is set to numbers of clusters corresponding to the no of topics (N=4)
     
     ![Comparative Analysis when N = 4](https://github.com/imsuraj675/Clustering-Algorithm/blob/main/Results/Comparative%20Analysis%20when%20N%3D4.png)

5. **Overall Performance**:
   - The TBCC algorithm demonstrated superior performance across multiple indices, forming more cohesive and accurate clusters compared to traditional clustering methods. The dynamic adaptation to semantic relationships and iterative optimization techniques contributed significantly to its enhanced clustering capability.

## Conclusion
The TBCC algorithm effectively forms semantically cohesive clusters, adapting dynamically to varying document relationships. Future work includes exploring automated threshold selection for improved performance.

## Future Discussion
- **Automated Threshold Selection**: Developing methods to automatically select optimal correlation thresholds.
- **Scalability**: Enhancing the algorithm's scalability for larger datasets.
- **Application Domains**: Applying the algorithm to various domains like data science, machine learning, and information retrieval.

## Limitations
- **Threshold Sensitivity**: Performance is sensitive to the choice of correlation threshold.
- **Computational Complexity**: May require significant computational resources for large datasets.

## Testing the Algorithm
To test the TBCC algorithm using the provided `main.py` file:
1. Clone the repository:
   ```bash
      git clone https://github.com/imsuraj675/Clustering-Algorithm/
      ```
2. Move to the Clustering-Algorithm directory
   ```bash
      cd Clustering-Algorithm
      ```
3. Ensure all dependencies are installed.
    - Install virtual environemnt (if installed ignore it)
      ```bash
      pip install virtualenv
      ```
    - Create a virtual environment using the following command.
      ```bash
      virtualenv env
      ```
    - Install requirements
      ```bash
      pip install -r requirements.txt
      ```
    - Run the intall_req.py
      ```bash
      python install_req.py
      ```
4. Put all the test document files inside the Input directory.
5. Run the `main.py` script:

   ```bash
   python main.py
   ```

