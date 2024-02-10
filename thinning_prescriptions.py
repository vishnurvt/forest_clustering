import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Loading the dataset 
file_path = r'D:\ML@IIT\Eli Jensen - Ironwood Forestry\Merged_Subsampling_Remove Outliers_Normalize by Ground Points_Point Cloud Segmentation.csv'
data = pd.read_csv(file_path)

# Select relevant features for clustering and removal heuristic
features_for_clustering = ['TreePosX', 'TreePosY']
features_for_removal = ['TreeID', 'TreePosX', 'TreePosY', 'TreePosZ', 'TreeHeight(metre)', 'DBH(metre)',
                         'CrownDiameter(metre)', 'CrownDiameter(S-N)(metre)', 'CrownDiameter(E-W)(metre)',
                         'CrownArea(square metre)', 'CrownVolume(cubic metre)', 'OldID']

# Choose the number of clusters
num_clusters = 10

# Perform KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[features_for_clustering])

# Heuristic for selecting trees to remove based on DBH, height, crown area, and crown volume
def select_trees_for_removal(cluster_data):
    # Sort trees within each cluster based on a combination of factors
    sorted_cluster = cluster_data.sort_values(by=features_for_removal[4:11], ascending=[True, True, True, True, True, True, True])
    
    # Select a percentage of trees for removal (adjustable)
    removal_percentage = 20
    num_trees_to_remove = int(len(sorted_cluster) * (removal_percentage / 100))
    
    # Return the selected trees for removal
    return sorted_cluster.head(num_trees_to_remove)

# Apply the heuristic to each cluster
trees_to_remove = data.groupby('Cluster').apply(select_trees_for_removal).reset_index(drop=True)

# Save the selected trees for removal to a CSV file
output_file_path = r'D:\ML@IIT\Eli Jensen - Ironwood Forestry\selected_trees_for_removal.csv'
trees_to_remove.to_csv(output_file_path, index=False)

# Visualize the clustering results with selected trees for removal
plt.scatter(data['TreePosX'], data['TreePosY'], c=data['Cluster'], cmap='viridis')
plt.scatter(trees_to_remove['TreePosX'], trees_to_remove['TreePosY'], marker='x', color='red', label='Trees for Removal')
plt.title('Tree Clustering with Selected Trees for Removal')
plt.xlabel('TreePosX')
plt.ylabel('TreePosY')
plt.legend()
plt.show()

