# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import necessary libraries such as pandas, matplotlib, seaborn, and scikit-learn modules.
2. Load the dataset and inspect the structure using functions like `head()` and `columns`.
3. Select relevant features (`Age`, `Annual Income (k$)`, and `Spending Score (1-100)`) for clustering.
4. Standardize the selected features using `StandardScaler` to normalize the data.
5. Use the Elbow Method to determine the optimal number of clusters by plotting inertia values.
6. Apply the K-Means algorithm with the chosen number of clusters (from elbow plot).
7. Add the cluster labels to the original dataset.
8. Compute the silhouette score to evaluate clustering quality.
9. Visualize the clusters using a scatter plot of `Annual Income (k$)` vs. `Spending Score (1-100)`, colored by cluster.


## Program:
```

Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: SREE HARI K
RegisterNumber: 212223230212


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/CustomerData.csv"
data = pd.read_csv(url)

# Step 2: Explore the data
# Display the first few rows of the dataset along with column names for inspection
print(data.head())
print(data.columns)

# Step 3: Select the relevant features for clustering
# Here, we are using 'Age', 'Annual Income (k$)', and 'Spending Score (1-100)' for the clustering process
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# Step 4: Data Preprocessing
# Standardize the features to enhance the performance of the K-Means algorithm
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Identify the optimal number of clusters using the Elbow Method
inertia_values = []  # List to store the inertia values (within-cluster sum of squares)
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve to visualize the optimal number of clusters
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Step 6: Train the K-Means model with the chosen number of clusters
# Based on the elbow plot, we select the optimal number of clusters, which is 4 in this case
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X_scaled)

# Step 7: Analyze and visualize the clusters
# Add the cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

# Calculate and display the silhouette score to evaluate the quality of clustering
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

# Visualize the clusters based on 'Annual Income (k$)' and 'Spending Score (1-100)'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

```

## Output:
### DATASET PREVIEW:
![image](https://github.com/user-attachments/assets/984dbb1a-433b-4132-a216-8e91cbe9ced3)
### Elbow Curve for Optimal Clusters:
![image](https://github.com/user-attachments/assets/50a5c71a-10ea-42a0-b43f-9db1b41ea816)
### Clustering Quality Metric:
![image](https://github.com/user-attachments/assets/068ace64-d726-4755-be5f-214d3ede80d8)
###  Cluster Visualization:
![image](https://github.com/user-attachments/assets/868f5102-5727-4abb-ac95-a63491f1c52f)

## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
