#!/usr/bin/env python
# coding: utf-8

# # In this challenge, use your knowledge of Python and unsupervised learning to predict if cryptocurrencies are affected by 24-hour or 7-day price changes.

# In[1]:


# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[13]:


import warnings
warnings.filterwarnings("ignore")


# In[14]:


# Dependencies
import pandas as pd
from pathlib import Path


# In[15]:


# Store filepath in a variable
file_one = Path("Documents\crypto market data.csv")


# In[16]:


# Read our data file with the Pandas library
file_one_df = pd.read_csv(file_one, encoding="ISO-8859-1")


# In[17]:


# Show the first five rows.
file_one_df.head()


# In[18]:


import pandas as pd
import numpy as np 

# Generate summary statistics
file_one_df.describe()


# In[19]:


# View column names of dataframe
file_one_df.columns


# In[20]:


# Plot your data to see what's in your DataFrame
file_one_df.hvplot.line(
    width=800,
    height=400,
    rot=90
)


# # Prepare our data :
# 
# Create a DataFrame with the scaled data and set the "coin_id" index from the original DataFrame as the index for the new DataFrame.

# In[21]:


import sklearn 
from sklearn.preprocessing import StandardScaler

# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
file_one_scaled = StandardScaler().fit_transform(
    file_one_df[['price_change_percentage_24h', 'price_change_percentage_7d',
       'price_change_percentage_14d', 'price_change_percentage_30d',
       'price_change_percentage_60d', 'price_change_percentage_200d',
       'price_change_percentage_1y']]
)


# In[22]:


# Create a DataFrame with the scaled data
file_one_scaled = pd.DataFrame(
    file_one_scaled,
    columns=['price_change_percentage_24h', 'price_change_percentage_7d',
       'price_change_percentage_14d', 'price_change_percentage_30d',
       'price_change_percentage_60d', 'price_change_percentage_200d',
       'price_change_percentage_1y']
)

# Copy the crypto names from the original data
file_one_scaled['coin_id'] = file_one_df.index

# Set the coinid column as index
file_one_scaled = file_one_scaled.set_index('coin_id')

# Display sample data
file_one_scaled.head()


# # Find the best value for K :
# 
# Use the elbow method to find the best value for K

# In[23]:


# Create a list with the number of k-values from 1 to 11
k = list(range(1, 11))


# In[24]:


# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    k_model = KMeans(n_clusters=i, random_state=0)
    k_model.fit(file_one_scaled)
    inertia.append(k_model.inertia_)


# In[25]:


# Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k": k, "inertia": inertia}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow = pd.DataFrame(elbow_data)

# Review the DataFrame
df_elbow.head()


# In[26]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
k_elbow = df_elbow.hvplot.line(
    x="k", 
    y="inertia", 
    title="Elbow Curve - Scaled Data", 
    xticks=k
)
k_elbow


# # Question: what is the best value for K?
# Answer: the best value for K is 4

# # Cluster Cryptocurrencies with K-means Using the Original Data

# In[27]:


# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4, random_state=0)


# In[28]:


# Fit the K-Means model using the scaled data
model.fit(file_one_scaled)


# In[29]:


# Predict the clusters to group the cryptocurrencies using the scaled data
km = model.predict(file_one_scaled)

# Print the resulting array of cluster values.
print(km)


# In[30]:


# Create a copy of the DataFrame
df_crypto_predictions = file_one_scaled.copy()


# In[31]:


# Add a new column to the DataFrame with the predicted clusters
df_crypto_predictions['clusters'] = km

# Display sample data
df_crypto_predictions.head()


# In[32]:


# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
k_plot = df_crypto_predictions.hvplot.scatter(
    x="price_change_percentage_24h", 
    y="price_change_percentage_7d", 
    title="KMeans Clusters - Original Scaled Data",
    by='clusters',
    hover_cols='coin_id' 
)
k_plot


# # Optimize Clusters with Principal Component Analysis.

# In[33]:


# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)


# In[34]:


# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
crypto_pca_data = pca.fit_transform(file_one_scaled)
# View the first five rows of the DataFrame. 
crypto_pca_data[0:5]


# In[35]:


# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
pca.explained_variance_ratio_


# In[36]:


# Create a new DataFrame with the PCA data.

# Creating a DataFrame with the PCA data
crypto_pca_df = pd.DataFrame(crypto_pca_data, columns=["PC1", "PC2", "PC3"])

# Copy the crypto names from the original data
crypto_pca_df['coin_id'] = file_one_scaled.index

# Set the coinid column as index
df_crypto_pca = crypto_pca_df.set_index(['coin_id'])

# Display sample data
df_crypto_pca.head()


# # Find the Best Value for k Using the PCA Data

# In[37]:


# Create a list with the number of k-values from 1 to 11
k_pca = list(range(1,11))


# In[38]:


# Create an empiy list to store the inertia values
inertia_pca = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
for i in k_pca:
    k_pca_model = KMeans(n_clusters=i, random_state=0)
    k_pca_model.fit(df_crypto_pca)
    inertia_pca.append(k_pca_model.inertia_)


# In[39]:


# Create a dictionary with the data to plot the Elbow curve
pca_elbow_data = {"k": k_pca, "inertia_pca": inertia_pca}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow_pca = pd.DataFrame(pca_elbow_data)

# Review the DataFrame
df_elbow_pca.head()


# In[40]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
pca_elbow = df_elbow_pca.hvplot.line(
    x="k", 
    y="inertia_pca", 
    title="Elbow Curve PCA",
    color='red',
    xticks=k
)
pca_elbow


# # Question: What is the best value for k when using the PCA data?
# Answer: the best value for K using the PCA data is 2

# # Question: Does it differ from the best k value found using the original data?
# Answer: the best value for K using the original data with the elbow method was 4, the PCA data the best value for K was 2. 

# # Cluster Cryptocurrencies with K-means Using the PCA Data

# In[41]:


# Initialize the K-Means model using the best value for k
pca_model = KMeans(n_clusters=4, random_state=0)


# In[42]:


# Fit the K-Means model using the PCA data
pca_model.fit(df_crypto_pca)


# In[43]:


# Predict the clusters to group the cryptocurrencies using the PCA data
k_pca_model = pca_model.predict(df_crypto_pca)

# Print the resulting array of cluster values.
print(k_pca_model)


# In[44]:


# Create a copy of the DataFrame with the PCA data
df_crypto_pca_predictions = df_crypto_pca.copy()

# Add a new column to the DataFrame with the predicted clusters
df_crypto_pca_predictions['clusters'] = k_pca_model

# Display sample data
df_crypto_pca_predictions.head()


# In[45]:


# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.

#UPDATE: Correcting to match Requirements
# Using hvPlot, create a scatter plot by setting x="PC1" and y="PC2".
# Color the graph points with the labels that you found by using K-means.
# Then add the crypto name to the hover_cols parameter to identify the cryptocurrency that each data point represents.
pca_plot = df_crypto_pca_predictions.hvplot.scatter(
    x="PC1", 
    y="PC2", 
    title="KMeans Clusters Using PCA = 3",
    by='clusters',
    hover_cols='coin_id' 
)
pca_plot


# # Visualize and compare results

# In[46]:


# Composite plot to contrast the Elbow curves

composite_scatter_distinct = k_elbow + pca_elbow
composite_scatter_distinct


# # Question: What is the impact of using fewer features to cluster the data using K-Means?

# Answer: The result of using less number of features to cluster the data using K-Means has a significant impact. In the initial data, the elbow curve gave us results that the optimal value for K was 4, which resulted in 4 clusters. By implementing PPCA and using the optimal value for K of 2, the result was more fitting with our data.

# In[ ]:




