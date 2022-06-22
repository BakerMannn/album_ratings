#Data Manipulation
import pandas as pd
import numpy as np
#Data Visulization
import plotly.express as px
import seaborn as sns  
import matplotlib.pyplot as plt
#Modeling
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#############################################################################
#Global Variables
album = 'Drake - Honestly, Nevermind'

#############################################################################
#Data Import
df = pd.read_excel(f'{album}/data.xlsx')

#Split
labels = df['song']
values = df.drop('song', axis=1)

#Categorical and Numerical Features
num_cols = values.select_dtypes(exclude=['object']).columns.tolist()
cat_cols = values.select_dtypes(include=['object']).columns.tolist()

#############################################################################
#Numeric Preprocessing Pipeline
num_pipe = make_pipeline(SimpleImputer(),
                        StandardScaler())

#Categorical Preprocessing Pipeline
cat_pipe = make_pipeline(SimpleImputer(strategy='constant', fill_value='N/A'),
                         OneHotEncoder(handle_unknown='ignore', sparse=False))

#Combined Preprocessing Pipeline
combined_pipe = ColumnTransformer([('num', num_pipe, num_cols),
                                   ('cat', cat_pipe, cat_cols)])

#############################################################################
#Optimal N_Cluster Selection
inertias=[]
k_list = range(1,10)

for k in k_list:
    full_pipe = make_pipeline(combined_pipe,
                          PCA(n_components=3),
                          KMeans(n_clusters=k))
    full_pipe.fit(values)
    inertias.append([k, full_pipe[2].inertia_])

inertia_df = pd.DataFrame(inertias, columns=['n_clusters', 'inertia'])

def optimal_n_clusters(data):
    x1, y1 = 2, data[0]
    x2, y2 = 20, data[len(data)-1]

    distances = []
    for i in range(len(data)):
        x0 = i+2
        y0 = data[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2    

n_clusters = optimal_n_clusters(inertia_df['inertia'])

#############################################################################
#KMeans Clustering

#Clustering Pipeline
full_pipe = make_pipeline(combined_pipe,
                          PCA(n_components=3, random_state=123),
                          KMeans(n_clusters=n_clusters, random_state=123))

clusters = full_pipe.fit_predict(values)
pca = full_pipe.fit_transform(values)

cluster_df = pd.DataFrame({'song':labels, 
                           'cluster':clusters, 
                           'PC1':pca[:,0],
                           'PC2':pca[:,1],
                           'PC3':pca[:,2]})
cluster_df['song'] = cluster_df['song'].astype(str)
df['song'] = df['song'].astype(str)
#############################################################################

#Color Map
#colors = ['#125A5F', '#814470', '#007749', "#F1753F", "#79DBC4", "#EEC6CB", "#97C74B", "FFD550"]
#unique_clusters = cluster_df['cluster'].unique()

#For Loop to Map Colors
#color_map = {}
#for i in range(len(unique_clusters)):
#    color_map[unique_clusters[i]] = colors[i]

#Change Cluster Column to String for Discrete Color Map
cluster_df['cluster'] = cluster_df['cluster'].astype(str)

#Plot
fig = px.scatter_3d(cluster_df, 
                    x='PC1', 
                    y='PC2', 
                    z='PC3',
                    color='cluster', 
                    hover_name='song',
                    hover_data ={'PC1': False, 'PC2': False, 'PC3': False},
                    opacity=0.7,
                    #color_discrete_map=color_map,
                    title = album,
                    template = 'seaborn'
                    )
fig.show()
fig.write_html(f'{album}/{album}_clusters.html')
#############################################################################
#Data Export
final_df = df.merge(cluster_df, how = 'inner', on='song')
final_df.to_excel(f'{album}/{album}_clusters.xlsx', index=False)