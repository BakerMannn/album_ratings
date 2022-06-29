#Data Manipulation
import matplotlib
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

############################################################################
##Global Variables
artist = 'Drake'
pca_components = 3

#############################################################################
#Data Import
df = pd.read_excel(f'{artist}/data.xlsx')

#Split
labels = df[['album', 'song']]
values = df.drop(['album', 'song'], axis=1)

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
                              PCA(n_components=pca_components, random_state=123),
                              KMeans(n_clusters=k, random_state=123))
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
                        PCA(n_components=pca_components, random_state=123),
                        KMeans(n_clusters=n_clusters, random_state=123))

pca = full_pipe[0:2].fit_transform(values)
clusters = full_pipe.fit_predict(values)

cluster_df = pd.DataFrame({'album':labels['album'],
                           'song':labels['song'],
                           'rating':values['overall'],
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
#color_map[unique_clusters[i]] = colors[i]

#Change Cluster Column to String for Discrete Color Map
cluster_df['cluster'] = cluster_df['cluster'].astype(str)

#Cluster Plot
fig = px.scatter_3d(cluster_df, 
                        x='PC1', 
                        y='PC2', 
                        z='PC3',
                        color='cluster', 
                        hover_name='song',
                        hover_data={'PC1': False, 'PC2': False, 'PC3': False},
                        opacity=0.7,
                        size='rating',
                        #color_discrete_map=color_map,
                        title = artist,
                        template = 'seaborn',
                        symbol='album'
                        )

fig.show()
fig.write_html(f'{artist}/{artist}_clusters.html')

#############################################################################
#Data Export
final_df = df.merge(cluster_df, how = 'inner', on='song')
final_df.to_excel(f'{artist}/clusters.xlsx', index=False)

#############################################################################
#Driver's Plot
"""
final_df.drop(['PC1', 'PC2', 'PC3'], axis=1, inplace=True)
sns.pairplot(data=final_df,
            hue='cluster',
            )
plt.show()
"""