import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv')
df=df_raw[['Latitude','Longitude', 'MedInc']]
escalador=StandardScaler()
df_norm=escalador.fit_transform(df)
kmeans = KMeans(n_clusters=2)
kmeans.fit(df_norm)
df2=escalador.inverse_transform(df_norm)
df2=pd.DataFrame(df2,columns=['MedInc','Latitude','Longitude'])
df2['Cluster'] = kmeans.labels_
df2['Cluster']=pd.Categorical(df2['Cluster'])
filename='../models/final_model.sav'
pickle.dump(kmeans, open(filename, 'wb'))
