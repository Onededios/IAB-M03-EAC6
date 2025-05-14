"""
main script
"""

from functions import *

# from functions import create_dataset, plot_data_attributes, model_kmeans, predict_clusters, plot_clusters, plot_clusters3D, plot_clusters3D_HTML, transform_PCA, plot_elbow

NOM_ALUMNE = "OliveraJoel"

# creem el dataset
X, y = create_dataset(4)

# tenim 4 atributs
# TODO: número d'atributs de les dades
print(f"número d'atributs de les dades: {X.shape[1]}")


# TODO: els 5 primers elements de l'atribut 1
print(f"els 5 primers elements de l'atribut 1: {X[:5, 0]}")

# printem les dades, amb els atributs en els diferents eixos

plot_data_attributes(X, NOM_ALUMNE, False)

# model de clúster amb KMeans
km = model_kmeans(3)
# entrenament del model
km, y_km = predict_clusters(km, X)

# grafiquem els clústers amb diferents formats:
# a) format 2D
plot_clusters(km, X, y_km, NOM_ALUMNE, False)
# b) format 3D
plot_clusters3D(km, X, y_km, NOM_ALUMNE, False)
# c) format 3D amb HTML
plot_clusters3D_HTML(X, y_km, NOM_ALUMNE, False)

# cada clúster té els 4 atributs
# TODO: printem les 5 primeres dades de cada clúster

# https://stackoverflow.com/questions/29799053/how-to-print-result-of-clustering-in-sklearn
clusters = {}
for i, item in enumerate(y_km):
    feature = X[i]
    if item in clusters:
        clusters[item].append(feature)
    else:
        clusters[item] = [feature]

for cluster in clusters:
    print(f"els 5 primeres dades del cluster {cluster}: {clusters[cluster][:5]}")

# PCA ============================
# transformem les dades a dos atributs
X_PCA = transform_PCA(X, 2)

# TODO: comprovar que després de la transformació PCA el número d'atributs és 2
print(f"després de la transformació PCA el número d'atributs és: {X_PCA.shape[1]}")
# mètode del colze per comprovar que n=3 és una bona opció
# TODO:
plot_elbow(X_PCA, NOM_ALUMNE, False)

# model KMeans amb PCA
km_PCA = model_kmeans(3)
km_PCA, y_km_PCA = predict_clusters(km_PCA, X_PCA)
print(y_km_PCA[:10])

# grafiquem els clústers
# TODO
plot_clusters_PCA(km_PCA, X_PCA, y_km_PCA, NOM_ALUMNE, False)

# comparació dels resultats (assignació dels clústers) sense i amb PCA
print(y_km)
print(y_km_PCA)
# TODO: comprovar que aquestes dues llistes de valors són iguals
# https://stackoverflow.com/questions/10580676/comparing-two-numpy-arrays-for-equality-element-wise
print(f"aquestes dues llistes de valors són iguals: {(y_km == y_km_PCA).all()}")
# TODO: calcular l'homogeneïtat i completesa de les dades originals (4 atributs) i de les dades amb PCA (2 atributs)
calcular_scores(y, y_km, "KM 3 clusters")
calcular_scores(y, y_km_PCA, "KM PCA 3 clusters")
