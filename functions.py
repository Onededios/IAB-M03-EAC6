"""Helper functions to generate data, models, train models and plot data"""

import os

import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA


# create dataset
def create_dataset(number_features):
    """
    creates a dataset based on features qty
    :param number_features:
    :return: tuple (x, y) -> (features, cluster assignments)
    """
    # pylint: disable=unbalanced-tuple-unpacking
    x, y = make_blobs(
        n_samples=250,
        n_features=number_features,
        centers=3,
        cluster_std=0.75,
        shuffle=True,
        random_state=100,
    )
    return x, y


# =========================================================


def plot_data_attributes(x, nom_alumne, show=True):
    """
    plots data attributes
    :rtype: object
    :param x:
    :param nom_alumne:
    :param show:
    """
    fig, axs = plt.subplots(2, 2)

    fig.suptitle(f"Data Attributes - {nom_alumne}", fontsize=12, y=1)
    fig.text(0.5, 0.95, "title", horizontalalignment="center")

    fig.set_size_inches(6, 6)

    axs[0, 0].set_title("attr1 vs attr2")
    axs[0, 0].scatter(x[:, 0], x[:, 1], c="white", marker="o", edgecolor="black", s=50)

    axs[0, 1].set_title("attr1 vs attr3")
    axs[0, 1].scatter(x[:, 0], x[:, 2], c="white", marker="o", edgecolor="black", s=50)

    axs[1, 0].set_title("attr1 vs attr4")
    axs[1, 0].scatter(x[:, 0], x[:, 3], c="white", marker="o", edgecolor="black", s=50)

    axs[1, 1].set_title("attr2 vs attr3")
    axs[1, 1].scatter(x[:, 1], x[:, 2], c="white", marker="o", edgecolor="black", s=50)

    plt.savefig(f"img/scatters_{nom_alumne}.png")

    if show:
        plt.show()

    plt.close()  # Force close this instance


# =========================================================


def model_kmeans(num_clusters):
    """
    returns a KMeans model based on cluster qty
    :rtype: object
    :param num_clusters: 
    :return: 
    """
    km = KMeans(
        n_clusters=num_clusters,
        init="random",
        n_init=10,
        max_iter=300,
        tol=1e-04,
        random_state=0,
    )

    return km


# =========================================================


def predict_clusters(model, x):
    """
    obtains label prediction from the model
    :rtype: object
    """
    y_km = model.fit_predict(x)
    return model, y_km


# =========================================================


def plot_clusters(km, x, y_km, nom_alumne, show=True):
    """
    plots cluster distribution
    :rtype: object
    :param km:
    :param x:
    :param y_km:
    :param nom_alumne:
    :param show:
    """
    # plot the 3 clusters
    fig, axs = plt.subplots(1, 3)

    fig.suptitle(f"Clusters - {nom_alumne}", fontsize=12, y=1)

    fig.set_size_inches(12, 4)

    axs[0].set_title("attr1 vs attr2")
    axs[0].scatter(
        x[y_km == 0, 0],
        x[y_km == 0, 1],
        s=50,
        c="lightgreen",
        marker="s",
        edgecolor="black",
        label="cluster 1",
    )
    axs[0].scatter(
        x[y_km == 1, 0],
        x[y_km == 1, 1],
        s=50,
        c="orange",
        marker="o",
        edgecolor="black",
        label="cluster 2",
    )

    axs[0].scatter(
        x[y_km == 2, 0],
        x[y_km == 2, 1],
        s=50,
        c="lightblue",
        marker="v",
        edgecolor="black",
        label="cluster 3",
    )

    # plot the centroids
    axs[0].scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 1],
        s=250,
        marker="*",
        c="red",
        edgecolor="black",
        label="centroids",
    )

    axs[1].set_title("attr1 vs attr3")
    axs[1].scatter(
        x[y_km == 0, 0],
        x[y_km == 0, 2],
        s=50,
        c="lightgreen",
        marker="s",
        edgecolor="black",
        label="cluster 1",
    )
    axs[1].scatter(
        x[y_km == 1, 0],
        x[y_km == 1, 2],
        s=50,
        c="orange",
        marker="o",
        edgecolor="black",
        label="cluster 2",
    )

    axs[1].scatter(
        x[y_km == 2, 0],
        x[y_km == 2, 2],
        s=50,
        c="lightblue",
        marker="v",
        edgecolor="black",
        label="cluster 3",
    )

    # plot the centroids
    axs[1].scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 2],
        s=250,
        marker="*",
        c="red",
        edgecolor="black",
        label="centroids",
    )

    axs[2].set_title("attr1 vs attr4")
    axs[2].scatter(
        x[y_km == 0, 0],
        x[y_km == 0, 3],
        s=50,
        c="lightgreen",
        marker="s",
        edgecolor="black",
        label="cluster 1",
    )
    axs[2].scatter(
        x[y_km == 1, 0],
        x[y_km == 1, 3],
        s=50,
        c="orange",
        marker="o",
        edgecolor="black",
        label="cluster 2",
    )

    axs[2].scatter(
        x[y_km == 2, 0],
        x[y_km == 2, 3],
        s=50,
        c="lightblue",
        marker="v",
        edgecolor="black",
        label="cluster 3",
    )

    # plot the centroids
    axs[2].scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 3],
        s=250,
        marker="*",
        c="red",
        edgecolor="black",
        label="centroids",
    )

    plt.legend(scatterpoints=1)

    filename = f"img/clusters_{nom_alumne}.png"
    counter = 1
    while os.path.exists(filename):
        filename = filename.replace(".png", f"_{counter}.png")
        counter += 1
    plt.savefig(filename)

    if show:
        plt.show()

    plt.close()  # Force close this instance


# =========================================================


def plot_clusters_pca(km, x, y_km, nom_alumne, show=True):
    """
    plots cluster distribution on pca 2 attr function
    :param km:
    :param x:
    :param y_km:
    :param nom_alumne:
    :param show:
    :rtype: object
    """
    plt.suptitle("attr1 vs attr2")
    plt.scatter(
        x[y_km == 0, 0],
        x[y_km == 0, 1],
        s=50,
        c="lightgreen",
        marker="s",
        edgecolor="black",
        label="cluster 1",
    )
    plt.scatter(
        x[y_km == 1, 0],
        x[y_km == 1, 1],
        s=50,
        c="orange",
        marker="o",
        edgecolor="black",
        label="cluster 2",
    )

    plt.scatter(
        x[y_km == 2, 0],
        x[y_km == 2, 1],
        s=50,
        c="lightblue",
        marker="v",
        edgecolor="black",
        label="cluster 3",
    )

    # plot the centroids
    plt.scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 1],
        s=250,
        marker="*",
        c="red",
        edgecolor="black",
        label="centroids",
    )

    filename = f"img/clusters_pca_{nom_alumne}.png"
    counter = 1
    while os.path.exists(filename):
        filename = filename.replace(".png", f"_{counter}.png")
        counter += 1
    plt.savefig(filename)

    if show:
        plt.show()

    plt.close()  # Force close this instance


# =========================================================


def plot_clusters_3d(km, x, y_km, nom_alumne, show=True):
    """
    plots 3D cluster distribution
    :rtype: object
    :param km:
    :param x:
    :param y_km:
    :param nom_alumne:
    :param show:
    """
    fig = plt.figure()
    fig.set_size_inches(6, 6)
    fig.suptitle(f"Clusters - {nom_alumne}", fontsize=12, y=1)
    ax = fig.add_subplot(projection="3d")

    ax.scatter(
        x[y_km == 0, 0],
        x[y_km == 0, 1],
        x[y_km == 0, 2],
        s=50,
        c="lightgreen",
        marker="s",
        edgecolor="black",
        label="cluster 1",
    )

    ax.scatter(
        x[y_km == 1, 0],
        x[y_km == 1, 1],
        x[y_km == 1, 2],
        s=50,
        c="orange",
        marker="s",
        edgecolor="black",
        label="cluster 2",
    )

    ax.scatter(
        x[y_km == 2, 0],
        x[y_km == 2, 1],
        x[y_km == 2, 2],
        s=50,
        c="lightblue",
        marker="s",
        edgecolor="black",
        label="cluster 3",
    )

    ax.scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 1],
        km.cluster_centers_[:, 2],
        s=250,
        marker="*",
        c="red",
        edgecolor="black",
        label="centroids",
    )

    ax.set_xlabel("attr1")
    ax.set_ylabel("attr2")
    ax.set_zlabel("attr3")

    ax.legend(scatterpoints=1)
    ax.grid()

    filename = f"img/clusters3D_{nom_alumne}.png"
    counter = 1
    while os.path.exists(filename):
        filename = filename.replace(".png", f"_{counter}.png")
        counter += 1
    plt.savefig(filename)

    if show:
        plt.show()

    plt.close()  # Force close this instance


# =========================================================


def plot_clusters_3d_html(x, y_km, nom_alumne, show=True):
    """
    plots cluster distribution on al html exportable file
    :rtype: object
    :param x:
    :param y_km:
    :param nom_alumne:
    :param show:
    """
    # Configure Plotly to be rendered inline in the notebook.
    plotly.offline.init_notebook_mode()

    # Added edge black color to avoid eye ache
    # Found and example here
    # https://plotly.com/python/3d-scatter-plots/#style-marker-size-and-color
    # Configure the trace.
    cluster1 = go.Scatter3d(
        x=x[y_km == 0, 0],
        y=x[y_km == 0, 1],
        z=x[y_km == 0, 2],
        mode="markers",
        marker={
            "size": 5,
            "opacity": 0.8,
            "color": "lightgreen",
            "symbol": "circle",
            "line": {"width": 1, "color": "black"},
        },
        name="cluster 1",
    )

    cluster2 = go.Scatter3d(
        x=x[y_km == 1, 0],
        y=x[y_km == 1, 1],
        z=x[y_km == 1, 2],
        mode="markers",
        marker={
            "size": 5,
            "opacity": 0.8,
            "color": "orange",
            "symbol": "circle",
            "line": {"width": 1, "color": "black"},
        },
        name="cluster 2",
    )

    cluster3 = go.Scatter3d(
        x=x[y_km == 2, 0],
        y=x[y_km == 2, 1],
        z=x[y_km == 2, 2],
        mode="markers",
        marker={
            "size": 5,
            "opacity": 0.8,
            "color": "lightblue",
            "symbol": "circle",
            "line": {"width": 1, "color": "black"},
        },
        name="cluster 3",
    )

    # Added legend to identify clusters
    # https://stackoverflow.com/questions/26941135/show-legend-and-label-axes-in-plotly-3d-scatter-plots
    # Added title
    # https://stackoverflow.com/questions/58166002/how-to-add-caption-subtitle-using-plotly-method-in-python
    # Configure the layout.
    # **Made tests and removed margins to correctly show legend and title
    layout = go.Layout(
        showlegend=True,
        title=go.layout.Title(
            text=f"Clusters - {nom_alumne}"
        )
    )

    data = [cluster1, cluster2, cluster3]

    plot_figure = go.Figure(data=data, layout=layout)

    # Render the plot.
    plotly.offline.iplot(plot_figure)

    filename = f"img/clusters3DHTML_{nom_alumne}.html"
    counter = 1
    while os.path.exists(filename):
        filename = filename.replace(".html", f"_{counter}.html")
        counter += 1

    plotly.offline.plot(plot_figure, filename=filename, auto_open=show)


# =========================================================


def transform_pca(x, num_components):
    """
    transforms a set into a PCA specified attr qty
    :rtype: object
    :param x:
    :param num_components:
    :return:
    """
    x_pca = PCA(n_components=num_components).fit_transform(x)
    return x_pca


# =========================================================


def plot_elbow(x_pca, nom_alumne, show=True, min_n_clusters=1, max_n_clusters=9):
    """
    plots elbow cluster metrics
    :rtype: object
    :param x_pca:
    :param nom_alumne:
    :param show:
    :param min_n_clusters:
    :param max_n_clusters:
    """
    # https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/

    clusters = range(min_n_clusters, max_n_clusters)

    inertias = []
    fig = plt.figure()
    fig.set_size_inches(6, 6)
    fig.suptitle(f"Elbow - {nom_alumne}", fontsize=12, y=1)
    ax = fig.add_subplot()

    for k in clusters:
        km = model_kmeans(k)
        km.fit(x_pca)
        inertias.append(km.inertia_)

    ax.plot(clusters, inertias)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia")
    ax.grid(True)

    filename = f"img/elbow_{nom_alumne}.png"
    counter = 1
    while os.path.exists(filename):
        filename = filename.replace(".png", f"_{counter}.png")
        counter += 1
    plt.savefig(filename)

    if show:
        plt.show()

    plt.close()  # Force close this instance


# =========================================================
def calcular_scores(y_test, y_pred, name):
    """
    prints model predicted label scores
    :rtype: object
    :param y_test:
    :param y_pred:
    :param name:
    """
    print(f"\n{name} model:")
    print(f"homogeneïtat -> {metrics.homogeneity_score(y_test, y_pred):.2}")
    print(f"completesa -> {metrics.completeness_score(y_test, y_pred):.2}")
    print(f"v_measure -> {metrics.v_measure_score(y_test, y_pred):.2}")
    print("Confusion matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
