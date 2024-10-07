import pandas as pd
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def process_file(file_path, file_label):
    df = pd.read_excel(file_path)

    # Remover a coluna 'CountryCode' para aplicar os métodos
    X = df.drop(columns=['CountryCode'])

    # Lista para armazenar os coeficientes de silhueta
    silhouette_scores_original = []
    silhouette_scores_pca = []
    silhouette_scores_tsne = []

    # Reduzir as dimensões dos dados para 2D usando PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)

    # Reduzir as dimensões dos dados para 2D usando t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(X)

    # Aplicar K-Medoids e calcular o coeficiente de silhueta para 3 a 10 clusters
    for n_clusters in range(3, 11):
        # K-Medoids nos dados originais
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=0, method='pam')
        labels_original = kmedoids.fit_predict(X)
        score_original = silhouette_score(X, labels_original)
        silhouette_scores_original.append(score_original)

        # K-Medoids nos dados reduzidos por PCA
        labels_pca = kmedoids.fit_predict(pca_result)
        score_pca = silhouette_score(pca_result, labels_pca)
        silhouette_scores_pca.append(score_pca)

        # K-Medoids nos dados reduzidos por t-SNE
        labels_tsne = kmedoids.fit_predict(tsne_result)
        score_tsne = silhouette_score(tsne_result, labels_tsne)
        silhouette_scores_tsne.append(score_tsne)

        # Printar os valores de clusters e coeficientes de silhueta
        print(f"Arquivo: {file_label}")
        print(f"Quantidade de clusters: {n_clusters}")
        print(f"Coeficiente de Silhueta - PCA: {score_pca}")
        print(f"Coeficiente de Silhueta - t-SNE: {score_tsne}")
        print('---')

    # Gráfico do coeficiente de silhueta PCA e t-SNE
    plt.figure(figsize=(10, 6))
    plt.plot(range(3, 11), silhouette_scores_pca, marker='o', linestyle='--', label='PCA')
    plt.plot(range(3, 11), silhouette_scores_tsne, marker='o', linestyle='--', label='t-SNE')
    plt.title(f'Coeficiente de Silhueta para K-Medoids ({file_label})')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Coeficiente de Silhueta')
    plt.legend()
    plt.grid(True)

    # Salvar o gráfico como uma imagem
    plt.savefig(f'GRAFICOS_CLUSTERS/{file_label}_coeficiente_1.png')
    plt.close()

# Processar os três arquivos
process_file('BOTH/both_base_normalizada_40.xlsx', 'both')
process_file('FEMALE/fmle_base_normalizada_40.xlsx', 'fmle')
process_file('MALE/male_base_normalizada_40.xlsx', 'male')
