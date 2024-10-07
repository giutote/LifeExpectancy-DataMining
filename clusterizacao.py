import pandas as pd
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def apply_best_clustering(file_path, file_label, filter_file_path, method, n_clusters):
    # Carregar os dados
    df = pd.read_excel(file_path)
    df_filter = pd.read_excel(filter_file_path)

    # Remover a coluna 'CountryCode' para aplicar os métodos
    X = df.drop(columns=['CountryCode'])

    if method == 'PCA':
        # Reduzir as dimensões dos dados para 2D usando PCA
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(X)
    elif method == 't-SNE':
        # Reduzir as dimensões dos dados para 2D usando t-SNE
        tsne = TSNE(n_components=2, random_state=0)
        reduced_data = tsne.fit_transform(X)

    # Aplicar K-Medoids com o melhor número de clusters
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=0, method='pam')
    df['Cluster'] = kmedoids.fit_predict(reduced_data)

    # Calcular o coeficiente de silhueta
    silhouette_score_value = silhouette_score(reduced_data, df['Cluster'])
    print(f"Coeficiente de Silhueta para {file_label} ({method} com {n_clusters} clusters): {silhouette_score_value}")

    # Plotar os clusters com CountryCodes
    plt.figure(figsize=(10, 8))
    for cluster_id in df['Cluster'].unique():
        cluster_data = reduced_data[df['Cluster'] == cluster_id]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_id}')

        # Adicionar os CountryCodes ao gráfico
        for i, txt in enumerate(df['CountryCode'][df['Cluster'] == cluster_id]):
            plt.text(cluster_data[i, 0], cluster_data[i, 1], txt, fontsize=8, ha='right')

    plt.title(f'Distribuição dos Países nos Clusters ({method}) - {file_label}')
    plt.xlabel(f'{method}1')
    plt.ylabel(f'{method}2')
    plt.legend()
    plt.grid(True)

    # Salvar o gráfico como uma imagem
    plt.savefig(f'{file_label}_cluster_plot.png')
    plt.close()

    # Fazer o merge com a base original pelos 'CountryCode' e salvar por cluster
    df_merged = df[['CountryCode', 'Cluster']].merge(df_filter, on='CountryCode')

    for cluster_id in df['Cluster'].unique():
        df_clustered = df_merged[df_merged['Cluster'] == cluster_id]
        df_clustered.to_excel(f'{file_label}_base_cluster_{cluster_id}.xlsx', index=False)

    print(f"Clusterização e associação concluídas para {file_label}, gráficos, bases e arquivos de associação salvos.")

# Aplicar a melhor clusterização para cada base
apply_best_clustering('BOTH/both_base_normalizada_40.xlsx', 'BOTH', 'BOTH/both_filtro_atributos.xlsx', 'PCA', 3)
apply_best_clustering('FEMALE/fmle_base_normalizada_40.xlsx', 'FEMALE', 'FEMALE/fmle_filtro_atributos.xlsx', 't-SNE', 3)
apply_best_clustering('MALE/male_base_normalizada_40.xlsx', 'MALE', 'MALE/male_filtro_atributos.xlsx', 't-SNE', 4)
