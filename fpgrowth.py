import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Discretizar os atributos e obter intervalos
def discretize_attributes(df, n_bins=3):  # Reduzido de 3
    df_discretized = df.drop(columns=['CountryCode', 'Cluster'], errors='ignore')
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    discretizer.fit(df_discretized)
    bin_edges = discretizer.bin_edges_

    df_discretized_intervals = pd.DataFrame()
    for i, col in enumerate(df_discretized.columns):
        bins = bin_edges[i]
        labels = [f'{bins[j]:.2f}-{bins[j+1]:.2f}' for j in range(len(bins)-1)]
        df_discretized_intervals[col] = pd.cut(df_discretized[col], bins=bins, labels=labels, include_lowest=True)
    
    return df_discretized_intervals

# One-hot encoding
def convert_to_one_hot(df_discretized):
    one_hot_df = pd.get_dummies(df_discretized.astype(str))
    return one_hot_df

# Regras de associação com consequente restrito às colunas de expectativa de vida
def generate_association_rules(df_onehot, min_support=0.5, top_n=30, target_columns=None):
    frequent_itemsets = fpgrowth(df_onehot, min_support=min_support, use_colnames=True, max_len=3)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    if target_columns:
        rules = rules[rules['consequents'].apply(lambda x: all(item in target_columns for item in x))]
    top_rules = rules.nlargest(top_n, 'confidence')
    return top_rules


# Carregar as bases e processar
def process_file(file_path, output_file):
    df = pd.read_excel(file_path)
    print(f"Excel {file_path} lido")
    
    # Discretizar os atributos
    df_discretized = discretize_attributes(df)
    print(f'Discretized {file_path}')
    
    # Converter para one-hot encoding
    df_onehot = convert_to_one_hot(df_discretized)
    print(f'One hot encoding aplicado em {file_path}')
    
    # Imprimir as colunas com 'Life expectancy at birth (years)' após o one-hot encoding para validacao
    columns_with_value_whosis = [col for col in df_onehot.columns if col.startswith('Life expectancy at birth (years)')]
    print(f"Colunas one-hot que começam com 'Life expectancy at birth (years)':\n{columns_with_value_whosis}")
    
    # Gerar as melhores regras de associação
    top_rules = generate_association_rules(df_onehot, min_support=0.001, target_columns=columns_with_value_whosis)
    
    # Salvar as regras em um arquivo csv
    top_rules.to_csv(output_file, index=False)
    print(f'Regras salvas em {output_file}')

# Processando todos os arquivos
process_file('MALE/clusters/MALE_base_cluster_0.xlsx', 'MALE/associacao/male_top_regras_associacao_0.csv')
process_file('MALE/clusters/MALE_base_cluster_1.xlsx', 'MALE/associacao/male_top_regras_associacao_1.csv')
process_file('MALE/clusters/MALE_base_cluster_2.xlsx', 'MALE/associacao/male_top_regras_associacao_2.csv')
process_file('MALE/clusters/MALE_base_cluster_3.xlsx', 'MALE/associacao/male_top_regras_associacao_3.csv')
process_file('FEMALE/clusters/FEMALE_base_cluster_0.xlsx', 'FEMALE/associacao/fmle_top_regras_associacao_0.csv')
process_file('FEMALE/clusters/FEMALE_base_cluster_1.xlsx', 'FEMALE/associacao/fmle_top_regras_associacao_1.csv')
process_file('FEMALE/clusters/FEMALE_base_cluster_2.xlsx', 'FEMALE/associacao/fmle_top_regras_associacao_2.csv')
process_file('BOTH/clusters/BOTH_base_cluster_0.xlsx', 'BOTH/associacao/both_top_regras_associacao_0.csv')
process_file('BOTH/clusters/BOTH_base_cluster_1.xlsx', 'BOTH/associacao/both_top_regras_associacao_1.csv')
process_file('BOTH/clusters/BOTH_base_cluster_2.xlsx', 'BOTH/associacao/both_top_regras_associacao_2.csv')