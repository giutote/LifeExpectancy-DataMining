from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Carregar o dataset
file_path = 'MALE/male_knn.xlsx'
df = pd.read_excel(file_path)

# Definir a coluna target
target_column = 'CATEG_LIFE_EXPECTANCY'

# Selecionar as colunas numéricas para normalização (excluindo a coluna target)
numeric_cols = df.select_dtypes(include=['number']).columns.difference([target_column])

# Aplicar a normalização Z-Score
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Salvar a base normalizada em um novo arquivo CSV
df.to_excel("male_base_normalizada.xlsx", index=False)

# Exibir as primeiras linhas da base normalizada (opcional)
print(df.head())

# Definir X (atributos) e y (variável alvo)
X = df[numeric_cols]
y = df[target_column]

# Treinando o modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Obter a importância dos atributos
importances = model.feature_importances_

# Criar um DataFrame para visualizar as importâncias
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Salvar a lista de atributos e seus pesos
feature_importance_df.to_excel("male_importancia_atributos.xlsx", index=False)
