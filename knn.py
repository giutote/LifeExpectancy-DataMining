import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# Carregar o dataset
file_path = 'MALE/male_70.xlsx'
df = pd.read_excel(file_path)

# Especificar as colunas categóricas que devem ser codificadas
categorical_cols = ['ParentLocationCode']
numeric_cols = df.columns.difference(categorical_cols + ['CountryCode'] + ['CATEG_LIFE_EXPECTANCY'])

# Codificar as colunas categóricas
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Aplicar o imputador KNN somente às colunas numéricas
imputer = KNNImputer(n_neighbors=5, weights='uniform')
df[numeric_cols] = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Reverter a codificação das colunas categóricas
for col in categorical_cols:
    le = label_encoders[col]
    df[col] = le.inverse_transform(df[col])

df.to_excel("male_knn.xlsx", index=False)
