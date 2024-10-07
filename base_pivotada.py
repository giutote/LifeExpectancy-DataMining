import requests
import pandas as pd
import time

file_path = 'BASE_GHO/data_2021_GHO_filtrada.xlsx'

# Read the data from the Excel file
df_full = pd.read_excel(file_path)

# Pivot the table so that GHO values become columns
df_full_pivot = df_full.pivot_table(index=['CountryCode', 'ParentLocationCode', 'Sex'], 
                                    columns='IndicatorCode', 
                                    values=['Value'],
                                    aggfunc='first').reset_index()

# Flatten the multi-index columns
df_full_pivot.columns = ['_'.join(col).strip() for col in df_full_pivot.columns.values]
df_full_pivot = df_full_pivot.rename(columns={'CountryCode_': 'CountryCode', 'ParentLocationCode_': 'ParentLocationCode', 'Sex_': 'Sex'})

df_full_pivot.to_excel('BASE_GHO/data_pivotada_numeric_GHO.xlsx', index=False)

