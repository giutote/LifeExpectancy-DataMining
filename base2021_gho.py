import requests
import pandas as pd
import time

# URL da API para obter a lista de todos os códigos de indicadores
url_codes = "https://ghoapi.azureedge.net/api/Indicator"

# Fazendo a requisição para obter a lista de códigos
response_codes = requests.get(url_codes)

# Verificando se a requisição foi bem-sucedida
if response_codes.status_code == 200:
    # Convertendo a resposta JSON
    data_codes = response_codes.json()
    
    # Extraindo a lista de códigos de indicadores
    codes = [entry['IndicatorCode'] for entry in data_codes['value']]
    
    # Lista para armazenar todos os dados
    all_data = []
    
    # Fazendo requisição para cada código de indicador com filtro de ano 2021
    for code in codes:
        url = f"https://ghoapi.azureedge.net/api/{code}?$filter=TimeDim eq 2021"
        try:
            response = requests.get(url, timeout=30)
            
            # Verificando se a requisição foi bem-sucedida
            if response.status_code == 200:
                data = response.json()
                facts = data.get('value', [])
                
                # Processando cada fato para extrair apenas as colunas desejadas
                for fact in facts:
                    entry = {
                        'IndicatorCode': fact.get('IndicatorCode'),
                        'SubGroup': fact.get('Dim2'),
                        'SubGroup2': fact.get('Dim3'),
                        'IndicatorName': fact.get('IndicatorName'),
                        'CountryCode': fact.get('SpatialDim'),
                        'ParentLocationCode':fact.get('ParentLocationCode'),
                        'ParentLocation':fact.get('ParentLocation'),
                        'Year': fact.get('TimeDim'),
                        'Sex': fact.get('Dim1'),
                        'Value': fact.get('NumericValue')
                    }
                    
                    all_data.append(entry)
            
            # Adicionando um pequeno atraso entre as requisições para evitar erros
            time.sleep(1)
        
        except requests.exceptions.RequestException as e:
            print(f"Erro ao acessar {url}: {e}")
    
    # Convertendo a lista processada em um DataFrame
    df = pd.DataFrame(all_data)
    
    # Selecionando apenas as colunas desejadas (se existirem)
    desired_columns = ['IndicatorCode','SubGroup','SubGroup2','IndicatorName','CountryCode','ParentLocationCode','ParentLocation','Year','Sex','Value']
    df = df[[col for col in desired_columns if col in df.columns]]
    
    # Salvando o DataFrame em um arquivo Excel e adicionando possibilidade de erro devido algumas consultas que podem dar erro e interromper o processamento
    try:
        df.to_excel("BASE_GHO/data_2021_GHO.xlsx", index=False)
        print("Arquivo Excel gerado com sucesso: data_2021_GHO.xlsx")
        
        # Exibindo o DataFrame gerado para verificação
        print(df.head())
    except ModuleNotFoundError as e:
        print("Erro de módulo:", e)
else:
    print("Falha na requisição para obter a lista de códigos:", response_codes.status_code)
