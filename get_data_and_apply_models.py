import pandas as pd
import numpy as np
import yfinance as yf
from top_tickers_in_sectors import get_info_all_tickers, get_topN_each
from var import criterios_var, var_predicao, var_EQM
from varma import criterios_varma, varma_predicao_func, varma_EQM
from mlp import mlp_treina_predicao, mlp_EQM

############ Download de precos dos top tickers de cada setor ############

### Previamente executado ###
# dict_nome_por_setor =  {
#     'Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CSCO', 'CRM', 'ACN', 'AMD'], 
#     'Communication Services': ['GOOG', 'META', 'NFLX', 'CMCSA', 'TMUS', 'DIS', 'VZ', 'T', 'CHTR', 'EA'], 
#     'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'ABNB'], 
#     'Financial Services': ['BRK-B', 'V', 'JPM', 'MA', 'BAC', 'WFC', 'MS', 'BX', 'SPGI', 'AXP'], 
#     'Healthcare': ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'PFE', 'ABT', 'DHR', 'AMGN'], 
#     'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PXD', 'OXY', 'PSX', 'HES'], 
#     'Consumer Defensive': ['WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MDLZ', 'MO', 'CL', 'MNST'], 
#     'Basic Materials': ['LIN', 'APD', 'SHW', 'FCX', 'ECL', 'CTVA', 'NUE', 'DOW', 'DD', 'NEM'], 
#     'Industrials': ['UPS', 'UNP', 'HON', 'GE', 'LMT', 'BA', 'DE', 'RTX', 'ADP', 'ETN'], 
#     'Utilities': ['NEE', 'SO', 'DUK', 'SRE', 'PCG', 'EXC', 'AEP', 'CEG', 'D', 'XEL'], 
#     'Real Estate': ['PLD', 'AMT', 'EQIX', 'PSA', 'WELL', 'SPG', 'CCI', 'DLR', 'O', 'CSGP']
# }

data_dict = get_info_all_tickers()
top_N_values, lista_N_names, dict_nome_por_setor = get_topN_each(data_dict)

date_init = '2021-12-31'
date_end = '2023-09-30'
precos_all = {}
# extrai preços de cada ação agrupado por setor
for setor in dict_nome_por_setor.keys():
    precos_all[setor] = yf.download(dict_nome_por_setor[setor],date_init,date_end, auto_adjust=True)['Close']

df_teste = pd.DataFrame()
# extrai o índice temporal de uma das séries para auxílio
df_teste.index = precos_all['Technology'].index
for setor in dict_nome_por_setor.keys():
    #calcula o retorno líquido simples e obtém a média por setor
    df_teste[setor] = precos_all[setor].pct_change().mean(axis=1)
retorno_setores = df_teste.dropna()

#separaçõa das janelas de dados
treino1, teste1 = retorno_setores[:130], retorno_setores[130:135]
treino2, teste2 = retorno_setores[:260], retorno_setores[260:265]
treino3, teste3 = retorno_setores[:-5], retorno_setores[-5:]


############ Modelo VAR -- seleção de ordem e predição ############


#criterios para as 3 janelas de dados
print(criterios_var(treino1))
print(criterios_var(treino2))
print(criterios_var(treino3))

teste_datas1 = teste1.index
teste_datas2 = teste2.index
teste_datas3 = teste3.index

#VAR(1) para a primeira janela de dados de treinamento
var_predicao_df1 = var_predicao(treino1.copy(), 1, teste_datas1)
print(var_EQM(var_predicao_df1, teste1))
#VAR(1) para a segunda janela de dados de treinamento
var_predicao_df2 = var_predicao(treino2.copy(), 1, teste_datas2)
print(var_EQM(var_predicao_df2, teste2))
#VAR(1) para a terceira janela de dados de treinamento
var_predicao_df3 = var_predicao(treino3.copy(), 1, teste_datas3)
print(var_EQM(var_predicao_df3, teste3))


############ Modelo VARMA -- seleção de ordem e predição ############


criterios_varma1 = criterios_varma(treino1)
criterios_varma2 = criterios_varma(treino2)
criterios_varma3 = criterios_varma(treino3)
print(criterios_varma1)
print(criterios_varma2)
print(criterios_varma3)

#VARMA(1,1) para a primeira janela de dados de treinamento
varma_predicao1 = varma_predicao_func(treino1, 1,1, teste1.index)
print(varma_EQM(varma_predicao1, teste1))
#VARMA(1,1) para a segunda janela de dados de treinamento
varma_predicao2 = varma_predicao_func(treino2, 1,1, teste2.index)
print(varma_EQM(varma_predicao2, teste2))
#VARMA(1,1) para a terceira janela de dados de treinamento
varma_predicao3 = varma_predicao_func(treino3, 1,1, teste3.index)
print(varma_EQM(varma_predicao3, teste3))


############ Modelo MLP -- remodelagem dos dados e predição ############


np.random.seed(71)
#conjunto de treino 1, usa 1 dia anterior para prever 5
predicao_MLP1_1 = mlp_treina_predicao(treino1, teste1, 1,5)
print(mlp_EQM(predicao_MLP1_1, teste1))

#conjunto de treino 2, usa 1 dia anterior para prever 5
predicao_MLP2_1 = mlp_treina_predicao(treino2, teste2, 1,5)
print(mlp_EQM(predicao_MLP2_1, teste2))

#conjunto de treino 3, usa 1 dia anterior para prever 5
predicao_MLP3_1 = mlp_treina_predicao(treino3, teste3, 1,5)
print(mlp_EQM(predicao_MLP3_1, teste3))

#conjunto de treino 1, usa 5 dias anterior para prever 5
predicao_MLP1_5 = mlp_treina_predicao(treino1, teste1, 5,5)
print(mlp_EQM(predicao_MLP1_5, teste1))

#conjunto de treino 2, usa 5 dias anterior para prever 5
predicao_MLP2_5 = mlp_treina_predicao(treino2, teste2, 5,5)
print(mlp_EQM(predicao_MLP2_5, teste2))

#conjunto de treino 3, usa 5 dias anterior para prever 5
predicao_MLP3_5 = mlp_treina_predicao(treino3, teste3, 5,5)
print(mlp_EQM(predicao_MLP3_5, teste3))
