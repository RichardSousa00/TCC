import pandas as pd # versão 1.5.3
from yahoo_fin import stock_info as si #versão 0.8.9.1
import yfinance as yf #versão 0.2.32
from collections import defaultdict
import itertools
MULT = {'K':1e3, 'M':1e6, 'B':1e9, 'T':1e12}

def obter_constituintes_sp500():
    #obtém lista de símbolos do Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tabela = pd.read_html(url, header=0)[0]
    nomes = tabela['Symbol'].tolist()
    return nomes

def get_info_all_tickers():
    #Obtém lista dos símbolos da S&P
    sp500_symbols = obter_constituintes_sp500()
    data_dict = {}
    # Itere através dos símbolos do S&P 500 e obtenha o valor de mercado e setor
    for symbol in sp500_symbols:
        try:
            quote_table = si.get_quote_table(symbol)
            market_cap = quote_table['Market Cap']
            sector = yf.Ticker(symbol).info['sector']
            data_dict[symbol] = {'Market Cap': market_cap,'Sector': sector}
        except:
            pass
    return data_dict

#data_dict é o dicionário retornado de get_info_all_tickers
def get_topN_each(data_dict,N=10): 
    sector_dict = defaultdict(list)
    top_N_values, top_N_names = {}, {}
    # Remove os multiplicadores K, M, B e T e ordena do maior para o menor
    sorted_data = sorted(data_dict.items(), key=lambda x: float(x[1]['Market Cap'][:-1])*MULT[x[1]['Market Cap'][-1]] if x[1]['Market Cap'][-1] in ['K', 'M', 'B', 'T'] else float(x[1]['Market Cap']), reverse=True)

    # No conjunto ordenado separa por setor
    for symbol, company_data in sorted_data:
        sector = company_data['Sector']
        market_cap_aux = company_data['Market Cap']
        market_cap = float(market_cap_aux[:-1]) * MULT[market_cap_aux[-1]] if market_cap_aux[-1] in MULT else float(market_cap_aux)
        sector_dict[sector].append((symbol, market_cap))

    # Extrai top N de cada setor
    for sector, companies in sector_dict.items():
        info = sorted(companies, key=lambda x: x[1], reverse=True)[:N]
        names = [k[0] for k in info]
        top_N_values[sector] = info
        top_N_names[sector] = names

    lista_N_names = list(itertools.chain.from_iterable(top_N_names.values()))
    dict_nome_por_setor = {setor: [ticker for ticker, _ in empresas] for setor, empresas in top_N_values.items()}
    return top_N_values, lista_N_names, dict_nome_por_setor
    
if __name__ == "__main__":
    data_dict = get_info_all_tickers()
    top_N_values, lista_N_names, dict_nome_por_setor = get_topN_each(data_dict)
    print("\nTop 10 tickers por setor:\n",dict_nome_por_setor)
