import statsmodels.api as sm   
import numpy as np #1.26.4
import pandas as pd 
        
def criterios_varma(treino, p_max=2, q_max=2):
    #loop para analisar os criterios de informação e escolher a ordem do modelo
    lista_criterios = []
    for ordem_p in range(1, p_max+1):
        for ordem_q in range(1, q_max+1):
            #VARMA é um caso especial do VARMAX com nenhuma variável X (exógena)
            model_varma = sm.tsa.VARMAX(treino.values, order=(ordem_p, ordem_q), trend='n')
            try:
                fitted_model = model_varma.fit()
                lista_criterios.append({'p':ordem_p,'q':ordem_q,'AIC':fitted_model.aic, 'BIC': fitted_model.bic, 'HQ':fitted_model.hqic})
            except:pass
    return pd.DataFrame(lista_criterios).round(2)

def varma_predicao_func(treino, p,q, indices_datas, n_predicao=5):
    model = sm.tsa.VARMAX(treino.values, order=(p,q), trend='n')
    fitted_model = model.fit(method_kwargs={'maxiter':200})
    predicao = fitted_model.forecast(steps=n_predicao)
    predicao_df = pd.DataFrame(predicao, index=indices_datas, columns=treino.columns)
    return predicao_df

def varma_EQM(predicao, teste):
    EQM = ((teste - predicao)**2).mean(axis=1)
    REQM = np.sqrt(EQM)
    return pd.DataFrame([EQM,REQM], index=['MSE', "RMSE"]).T
