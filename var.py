import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

#criterios de seleção de ordem
def criterios_var(treino, p_max=10):
    dict_criterio_var = {}
    #loop para analisar os criterios de informação e escolher a ordem do modelo
    for ordem in range(1, p_max+1):
        try:
            model_var = VAR(treino.values)
            fitted_model = model_var.fit(ordem, trend='n')
            dict_criterio_var[ordem] = {'AIC':fitted_model.aic, 'BIC': fitted_model.bic, 'HQ':fitted_model.hqic}
        except:
            pass
    return pd.DataFrame(dict_criterio_var).T.round(2)

#aplicação do modelo para predição de n períodos
def var_predicao(treino, ordem, indices_datas, n_predicao=5):
    model = VAR(treino.values)
    fitted_model = model.fit(ordem, trend='n')
    predicao = fitted_model.forecast(y=treino.values[-ordem:], steps=n_predicao)
    predicao_df = pd.DataFrame(predicao, index=indices_datas, columns=treino.columns)
    return predicao_df

#métricas de erro
def var_EQM(predicao, teste):
    EQM = ((teste - predicao)**2).mean(axis=1)
    REQM = np.sqrt(((teste - predicao)**2).mean(axis=1))
    return pd.DataFrame([EQM,REQM], index=['MSE', "RMSE"]).T

