import pandas as pd
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense

"""
Adaptado de:
Jason Brownlee. Deep learning for time series forecasting: predict the
future with MLPs, CNNs and LSTMs in Python. Machine Learning Mastery, 2018, p.84
"""
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# multivariate multi-step data preparation
def mlp_treina_predicao(treino, teste, n_lags, n_predicao):    
    # converte em X e Y
    X_treino, y_treino = split_sequences(treino.values.astype('float32'), n_lags, n_predicao)
    # converte para estrutura [linhas, colunas] para o conjunto de treino
    n_input = X_treino.shape[1] * X_treino.shape[2]
    X_treino = X_treino.reshape((X_treino.shape[0], n_input))
    n_output = y_treino.shape[1] * y_treino.shape[2]
    y_treino = y_treino.reshape((y_treino.shape[0], n_output))

    # Construir e treinar o modelo
    ## camada com 30, 100 e 30 ativações respectivamente
    model = Sequential()
    model.add(Dense(30, activation='relu', input_dim=n_input))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse') 
    model.fit(X_treino, y_treino, epochs=300, verbose=0)
    
    #predição
    teste_modificado = treino[-n_lags:].values.reshape((1, n_input))
    yhat = model.predict(teste_modificado, verbose=0)
    return pd.DataFrame(yhat.reshape((5,11)), index=teste.index, columns=teste.columns)
    
def mlp_EQM(predicao, teste):
    EQM = ((teste - predicao)**2).mean(axis=1)
    REQM = np.sqrt(EQM)
    return pd.DataFrame([EQM,REQM], index=['MSE', "RMSE"]).T