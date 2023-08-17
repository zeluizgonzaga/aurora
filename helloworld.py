import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# Dados de entrada e saída
entradas = np.array([[0], [1]])
saidas = np.array([[1], [0]])

# Criar o modelo
modelo = Sequential()
modelo.add(Dense(1, input_dim=1, activation='sigmoid'))

# Compilar o modelo
modelo.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
modelo.fit(entradas, saidas, epochs=1000, verbose=0)

# Fazer previsões
entrada_nova = np.array([[0.5]])
previsao = modelo.predict(entrada_nova)

# Imprimir a previsão
print("Previsão:", previsao)