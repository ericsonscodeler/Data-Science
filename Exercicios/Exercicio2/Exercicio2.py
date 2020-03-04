#Ericson Scodeler

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

import pandas as pd;

dados = pd.read_csv("tracking.csv",);

treino = dados[['home','how_it_works','contact']];
resultado = dados['bought'];

treino_x = treino[0:75];
treino_y = resultado[0:75];

rede = MLPClassifier();

rede.fit(treino_x, treino_y);

teste_x = treino[75:100];
teste_y = resultado[75:100];

previsao_y = rede.predict(teste_x);
print(previsao_y);
print(accuracy_score(teste_y,previsao_y)* 100, '%');
print(confusion_matrix(teste_y,previsao_y));