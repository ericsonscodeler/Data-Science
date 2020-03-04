import pandas as pd
import numpy as np
import os 

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class NeuralNetwork():
    df = None
    rede = None
    def __init__(self,file=None):
        self.df = pd.read_csv(file)
        #print(self.df)

        # print(self.df.head()) #apresenta as 5 primeiras linhas
        # print(self.df.shape) # tupla de dimensao 
        print(self.df.max())
        print(self.df.min(axis=0))




    # def menu(self):
    #     while(True):
    #        os.system("cls")
    #        print("#######REDE NEURAL PARA PREVISAO DE PRODUTORES DE VINHO ########")
    #        print("1 - Exibir dados originais e estatísticas")         



rede = NeuralNetwork("wine.csv")