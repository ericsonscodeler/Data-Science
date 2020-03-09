import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot


class NeuralNetwork():
    df = None
    rede = None

        
    def __init__(self,file=None):
     
        self.df = pd.read_csv(file)

        
        self.df.columns = ['Produto','Alcool', 'Acido malico', 'cinza','Alcalinidade das cinzas',
        'Magnesio', 'Fenois totais', 'Flavonoides','Fenois não flavonoides ', 'Proantocianinas',
        'Intensidade da cor', 'Matiz','OD315 de vinhos diluídos ', 'Prolina']
        self.df.columns
        print("Menor valor de cada coluna")
        print(self.df.min())
        print("###########################")
        print("Maior valor de cada coluna")
        print(self.df.max())
        print("###########################")
        print("mediana de cada coluna")
        print(self.df.min())
        print("###########################")
        print("media de cada coluna")
        print(self.df.mean())
        print("###########################")
        print("desvio padrao")
        print(self.df.std())


        print("5 primeiras linhas do arquivo csv")       
        print(self.df.head()) #apresenta as 5 primeiras linhas
        print("dimensao do arquivo csv")
        print(self.df.shape) # tupla de dimensao


        ####################################
            ###GRAFICO
        ####################################

        y = np.linspace(80,160,20)
        x = np.linspace(11,15,20)
        plt.xlabel(u'Alcool')
        plt.ylabel(u'Magnesio')
        self.df.plot(kind='scatter',x='Alcool',y='Magnesio')
        




        #plt.plot(x,y)
        plt.show(True)


        

        

    # def menu(self):
    #     while(True):
    #        os.system("cls")
    #        print("#######REDE NEURAL PARA PREVISAO DE PRODUTORES DE VINHO ########")
    #        print("1 - Exibir dados originais e estatísticas")
    #        print("2 - Imprimir grafico de Teor Alcoolico / Magnesio / Fenois por produtor")         
    #        print("3 - Exibir previsao da Rede Neural")
    #        choice =input()

    #        if choice == '1':
    #            self.__init__()


rede = NeuralNetwork("wine.csv")