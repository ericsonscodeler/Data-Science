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

        self.df.columns = ['Produtor','Alcool', 'Acido malico', 'cinza','Alcalinidade das cinzas',
        'Magnesio', 'Fenois totais', 'Flavonoides','Fenois não flavonoides ', 'Proantocianinas',
        'Intensidade da cor', 'Matiz','OD315 de vinhos diluídos ', 'Prolina']
        ##self.df.columns
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
        
        #self.menu()
        self.grafico()
        self.previsao_rede_neural()
        


        ####################################
            ###GRAFICO
        ####################################
    def grafico(self):

        y = np.linspace(80,160,20)
        x = np.linspace(11,15,20)
        plt.xlabel(u'Alcool')
        plt.ylabel(u'Magnesio')
        self.df.plot(kind='scatter',x='Alcool',y='Magnesio',color=self.df['Produtor'],s=self.df['Fenois totais']*50)


        plt.legend()
        
        plt.show(True)


    def previsao_rede_neural(self):
        
        produtor1 = self.df.loc[(self.df['Produtor']) == 1]
        produtor2 = self.df.loc[(self.df['Produtor']) == 2]
        produtor3 = self.df.loc[(self.df['Produtor']) == 3]

        produtor1_x = produtor1.drop('Produtor',axis=1)
        produtor1_y = produtor1['Produtor']

        produtor2_x = produtor2.drop('Produtor',axis=1)
        produtor2_y = produtor2['Produtor']

        produtor3_x = produtor3.drop('Produtor',axis=1)
        produtor3_y = produtor3['Produtor']

        treino_produtor1_x,teste_produtor1_x,treino_produtor1_y,teste_produtor1_y = train_test_split(produtor1_x,produtor1_y,test_size=0.25,stratify=produtor1_y)
        treino_produtor2_x,teste_produtor2_x,treino_produtor2_y,teste_produtor2_y = train_test_split(produtor2_x,produtor2_y,test_size=0.25,stratify=produtor2_y)
        treino_produtor3_x,teste_produtor3_x,treino_produtor3_y,teste_produtor3_y = train_test_split(produtor3_x,produtor3_y,test_size=0.25,stratify=produtor3_y)

        self.rede = MLPClassifier()
        self.rede.fit(treino_produtor1_x,treino_produtor1_y)
        self.rede.fit(treino_produtor2_x,treino_produtor2_y)
        self.rede.fit(treino_produtor3_x,treino_produtor3_y)

        produtor1_previsao_y = self.rede.predict(teste_produtor1_x)
        produtor2_previsao_y = self.rede.predict(teste_produtor2_x)
        produtor3_previsao_y = self.rede.predict(teste_produtor3_x)

        print("######## PRODUTOR 1 ########")
        print(accuracy_score(teste_produtor1_y,produtor1_previsao_y)* 100)
        print(confusion_matrix(teste_produtor1_y,produtor1_previsao_y))

        print("######## PRODUTOR 2 ########")
        print(accuracy_score(teste_produtor2_y,produtor2_previsao_y)* 100)
        print(confusion_matrix(teste_produtor2_y,produtor2_previsao_y))

        print("######## PRODUTOR 3 ########")
        print(accuracy_score(teste_produtor3_y,produtor3_previsao_y)* 100)
        print(confusion_matrix(teste_produtor3_y,produtor3_previsao_y))

        


    def menu(self):
        while(True):
           os.system("cls")
           print("#######REDE NEURAL PARA PREVISAO DE PRODUTORES DE VINHO ########")
           print("1 - Exibir dados originais e estatísticas")
           print("2 - Imprimir grafico de Teor Alcoolico / Magnesio / Fenois por produtor")         
           print("3 - Exibir previsao da Rede Neural")
           choice = input()

           if choice == '1':
                self.__init__(self) 
           elif choice == '2':
                self.grafico(self)
           elif choice == '3':
                self.previsao_rede_neural(self)




rede = NeuralNetwork("wine.csv")