import pandas as pd 



class NeuralNetwork():
       
    def __init__(self):
        self.portaOU()
    

    def portaOU(self):
        dataOR = {'x1': [0,0,1,1],
                  'x2': [0,1,0,1],
                  'y' : [0,1,1,1]}

        df = pd.DataFrame(dataOR,columns = ['x1','x2','y'])
        print(df)

    # def portaAND(self):
    #     dataAND = {'x1': [0,0,1,1],
    #                'x2': [0,1,0,1],
    #                'y' : [0,0,0,1]}

    #     print(df)        

         



