# apyori template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transactions=[]  
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j])for j in range(0,20)])

#training Apriori on the dataset
from apyori import apriori
rules =apriori(transactions , min_support =0.003  , min_confifence =0.2 , min_lift=3, min_lenght = 2 )

results = list(rules)
result_list=[]
for i in range(0,len(results)):
    result_list.append('Rules:\t'+str(results[i][0]) + '\n Support:\t'+str(results[i][1]))
    
    