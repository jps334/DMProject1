# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:37:54 2017

@author: João Silva
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
"""
import dataset
"""

sixpm = pd.read_excel("C:\\Users\\mboni\\OneDrive\\Área de Trabalho\\NOVA IMS\\DM\\Group_01_SixPM_Cluster.xlsx")



"""PRE PROCESSING"""

print(sixpm.info())

print(sixpm.describe())


print(sixpm.describe(include=['O']))


sixpm['MntAthletic']= sixpm['MntAthletic'].fillna(0)

sixpm['MntClothing']= sixpm['MntClothing'].fillna(0)

"""CHARTS"""
sixpm_no_missing=sixpm.dropna()
box_Income = sns.boxplot(x="Income", data = sixpm_no_missing)
plt.title("Boxplot of Income")
plt.show(box_Income)


labels = 'Acessories', 'Bags', 'Clothing', 'Athletic', 'Shoes', 'Premium'
proportions = [sixpm['MntAcessories'].sum(), sixpm['MntBags'].sum(), sixpm['MntClothing'].sum(),\
               sixpm['MntAthletic'].sum(), sixpm['MntShoes'].sum(), sixpm['MntPremiumProds'].sum()]
colors = ['lightcoral', 'crimson', 'wheat','brown',  'sandybrown', 'sienna']
explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
plt.pie(proportions, explode=explode, labels=labels, colors=colors, shadow=False, startangle=90)
plt.axis('equal')
plt.title('Amounts per type of product')
plt.show()


labels = 'Web', 'Catalog', 'Store'
proportions = [sixpm['NumWebPurchases'].sum(),\
        sixpm['NumCatalogPurchases'].sum(), sixpm['NumStorePurchases'].sum()]
colors = ['goldenrod', 'khaki', 'beige']
explode = (0.05, 0.05, 0.05)
plt.pie(proportions, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=False, startangle=90)
plt.axis('equal')
plt.title('Purchases per method')
plt.show()


Kidhome=sns.countplot(sixpm['Kidhome'], color = "coral")
plt.title("Kidhome frequency")
plt.show(Kidhome)


Teenhome=sns.countplot(sixpm['Teenhome'], color= "steelblue")
plt.title("Teenhome frequency")
plt.show(Teenhome)


labels =('Cmp1', 'Cmp2', 'Cmp3', 'Cmp4', 'Cmp5')
data = [sixpm['AcceptedCmp1'].sum(), sixpm['AcceptedCmp2'].sum(),\
        sixpm['AcceptedCmp3'].sum(), sixpm['AcceptedCmp4'].sum(), sixpm['AcceptedCmp5'].sum()]
plt.bar(np.arange(len(labels)), data, align = 'center', alpha=0.5)
plt.xticks(np.arange(len(labels)), labels)
plt.ylabel('Nr Accepted Campaigns')
plt.title('Accepted Campaigns')
plt.show()


print(sixpm['Recency'].describe())

Recency=sns.countplot(sixpm['Recency'])
plt.title("Recency frequency")
plt.show(Recency)

print(sixpm['Complain'].mean())

print(sixpm['Complain'].value_counts())
print(sixpm['Complain'].describe())


print(sixpm.describe(include=['O']))


colors = [  "darkseagreen",  "salmon", "lightblue"]
box_income_per_gen = sns.boxplot(x = 'Generation', y = 'Income',\
                                  data = sixpm_no_missing, palette= colors)
plt.show(box_income_per_gen)



colors = [  "darkseagreen",  "salmon", "lightblue", "grey", "plum"]
box_income_per_educ = sns.boxplot(x = "Education", y = 'Income',\
                                  data = sixpm_no_missing, palette= colors)
plt.show(box_income_per_educ)




"""
Preencher Income utilizando regressão múltipla
"""
sixpm['Income']= sixpm['Income'].fillna(100000000)

import numpy as np
sixpm['Generation'] = np.where((sixpm['Year_Birth']<1964) & \
     (sixpm['Year_Birth']>1940), 'BabyBoomer', 'GenZ')
sixpm['Generation'] = np.where((sixpm['Year_Birth']<1985) & \
     (sixpm['Year_Birth']>=1964), 'GenX', sixpm['Generation'])
sixpm['Generation'] = np.where((sixpm['Year_Birth']<2000) & \
     (sixpm['Year_Birth']>=1985), 'Millenial', sixpm['Generation'])
sixpm['Generation'] = np.where((sixpm['Year_Birth']<2020) & \
     (sixpm['Year_Birth']>=2000), 'GenZ', sixpm['Generation'])


sixpm['Income'] = np.where((sixpm['Income']==100000000) & (sixpm['Education']=='Basic') &\
                 (sixpm['Generation']=='BabyBoomer'), 31870.95, sixpm['Income']) 
sixpm['Income'] = np.where((sixpm['Income']==100000000) & (sixpm['Education']=='2n Cycle') &\
                 (sixpm['Generation']=='BabyBoomer'), 68784.61, sixpm['Income'])
sixpm['Income'] = np.where((sixpm['Income']==100000000) & (sixpm['Education']=='Graduation') &\
                 (sixpm['Generation']=='BabyBoomer'), 77335.12, sixpm['Income'])
sixpm['Income'] = np.where((sixpm['Income']==100000000) & (sixpm['Education']=='Master') &\
                 (sixpm['Generation']=='BabyBoomer'), 81008.1, sixpm['Income'])
sixpm['Income'] = np.where((sixpm['Income']==100000000) & (sixpm['Education']=='PhD') &\
                 (sixpm['Generation']=='BabyBoomer'), 84634.62, sixpm['Income'])

sixpm['Income'] = np.where((sixpm['Income']==100000000) & (sixpm['Education']=='Basic') &\
                 (sixpm['Generation']=='GenX'), 27472.52, sixpm['Income'])
sixpm['Income'] = np.where((sixpm['Income']==100000000) & (sixpm['Education']=='2n Cycle') &\
                 (sixpm['Generation']=='GenX'), 59291.82, sixpm['Income'])
sixpm['Income'] = np.where((sixpm['Income']==100000000) & (sixpm['Education']=='Graduation') &\
                 (sixpm['Generation']=='GenX'), 66662.29, sixpm['Income'])
sixpm['Income'] = np.where((sixpm['Income']==100000000) & (sixpm['Education']=='Master') &\
                 (sixpm['Generation']=='GenX'), 69828.38, sixpm['Income'])
sixpm['Income'] = np.where((sixpm['Income']==100000000) & (sixpm['Education']=='PhD') &\
                 (sixpm['Generation']=='GenX'), 72954.40428, sixpm['Income'])

sixpm['Income'] = np.where((sixpm['Income']==100000000) & (sixpm['Education']=='Basic') &\
                 (sixpm['Generation']=='Millenial'), 26074.54, sixpm['Income'])
sixpm['Income'] = np.where((sixpm['Income']==100000000) & (sixpm['Education']=='2n Cycle') &\
                 (sixpm['Generation']=='Millenial'), 56274.67, sixpm['Income'])
sixpm['Income'] = np.where((sixpm['Income']==100000000) & (sixpm['Education']=='Graduation') &\
                 (sixpm['Generation']=='Millenial'), 63270.08, sixpm['Income'])
sixpm['Income'] = np.where((sixpm['Income']==100000000) & (sixpm['Education']=='Master') &\
                 (sixpm['Generation']=='Millenial'), 66275.06, sixpm['Income'])
sixpm['Income'] = np.where((sixpm['Income']==100000000) & (sixpm['Education']=='PhD') &\
                 (sixpm['Generation']=='Millenial'), 69242.01, sixpm['Income'])

print (sixpm.isnull().sum())

"""Correlation"""
sixpm.corr()

"""New variables"""

sixpm['Amnt_Spent']= sixpm['MntBags'] + sixpm['MntAcessories'] + sixpm['MntClothing'] \
                    + sixpm['MntShoes'] + sixpm['MntAthletic'] + sixpm['MntPremiumProds']

sixpm['Per_Income_Spent'] = (sixpm['Amnt_Spent']/sixpm['Income'])*100

sixpm['Num_Purchases']= sixpm['NumCatalogPurchases'] + sixpm['NumStorePurchases'] + sixpm['NumWebPurchases']


""" Retiramos do dataset o que não era relevante"""
sixpm.drop(['Group'], axis = 1, inplace = True)
sixpm.drop(['Element1'], axis = 1, inplace = True)
sixpm.drop(['Element2'], axis = 1, inplace = True)
sixpm.drop(['Element3'], axis = 1, inplace = True)
sixpm.drop(['Custid'], axis = 1, inplace = True)



""" Assumindo que caso haja amounts spent e nao haja purchases, então é porque houve uma compra de uma forma que não foi registada"""

sixpm['NumOtherPurchases'] = np.where((sixpm['Amnt_Spent']>0) & (sixpm['Num_Purchases']==0), 1, 0)

sixpm['Num_Purchases']= sixpm['NumCatalogPurchases'] + sixpm['NumStorePurchases'] + sixpm['NumWebPurchases'] + sixpm['NumOtherPurchases']


import datetime
from datetime import date
sixpm['TimeCustomer'] = sixpm['Custid']
for x,y in sixpm['Dt_Customer'].iteritems():
    y = date.today().year - y.date().year
    sixpm['TimeCustomer'] = sixpm['TimeCustomer'].set_value(x,y)
    
    
sixpm['Age']=date.today().year -sixpm.Year_Birth

sixpm['Avg_Spent'] = (sixpm['Amnt_Spent']/sixpm['Num_Purchases'])

sixpm['NumDealsfficency'] = (sixpm['NumDealsPurchases']/sixpm['Num_Purchases'])

sixpm['Web_efficency'] = ((sixpm['NumWebPurchases']/12)/sixpm['NumWebVisitsMonth'])


sixpm['Kidhome_Flag'] = np.where((sixpm['Kidhome']>0), 1, 0)


sixpm['Teenhome_Flag'] = np.where((sixpm['Teenhome']>0), 1, 0)


sixpm['Type_of_Customer'] = np.where((sixpm['Amnt_Spent']>0) & (sixpm['TimeCustomer']==1), 'New', 'Regular')

sixpm['Marital_Status_test'] = np.where((sixpm['Marital_Status']=='Divorced') | \
     (sixpm['Marital_Status']=='Widow'), 'Post_Marriage', sixpm['Marital_Status'])

sixpm['Marital_Status_test'] = np.where((sixpm['Marital_Status']=='Together') | \
     (sixpm['Marital_Status']=='Married'), 'With Someone', sixpm['Marital_Status_test'])



"""Normalizacoes"""
 
sixpm['Income_norm'] = np.log(sixpm['Income'])

sixpm['MntAcessories_norm'] = (sixpm['MntAcessories']-sixpm['MntAcessories'].min())/ \
                            (sixpm['MntAcessories'].max()-sixpm['MntAcessories'].min())
sixpm['MntClothing_norm'] = (sixpm['MntClothing']-sixpm['MntClothing'].min())/ \
                            (sixpm['MntClothing'].max()-sixpm['MntClothing'].min())
sixpm['MntBags_norm'] = (sixpm['MntBags']-sixpm['MntBags'].min())/ \
                            (sixpm['MntBags'].max()-sixpm['MntBags'].min())
sixpm['MntAthletic_norm'] = (sixpm['MntAthletic']-sixpm['MntAthletic'].min())/ \
                        (sixpm['MntAthletic'].max()-sixpm['MntAthletic'].min())
sixpm['MntPremiumProds_norm'] = (sixpm['MntPremiumProds']-sixpm['MntPremiumProds'].min())/ \
                        (sixpm['MntPremiumProds'].max()-sixpm['MntPremiumProds'].min())
sixpm['MntShoes_norm'] = (sixpm['MntShoes']-sixpm['MntShoes'].min())/ \
                        (sixpm['MntShoes'].max()-sixpm['MntShoes'].min())
sixpm['Amnt_spent_norm'] = (sixpm['Amnt_Spent']-sixpm['Amnt_Spent'].min())/ \
                        (sixpm['Amnt_Spent'].max()-sixpm['Amnt_Spent'].min())
sixpm['Recency_norm'] = sixpm['Recency'] = (sixpm['Recency']-sixpm['Recency'].min())/ \
                        (sixpm['Recency'].max()-sixpm['Recency'].min())


"""PCA"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

finalData = sixpm[['MntAcessories_norm', 'MntClothing_norm', 'MntBags_norm', 'MntAthletic_norm','MntPremiumProds_norm','MntShoes_norm']]

pca = PCA(n_components=6)
pca.fit(finalData)
        
print('\nComponents: \n', pca.components_)
print('Explained variance: \n', pca.explained_variance_)
print('Explained variance in percentage: \n', pca.explained_variance_ratio_)
print('Covariance matrix: \n', pca.get_covariance())

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.xticks([0, 1, 2, 3, 4, 5])
plt.plot(var1)

pca = PCA(n_components=5)
pca.fit(finalData)
projected=pca.fit_transform(finalData)


"""SOM PU"""
import numpy as np
import somoclu
import pandas as pd

data = sixpm[['MntAcessories_norm', 'MntClothing_norm', 'MntBags_norm',\
              'MntAthletic_norm','MntPremiumProds_norm','MntShoes_norm']]
df=np.float32(data.values)

n_rows, n_columns = 20, 20

som = somoclu.Somoclu(n_columns, n_rows, maptype="toroid")
som = somoclu.Somoclu(n_columns, n_rows, gridtype="rectangular")
som = somoclu.Somoclu(n_columns, n_rows, initialization = 'pca')
som.train(df, epochs = 50)

som.train(df, radius0 = 0.1, radiusN = 0.01)

som.view_component_planes()
som.cluster()
som.view_umatrix(bestmatches=True)



"""SOM CV"""
sixpm['Education'] = np.where((sixpm['Education']=='Basic'), 1, sixpm['Education'])
sixpm['Education'] = np.where((sixpm['Education']=='2n Cycle'), 2, sixpm['Education'])
sixpm['Education'] = np.where((sixpm['Education']=='Graduation'), 3, sixpm['Education'])
sixpm['Education'] = np.where((sixpm['Education']=='Master'), 4, sixpm['Education'])
sixpm['Education'] = np.where((sixpm['Education']=='PhD'), 5, sixpm['Education'])
 
sixpm['Type_of_Customer'] = np.where((sixpm['Amnt_Spent']>0) & (sixpm['TimeCustomer']==1), 1, 0)

sixpm['Marital_Status'] = np.where((sixpm['Marital_Status']=='Divorced') | (sixpm['Marital_Status']=='Widow'), 0, sixpm['Marital_Status'])
sixpm['Marital_Status'] = np.where((sixpm['Marital_Status']=='Together') | (sixpm['Marital_Status']=='Married'), 1, sixpm['Marital_Status'])
sixpm['Marital_Status'] = np.where((sixpm['Marital_Status']=='Single'), 2, sixpm['Marital_Status'])
 
sixpm['Generation'] = np.where((sixpm['Generation']=='BabyBoomer'), 0, sixpm['Generation'])
sixpm['Generation'] = np.where((sixpm['Generation']=='GenX'), 1, sixpm['Generation'])
sixpm['Generation'] = np.where((sixpm['Generation']=='Millenial'), 2, sixpm['Generation'])
sixpm['Generation'] = np.where((sixpm['Generation']=='GenZ'), 3, sixpm['Generation'])

import numpy as np
import somoclu
import pandas as pd

data2 = sixpm[['Kidhome_Flag', 'Marital_Status', 'Type_of_Customer',\
               'Income_norm' , 'Teenhome_Flag', 'Recency_norm', 'Generation',\
               'Education']]
df2=np.float32(data2.values)

n_rows, n_columns = 20, 20

som = somoclu.Somoclu(n_columns, n_rows, maptype="toroid")
som = somoclu.Somoclu(n_columns, n_rows, gridtype="rectangular")
som = somoclu.Somoclu(n_columns, n_rows, initialization = 'pca')
som.train(df2, epochs = 50)

som.train(df2, radius0 = 0.1, radiusN = 0.01)

som.view_component_planes()
som.cluster()
som.view_umatrix(bestmatches=True)

