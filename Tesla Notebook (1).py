#!/usr/bin/env python
# coding: utf-8

# # Tesla Deaths Exploratory Data Analysis

# In[1]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# # Contando a Introdução do Paper  
# Baixei, primeiramente, a biblioteca que contabilizava o número de acidentes envolvendo óbitos com condutores da Tesla desde o ano de 2014.

# In[6]:


df = pd.read_csv('tesla.csv')


# In[7]:


sns.set_theme(style='whitegrid',palette='mako' )


# # Métricas Selecionadas
# Podemos observar, abaixo, o número de casos contabilizados no total, além do ano e data de ocorrência. Temos, na coluna de 'Country' o país no qual ocorreu o acidente, seguido pelo Estado do óbito. Nas colunas adiante, de 'Description', há um breve resumo sobre o ocorrido. Em 'Deaths', podemos observar o número de óbitos ocorridos. Posteriormente, em 'Tesla driver', há a especificação se havia condução autônoma do veículo ou se um passageiro guiava o condutor. Em colunas como 'Occupant', descrevemos o número de passageiros. Nas colunas seguintes, temos 'Model', especificando o veículo em si. Temos, posteriormente, 'Sources', que especifica a fonte que noticiou o acidente. 

# In[8]:


df.head()


# In[ ]:





# In[9]:


df.columns


# In[10]:


df.columns = df.columns.str.strip() 


# In[11]:


df_date = df.groupby(by='Year').sum().reset_index()


# # Evolução de entregas e Mortes
# A seguir, no gráfico o qual estipula, no Eixo X o ano de ocorrência de mortes, e no Eixo Y, o número de mortes em cada ano, podemos observar que há uma evolução crescente de casos nos últimos cinco anos, entre 2018 e 2022, diante de uma maior popularização da Tesla. Dados das Relações com Investidores da Tesla mostram que, em 2018, 245.491 veículos foram entregues, montante 138% acima do registrado no acumulado de 2017. Podemos correlacionar o número de acidentes por uma maior evolução e popularização da montadora elétrica, baseada em Palo Alto, no Estado da Califórnia. Eis, então, os números de entrega da Tesla desde 2014.
# 
# 2014 - 31.655
# 2015 - 50.517
# 2016 - 76.243
# 2017 - 103.091
# 2018 - 245.491
# 2019 - 367.656
# 2020 - 499.535 
# 2021 - 936.000
# 2022 - 1.369.611
# 

# ## Years

# In[12]:


sns.lineplot(data=df, x='Year', y='Deaths', estimator='sum').set(title='Tesla Deaths over the Years', );


# In[13]:


df['Date']=pd.to_datetime(df['Date'], format='%m/%d/%Y')


# In[14]:


df['Day']=df['Date'].dt.strftime('%d')
df['Week']= df['Date'].dt.strftime('%A')
df['Month']=df['Date'].dt.strftime('%m')


# In[15]:


df['Month'] = df['Date'].dt.strftime('%m')


# In[ ]:


df.head()


# # Número de mortes em dias da semana 
# Podemos observar, pelo levantamento, que sábado foi o dia da semana que mais contabilizou acidentes e óbitos envolvendo condutores do Tesla, seguidos pela Segunda-feira e, em sequência, Sexta-feira. Há também uma maior especificação sobre a data, no mês, em que ocorreram os acidentes. Posteriormente, na linha 21, podemos observar a contabilização dos meses que houve registro de mortes envolvendo condutores de Tesla. Pelo gráfico, é possível apontar que o mês de maio foi o que mais contabilizou acidentes e mortes envolvendo a montadora elétrica. 
# 
# 
# 
# 

# In[17]:


sns.countplot(data=df, x='Week', palette='mako');


# In[18]:


df.sort_values('Day', inplace=True)


# In[19]:


#days
plt.figure(figsize=(14,7))
sns.countplot(data=df, x='Day', palette='mako');


# In[20]:


df.sort_values(by='Month', inplace=True)


# In[21]:


sns.countplot(data=df, x='Month', palette='mako');


# ## Country

# # EUA assumem liderança
# Em termos de mortes por país, os Estados Unidos são o país com o maior número de óbitos, naturalmente, pelo fato da montadora elétrica estar baseada no Estado da Califórnia. Naturalmente, por possuir plantas fabris na Alemanha e na China, a montadora elétrica de Musk também conta com razoável popularidade nos dois países, nos continente europeu e asiático, respectivamente, o que pode explicar as mortes. 

# In[22]:


plt.figure(figsize=(15,8))
sns.countplot(data=df, x='Country').set(title='Number of Deaths Per Country');
plt.xticks(rotation=45)

plt.show()


# In[23]:


df['Country'].value_counts()


# # Califórnia lidera mortes
# O Estado da Califórnia, naturalmente por concentrar a base principal da Tesla, em Palo Alto. 

# In[24]:


usa = df[df['Country'] == ' USA ']
germany = df[df['Country'] == ' Germany ']


# In[25]:


usa_state = usa.groupby(by='State').sum()


# In[26]:


usa_state.reset_index(inplace=True)


# In[27]:


plt.figure(figsize=(15,8))
sns.barplot(data=usa_state, x='State', y='Deaths').set(title='Deaths per State in the USA')

plt.show()


# In[28]:


# of these deaths how many are tesla drivers,     tesla occupants


# # Resultado de Mortes
# Falecimento de condutores ou passageiros, em decorrência dos acidentes envolvendo Tesla. 
# 

# In[29]:


df['Tesla driver'].value_counts()


# In[30]:


f, axes = plt.subplots(1, 2, figsize=(10,3))
sns.countplot(data=df, x='Tesla driver' , ax=axes[0]);
sns.countplot(data=df, x="Tesla occupant", ax=axes[1]);


# In[31]:


# df.replace(' - ', 0, inplace=True)


# In[32]:


#deaths groupedby country
grouped_country = df.groupby(by=['Country']).sum().reset_index()


# In[ ]:





# In[ ]:





# # Modelo 
# Há uma especificação, abaixo, dos veículos envolvidos em acidentes com óbitos da Tesla e, por consequência, sua distribuição por país. Podemos observar uma maior ocorrência envolvendo o Model S, seguido pelo Model 3. 

# In[33]:


# model of the car


# In[34]:


model = df.groupby(by='Model').sum()


# In[35]:


model.reset_index(inplace=True)
model


# In[36]:


sns.barplot(data=model, x='Model', y='Deaths')


# In[37]:


models_country =df.groupby(by=['Country', 'Model']).sum().reset_index()


# In[38]:


models_country.head(10)


# In[39]:


models_usa = models_country[models_country['Country']==' USA ']


# In[40]:


models_usa


# In[41]:


models_china = models_country[models_country['Country']==' China ']


# # Modelos de mortes por EUA, China
# Podemos observar que há uma maior ocorrência de mortes envolvendo o Model S da Tesla, nos EUA, que é um dos princupais modelos da montadora, lançado em julho de 2012. Na China, há uma maior equiparidade entre os modelos X e Y. 

# In[42]:


f, axes = plt.subplots(1, 2, figsize=(13,3))

sns.barplot(data=models_usa, x='Model', y='Deaths', ax=axes[0]).set(title='Deaths per Model in USA');
sns.barplot(data=models_china, x='Model', y='Deaths' ,ax=axes[1]).set(title='Deaths per Model in China');


# In[ ]:




