#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Recommender
import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt


# In[2]:


#gets users training data to normalize new data
training_df=pd.read_csv('Training Inputs Database.csv')
movies_df=pd.read_csv('Movie_Ratings.csv')
weights = pd.read_csv('Weights.csv')

test_df=pd.read_csv('Test Data.csv') #for getting the info on the movies to be rated (raw test data)
test_data_df=pd.DataFrame(columns=training_df.columns) #dataframe which will contain the normalized and prepared test data


# In[3]:


#TO BE INPUTED BY USER
training_languages=['English', 'Hindi']


# In[4]:


test_df.fillna('', inplace=True)


# In[5]:


#to remove unnecessary NaN errors while finding genres
test_df.fillna('', inplace=True)


# In[6]:


training_genres=[] #genres that the system is trained with

with open ("Training Genres.txt", "r") as fread:
    training_genres=fread.readline().split() #creates a list of the various genres used in training


# In[7]:


#pandas dummies cannot be used for genres as each movie has many genres associated with it
#getting all the various genres of the training data into a list
genres=[]
languages=[]

new_columns_genres=[[] for i in range(len(training_genres))]
new_columns_languages=[[] for i in range(len(training_languages))]

for i in test_df['Genres'].as_matrix():
    
    current_genres=i.split(', ')
    
    #adds the value of the genre dummies to the dataframe
    for i in range(len(training_genres)):
        if training_genres[i] in current_genres:
            new_columns_genres[i]+=[1]
        else:
            new_columns_genres[i]+=[0]

for i in range(len(training_genres)):
    test_data_df["is_"+training_genres[i]]=new_columns_genres[i]
    
for i in test_df['Language'].as_matrix():
    
    current_lang=i.split(', ')
    
    for i in range(len(training_languages)):
        if training_languages[i] in current_lang:
            new_columns_languages[i]+=[1]
        else:
            new_columns_languages[i]+=[0]

for i in range(len(training_languages)):
    test_data_df["is_"+training_languages[i]]=new_columns_languages[i]


# In[8]:


test_data_df.fillna(0, inplace=True)


# In[9]:


#to copy the basic values into the test_data_df dataframe

test_data_df['Title']=test_df['Title']
test_data_df['IMDb Rating']=test_df['IMDb Rating']
test_data_df['Runtime (mins)'] = test_df['Runtime (mins)']
test_data_df['Year'] = test_df['Year']


# In[10]:


test_data_df['IMDb Rating'] = test_data_df['IMDb Rating'].apply(pd.to_numeric, errors='coerce')
test_data_df['Runtime (mins)'] = test_data_df['Runtime (mins)'].apply(pd.to_numeric, errors='coerce')
test_data_df['Year'] = test_data_df['Year'].apply(pd.to_numeric, errors='coerce')


# In[11]:


#to normalize the data
test_data_df['IMDb Rating'] = test_data_df['IMDb Rating']/float(training_df['IMDb Rating'].max())
test_data_df['Runtime (mins)'] = test_data_df['Runtime (mins)']/float(training_df['Runtime (mins)'].max())
test_data_df['Year'] = test_data_df['Year']/float(training_df['Year'].max())

test_data_df['IMDb Rating'] = test_data_df['IMDb Rating'].apply(pd.to_numeric, errors='coerce')
test_data_df['Runtime (mins)'] = test_data_df['Runtime (mins)'].apply(pd.to_numeric, errors='coerce')
test_data_df['Year'] = test_data_df['Year'].apply(pd.to_numeric, errors='coerce')


# In[19]:


#to calculate final ratings of the user
final_weights=weights['Weights'].as_matrix()

columns=test_data_df.columns

your_rating=[]

title=test_data_df['Title'].as_matrix()

for j in range(len(title)):
    if title[j]!='':
        rating=0.0
        for i in range(3,len(columns)):
            rating+=(test_data_df[columns[i]].as_matrix()[j]*final_weights[i-3])
        
        if rating>10:
            rating=10.0
            
        your_rating+=[[title[j],rating]]
    
    else:
        break
    


# In[22]:


your_rating.sort(key=lambda x : x[1], reverse=True)

for i in range(len(your_rating)):
    print(your_rating[i][0], '-', your_rating[i][1])

