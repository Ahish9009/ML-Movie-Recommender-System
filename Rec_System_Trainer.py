#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[2]:


#to load the various databases
user_ratings=pd.read_csv('User_Ratings.csv')
movie_ratings=pd.read_csv('Movie_Ratings.csv')


# In[3]:


num_movies=len(movie_ratings['Title'].as_matrix())


# In[4]:


#pandas dummies cannot be used for genres as each movie has many genres associated with it
#getting all the various genres into a list
genres=[]

for i in movie_ratings['Genres'].as_matrix():
    
    current_genres=i.split(', ')
    for j in current_genres:
        if j not in genres:
            genres+=[j]

with open("Training Genres.txt", "w") as g_file:
    
    line=""
    for i in genres:
        line+=str(i)+" "
    
    g_file.write(line)

len(genres)
    


# In[5]:


inputs=np.array([0.0 for i in range(23)]) #initializes inputs matrix
weights=np.array([np.random.random()*10 for i in range(23)]) #23 features initialized as 0


# In[6]:


#creating dataframe with 23 values
input_df=pd.get_dummies(movie_ratings, prefix='is', columns=['Language'])


# In[7]:


#to add the is_'genre' columns
for i in genres:    
    input_df['is_'+i]=[0 for j in range(num_movies)]
    


# In[8]:


#goes through all the movies
for i in range(num_movies):
    
    #gets the genres of the current movie
    current_genres=input_df['Genres'][i]
    
    #traverses through the current movie's genres
    for j in current_genres.split(', '):
        
        #sets the values to 1 to the is_'genre' column
        input_df['is_'+j][i]=1
        


# In[9]:


#input_df.dtypes


# In[10]:


#to delete the unnecessary columns
input_df.drop(['Const', 'Date Rated', 'URL', 'Title Type', 'Num Votes', 'Release Date', 'Directors'], axis=1, inplace=True)


# In[11]:


#input_df


# In[12]:


len(input_df.columns)


# In[13]:


#to move the Genres before title
headers=input_df.columns.values

headers=np.delete(headers, 5)

headers=np.insert(headers, 1, 'Genres')

input_df=input_df[headers]


# In[14]:


input_df.to_csv('Training Inputs Database.csv', index=False)


# In[15]:


#to get the Z-scores for each input column
#for i in range(23):
    
    #input_df[headers[i+3]]=(input_df[headers[i+3]]-input_df[headers[i+3]].mean())/float(input_df[headers[i+3]].std())

#to get the min-max scores for each input column
for i in range(23):
    
    input_df[headers[i+3]]=(input_df[headers[i+3]])/float(input_df[headers[i+3]].max())
    
#input_df
    


# In[16]:


training_loss_df=pd.DataFrame([], columns=['Training Loss'])

#training_loss_df


# In[17]:


def sgn(x):
    if x>0:
        return 1
    elif x==0:
        return 0
    else:
        return -1


# In[18]:


#Learning rate
LR=0.1


# In[21]:


#number of times to iterate
num_iterations=500
training_loss=num_movies*100

for k in range(num_iterations):
    #main algorithm 

    #change in Weights
    dW=np.array([0.0 for i in range(23)])

    #one iteration:
    #loads each movie one by one
    for i in range(num_movies):

        #to load the required columns
        for j in range(0,23):

            #gets the correct input
            inputs[j]=input_df[input_df.columns[j+3]][i]

        #value of rating obtained by multiplying inputs with weights
        calc_value = inputs.dot(weights)

        #to calculate change in each weight
        for j in range(0,23):
            
            #change in weight, dWi, for this particular movie
            if inputs[j]!=0: #0 input has no effect with weight
                dWi=float(input_df['Your Rating'][i] - calc_value)/(inputs[j])
            
                if abs(dWi)>LR:
                    dWi/=10
            
                #print (dWi)

                #adds so that the sum can be divided later by number 
                dW[j]+=dWi

    #finding the average change
    
    #print (dW)
    
    for i in range(len(dW)):
        dW[i]/=num_movies
    
    #changing the weights
    for i in range(len(weights)):

        weights[i]+=dW[i]

    #to find new training loss
    old_training_loss=training_loss
    training_loss=0
    for i in range(num_movies):

        #to load the required columns
        for j in range(0,23):

            #gets the correct input
            inputs[j]=input_df[input_df.columns[j+3]][i]

        training_loss+=abs(input_df['Your Rating'][i] - inputs.dot(weights))
        
    new_data=pd.DataFrame([training_loss], columns=['Training Loss'])
    training_loss_df=training_loss_df.append(new_data)
    
    if training_loss>old_training_loss:
        break
        
training_loss_df.to_csv(r'Training Loss.csv')
        
print (training_loss)


# In[20]:


weights_df=pd.DataFrame(weights, columns=['Weights'])
weights_df.to_csv(r''+str(training_loss)+'.csv')
