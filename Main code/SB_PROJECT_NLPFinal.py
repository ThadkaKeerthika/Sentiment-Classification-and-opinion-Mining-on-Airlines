# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:11:10 2020

@author: HP
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer


dataset=pd.read_csv("Dataset/Tweets.csv")
dataset.head(5)



dataset.isnull().any()
dataset.info()
dataset['negativereason'].fillna(dataset['negativereason'].mode()[0],inplace=True)
dataset['negativereason_confidence'].fillna(dataset['negativereason_confidence'].mean(),inplace=True)
dataset['airline_sentiment_gold'].fillna(dataset['airline_sentiment_gold'].mode()[0],inplace=True)
dataset['negativereason_gold'].fillna(dataset['negativereason_gold'].mode()[0],inplace=True)
dataset['tweet_coord'].fillna(dataset['tweet_coord'].mode()[0],inplace=True)
dataset['tweet_location'].fillna(dataset['tweet_location'].mode()[0],inplace=True)
dataset['user_timezone'].fillna(dataset['user_timezone'].mode()[0],inplace=True)
dataset.isnull().any()



sns.pairplot(dataset,palette='rainbow')
sns.heatmap(dataset.corr(),annot=True)




dataset.drop(['tweet_id'], inplace=True, axis=1)
dataset.drop(['name'], inplace=True, axis=1)
dataset.drop(['tweet_coord','tweet_created', 'tweet_location', 'user_timezone'], inplace=True, axis=1)
dataset.drop(['airline_sentiment_confidence','negativereason', 'negativereason_confidence', 'negativereason_gold'], inplace=True, axis=1)
dataset.drop(['airline_sentiment_gold', 'airline'], inplace=True,axis=1)
    #LabelEncoding
le = LabelEncoder()
dataset['airline_sentiment'] = le.fit_transform(dataset['airline_sentiment'])
 #OneHotEncoding
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 0:1].values
#onehot encoder
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
a = one.fit_transform(y[:,0:1]).toarray()
y = np.delete(x, 0, axis=1)
y = np.concatenate((a, y), axis=1)
    #train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
data=[] 
for i in range(0,14640):
    review=dataset['text'][i]
    review=re.sub('[^a-zA-Z]',' ',review)
    review=review.lower()
    review=review.split() 
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    review=' '.join(review) 
    data.append(review) 
    




from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x1 = cv.fit_transform(data).toarray()
y1 = dataset.iloc[:,0:1].values



from sklearn.model_selection import train_test_split
x_train1,x_test1,y_train1,y_test1=train_test_split(x1,y1,test_size=0.2,random_state=0)




from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(units=2000,init='uniform',activation='relu'))
model.add(Dense(units=4000,init='uniform',activation='relu'))
model.add(Dense(units=1,init='uniform',activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train1,y_train1,epochs=10,batch_size=32)

model.save("twitter1.h5")

#Accuracy 82.3%

#prediction
#Team 40
text='Hey it was terrible'
text=re.sub('[^a-zA-z]',' ',text)
text=text.lower()
text=text.split()
text=[ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
text=' '.join(text)
pred=model.predict(cv.transform([text]))
pred = pred>0.5

print(pred)
