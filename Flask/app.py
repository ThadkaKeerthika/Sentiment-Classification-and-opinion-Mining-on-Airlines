# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:29:19 2019

@author: lalit
"""
    
from flask import render_template, Flask, request,url_for
from keras.models import load_model
import pickle 
import tensorflow as tf
graph = tf.get_default_graph()
with open(r'CountVectorizer','rb') as file:
    cv=pickle.load(file)
cla = load_model('twitter1.h5')
cla.compile(optimizer='adam',loss='binary_crossentropy')
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/tpredict')
@app.route('/', methods = ['GET','POST'])
def page2():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        topic = request.form['tweet']
        with graph.as_default():
            y_pred = cla.predict(topic)
            print("pred is "+str(y_pred))
        if(y_pred > 0.5):
            topic = "Positive Tweet - Hope you will fly with us again"
        elif(y_pred < 0.5):
            topic = "Negative Tweet - Sorry about that"
        else:
            topic = "Neutral Tweet"
        return render_template('index.html',ypred = topic)
        



if __name__ == '__main__':
    app.run(host = 'localhost', debug = True , threaded = False)
    
