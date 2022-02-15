import nltk

import string
from tqdm import tqdm
import numpy as np
#import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from flask import Flask,render_template,url_for,request
from keras.models import load_model

# import tensorflow as tf
# from keras.utils import np_utils
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.layers import Dense, Dropout, Activation
from nltk.corpus import stopwords
#

filename='model.pickle'
#clf=open(filename,'rb')
clf=load_model('swahili.h5')
cv=pickle.load(open('vectorizer.pickle','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method =='POST':
        message=request.form['message']
        data=[message]
        vect=cv.transform(data).toarray()
        my_prediction=np.argmax(clf.predict(vect),axis=1)
    return render_template('result.html',prediction=my_prediction)
    
if __name__ == '__main__':
	app.run(debug=True)