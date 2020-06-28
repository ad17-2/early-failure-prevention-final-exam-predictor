import pickle
import numpy as np
import pandas
from sklearn import preprocessing as pre
from sklearn import utils
from sklearn.neighbors import KNeighborsClassifier

filename = 'finalized_model.sav'

loadmodel = pickle.load(open(filename,'rb'))

from flask import Flask, request, jsonify , render_template
import flask_excel as excel

app = Flask(__name__)

@app.route('/', methods = ['POST','GET'])
def view_index():
    if request.method == 'POST' :
        try :
            file = request.files['file']
            data_ = pandas.read_excel(file , usecols = [2,3,4,5] , sheet_name = 'Sheet1')
            data_student = pandas.read_excel(file , usecols = [0,1] , sheet_name = 'Sheet1')
            Predictions = loadmodel.predict(np.array(data_, dtype=float))
            return render_template('index.html' ,show_prediction = True ,len = len(Predictions) , Predictions = Predictions ,
                nama = data_student['name'] , nims = data_student['nim'])
        except Exception as e:
            print('Exception! {0}'.format(e))
            return render_template('index.html', error=e)
    else :
        return render_template('index.html')
app.run(host='0.0.0.0', port=4000 , debug=True)