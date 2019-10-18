import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

session = tf.compat.v1.Session()

data = pandas.read_excel('data_clean\dataset_v2.xlsx' , sheet_name = 'Sheet1')

dataset = []
target = []

dataset_count = len(data)

avg_asg_1 = data['avg_asg_1']
avg_asg_2 = data['avg_asg_2']
avg_asg_3 = data['avg_asg_3']
avg_asg_4 = data['avg_asg_4']

final_status = data['final_status']

for i in range(dataset_count):
    dataset.append([
        [avg_asg_1[i]],
        [avg_asg_2[i]],
        [avg_asg_3[i]],
        [avg_asg_4[i]],
    ])
    target.append(final_status[i])

dataset = np.array(dataset, dtype = float)
target = np.array(target, dtype = float)

x_train, x_validation, y_train, y_validation = train_test_split(dataset, target, test_size=0.2, random_state=4)

with session.as_default():
    model = tf.keras.Sequential([
        layers.Bidirectional(layers.LSTM(4, batch_input_shape=(None, None, 1), return_sequences=True)),
        layers.Dropout(0.2),
        layers.Dense(2 , activation = 'relu'),
        layers.LSTM(1, return_sequences=False)
    ])
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=500, validation_data=(x_validation, y_validation))
    results = model.predict(x_validation)

    normalized_results = []

    count = len(y_validation)

    tp, fp, tn, fn = 0, 0, 0, 0
    correct = 0
    for i in range(count):
        expected = y_validation[i]
        predicted = round(results[i][0], 0)
        normalized_results.append(predicted)
        
        if expected == predicted:
            correct = correct + 1
            if expected == 1:
                tp = tp + 1
            else:
                tn = tn + 1
        else:
            if expected == 1 and predicted == 0:
                fn = fn + 1
            else:
                fp = fp + 1

    # print('TP = {0}, FP = {1}, TN = {2}, FN = {3}'.format(tp, fp, tn, fn))

    plt.scatter(range(count), results, c='b')
    plt.scatter(range(count), y_validation, c='g')
    plt.scatter(range(count), normalized_results, c='k')
    plt.show()

    precision = fp == 0 and 1 or tp / (tp + fp)
    recall = fn == 0 and 1 or tp / (tp + fn)

    print('Accuracy: {0}%'.format(round(correct * 100 / count, 2)))
    print('=> {0} correct predictions out of {1}'.format(correct, count))
    print('Precision: {0}%'.format(round(precision * 100, 2)))
    print('Recall: {0}%'.format(round(recall * 100, 2)))

    # for layer in model.layers:
    #     print('Layer: {0}\nWeight: {1}\n====='.format(layer, layer.get_weights()))

from flask import Flask, request, jsonify , render_template
import flask_excel as excel

app = Flask(__name__)

@app.route('/', methods = ['POST','GET'])
def view_index():
    if request.method == 'POST' :
        try:
            with session.as_default():
                file = request.files['file']
                data_ = pandas.read_excel(file , usecols = [2,3,4,5] , sheet_name = 'Sheet1')
                data_student = pandas.read_excel(file , usecols = [0,1] , sheet_name = 'Sheet1')
                dataset_count = len(data_)
                dataset = []
                avg_asg_1 = data_['asg_1']
                avg_asg_2 = data_['asg_2']
                avg_asg_3 = data_['asg_3']
                avg_asg_4 = data_['asg_4']
                for i in range(dataset_count):
                    dataset.append([
                        [avg_asg_1[i]],
                        [avg_asg_2[i]],
                        [avg_asg_3[i]],
                        [avg_asg_4[i]],
                    ])
                dataset = np.array(dataset, dtype = float)
                Predictions = model.predict(dataset)

                return render_template('index.html' ,show_prediction = True ,len = len(Predictions) , 
                    Predictions = Predictions , 
                    nama = data_student['name'] , nims = data_student['nim'])
        except Exception as e:
            print('Exception! {0}'.format(e))
            return render_template('index.html', error=e)
    else :
        return render_template('index.html')
app.run(host='0.0.0.0', port=4000 , debug=True)