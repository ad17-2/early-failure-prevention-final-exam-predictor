import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from time import time

names = ['avg_asg_1','avg_asg_2','avg_asg_3','avg_asg_4','final_status']

dataset = pandas.read_csv('dataset/dataset_v2.csv' , names = names)

# sns.heatmap(dataset.corr(), annot=True)
# plt.show()

dataset_values = dataset.values

x = dataset_values[:,0:4]
y = dataset_values[:,4]

seed = np.random

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2 , random_state = seed)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1 , patience=50)
mc = ModelCheckpoint('model\optimized_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

model = Sequential()
model.add(Dense(8, activation='relu', kernel_initializer='random_normal', input_dim=4))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
model.compile(optimizer ='adam',loss='mse', metrics =['accuracy'])
history = model.fit(x_train,y_train, batch_size=10, epochs=500 ,validation_data=(x_val, y_val) , callbacks=[es , mc , tensorboard])
results = model.predict(x_val)
results = (results > 0.5)


_,train_accuracy = model.evaluate(x_train, y_train , verbose = 0)
_,test_accuracy = model.evaluate(x_val,y_val, verbose = 0)

print('Train: %.3f, Test: %.3f' % (train_accuracy, test_accuracy))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

normalized_results = []

count = len(y_val)

tp, fp, tn, fn = 0, 0, 0, 0
correct = 0
for i in range(count):
    expected = y_val[i]
    predicted = results[i]
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

precision = fp == 0 and 1 or tp / (tp + fp)
recall = fn == 0 and 1 or tp / (tp + fn)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=np.random)

for train, test in kfold.split(x, y):
    model = load_model('model\optimized_model.h5')
    history = model.fit(x_train,y_train, batch_size=10, epochs=150 ,validation_data=(x_test, y_test))
    results = model.predict(x_test)
    results = (results > 0.5)
    scores = model.evaluate(x[test], y[test], verbose=0)
    cvscores.append(scores[1] * 100)

	
print("Test Size: %d , Validation Size : %d , Test Size : %d" % (len(x_train) , len(x_val) , len(x_test)))
print('Accuracy: {0}%'.format(round(correct * 100 / count, 2)))
print('=> {0} correct predictions out of {1}'.format(correct, count))
print('Precision: {0}%'.format(round(precision * 100, 2)))
print('Recall: {0}%'.format(round(recall * 100, 2)))
print("Cross Validation Mean : %.2f%% (Cross Validation STD +/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# plt.scatter(range(count), results, c='b')
# plt.scatter(range(count), y_val, c='g')
# plt.scatter(range(count), normalized_results, c='k')
# plt.show()