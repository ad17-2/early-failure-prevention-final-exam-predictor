import numpy as np
import sys
import scipy
import pandas
import sklearn
import matplotlib
import seaborn as sns
from pandas.plotting import scatter_matrix 
import matplotlib.pyplot as plt
from sklearn import preprocessing as pre
from sklearn import utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold


names = ['avg_asg_1','avg_asg_2','avg_asg_3','avg_asg_4','final_status']

dataset = pandas.read_csv('data_clean\dataset_v2.csv',names = names , usecols=[0,1,2,3,4])

data_validation_size = 0.20
dataset_array_splice = dataset.values

dataset_array_x = dataset_array_splice[:,0:4]
dataset_array_y = dataset_array_splice[:,4]

seed = 4

X_train, X_validation, Y_train, Y_validation = sklearn.model_selection.train_test_split(
    dataset_array_x, dataset_array_y, test_size = data_validation_size, random_state = seed)
scoring = 'accuracy'

from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score

accuracy_list = []
# k_list = []

# for j in range(1,11):
#     for i in range(1,31) :
#         knn = KNeighborsClassifier(n_neighbors=i)
#         knn.fit(X_train,Y_train)
#         prediction = knn.predict(X_validation)
#         accuracies = accuracy_score(Y_validation,prediction)
#         accuracy_list.append(accuracies)
#         k_list.append(i)
#     plt.plot(k_list,accuracy_list)
#     plt.xlabel('K Value')
#     plt.ylabel('Accuracy')
#     plt.savefig("plot_result\plot_result_%d.png" % (j))


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
prediction = knn.predict(X_validation)

print(accuracy_score(Y_validation,prediction))
print(confusion_matrix(Y_validation,prediction))
print(classification_report(Y_validation,prediction))

avg_all_scores = []

for i in range(len(X_validation)):
    avg = 0
    for j in range(len(X_validation[0])):
        avg += X_validation[i][j]
    avg_all_scores.append(avg/4)

colors = (0,0,0)
area = np.pi*3


graf = sns.scatterplot(x=avg_all_scores,y=prediction,hue = prediction, style = prediction)
graf.set(xlabel='avg_all_scores',ylabel='final_exam_status')

#plt.scatter(avg_all_scores, prediction , s=area, c=colors, alpha=0.5)
#plt.title('Predicton')
#plt.xlabel('Score')
#plt.ylabel('Final Exam Status')
plt.show()

normalized_results = []
count = len(Y_validation)
tp, fp, tn, fn = 0, 0, 0, 0
correct = 0

for i in range(count):
    expected = Y_validation[i]
    predicted = prediction[i]
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
f1_score = 2*((precision * recall) / (precision + recall))

plt.scatter(range(count), prediction, c='b')
plt.scatter(range(count), Y_validation, c='g')
plt.scatter(range(count), normalized_results, c='k')
plt.show()



print('Accuracy: {0}%'.format(round(correct * 100 / count, 2)))
print('=> {0} correct predictions out of {1}'.format(correct, count))
print('Precision: {0}%'.format(round(precision * 100, 2)))
print('Recall: {0}%'.format(round(recall * 100, 2)))
print('F1 Score: {0}%'.format(round(f1_score * 100, 2)))

models = []
models.append(('Cross Validation', KNeighborsClassifier()))
results = [] 
names = [] 
  
for name, model in models: 
    kfold = model_selection.KFold(n_splits = 10, random_state = seed) 
    cv_results = model_selection.cross_val_score( 
            model, X_train, Y_train, cv = kfold, scoring = scoring) 
    results.append(cv_results) 
    names.append(name) 
    msg = "% s: % f (% f)" % (name, cv_results.mean() * 100, cv_results.std() * 100) 
    print(msg) 

from sklearn.externals import joblib

filename = 'finalized_model.sav'
joblib.dump(knn,filename)
