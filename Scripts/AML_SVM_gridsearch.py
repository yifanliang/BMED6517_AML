### AML SVM with grid search

import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn import svm

from sklearn import metrics
from statistics import mean


## setup data
aml_data = pandas.read_csv('D:/BMED 6517/test_ml_dataset.csv', header=0)
##### add size features FS/SS
aml_data['1_Size'] = aml_data['1_FS Lin']/aml_data['1_SS Log']
aml_data['2_Size'] = aml_data['2_FS Lin']/aml_data['2_SS Log']
aml_data['3_Size'] = aml_data['3_FS Lin']/aml_data['3_SS Log']
aml_data['4_Size'] = aml_data['4_FS Lin']/aml_data['4_SS Log']
aml_data['5_Size'] = aml_data['5_FS Lin']/aml_data['5_SS Log']
aml_data['6_Size'] = aml_data['6_FS Lin']/aml_data['6_SS Log']
aml_data['7_Size'] = aml_data['7_FS Lin']/aml_data['7_SS Log']
aml_data['8_Size'] = aml_data['8_FS Lin']/aml_data['8_SS Log']

aml_data = aml_data.drop(['1_FS Lin', '2_FS Lin', '3_FS Lin', '4_FS Lin', '5_FS Lin', '6_FS Lin', '7_FS Lin', '8_FS Lin'], axis = 1)
aml_data = aml_data.drop(['1_SS Log', '2_SS Log', '3_SS Log', '4_SS Log', '5_SS Log', '6_SS Log', '7_SS Log', '8_SS Log'], axis = 1)
#aml_data = aml_data.drop(['8_FL1 Log','8_FL2 Log', '8_FL3 Log', '8_FL4 Log', '8_FL5 Log'], axis = 1)
##### new features scaling
from sklearn.preprocessing import minmax_scale
aml_data[['1_Size', '2_Size', '3_Size', '4_Size', '5_Size', '6_Size', '7_Size', '8_Size']] = minmax_scale(X = aml_data[['1_Size', '2_Size', '3_Size', '4_Size', '5_Size', '6_Size', '7_Size', '8_Size']], feature_range = (0, 1))


## formatting
known_set = aml_data[0:179].copy()
prediction_set = aml_data[179:].copy()
prediction_set = prediction_set.reset_index()
prediction_set = prediction_set.drop(['index'], axis = 1)



### prediction_set (X, Y)
prediction_x = prediction_set.drop(['Label', 'Unnamed: 0'], axis = 1).copy()


### training set (X, Y)
known_y = known_set['Label'].copy()
for i in range(0, len(known_y)):
    if known_y[i] == 'normal':
        known_y.at[i] = 0
    if known_y[i] == 'aml':
        known_y.at[i] = 1
known_y = known_y.astype('int')
known_y = known_y.to_numpy()

known_x = known_set.drop(['Label', 'Unnamed: 0'], axis = 1).copy()
known_x = known_x.to_numpy()

### run 20 times, take average accuracy, precision, and recall

accuracylist = []
precisionlist = []
recalllist = []
f1list = []
prediction_probs = numpy.zeros((180,2))
predictions = numpy.zeros((180,))
count = 0
#### split 80/20

classifier = svm.SVC(kernel = 'poly')
classifier.probability = True
    
    # defining parameter range
from sklearn.model_selection import GridSearchCV
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma': gammas}
grid = GridSearchCV(svm.SVC(kernel = 'poly', probability = True), param_grid, refit= True, verbose= 3)
while count < 100:
    from imblearn.over_sampling import SMOTE
    sm = SMOTE()
    known_x, known_y = sm.fit_resample(known_x, known_y)
    x_train, x_test, y_train, y_test = train_test_split(known_x, known_y, test_size = 0.2)



# Build SVM Model

### creating classifier


  
    # fitting the model for grid search
    grid.fit(x_train, y_train)

    # print best parameter after tuning
    #print(grid.best_params_)
  
# print how our model looks after hyper-parameter tuning
#print(grid.best_estimator_)

    #grid_predictions= grid.predict(x_test)
  
# print classification report
    #print(classification_report(y_test, grid_predictions))

### training
    #classifier.fit(x_train, y_train)

### prediction
    y_pred = grid.predict(x_test)


#Evaluating SVM Model

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    '''if f1 > maxf1:
        maxf1 = f1
        maxSVM = classifier'''
    
    accuracylist.append(accuracy)
    precisionlist.append(precision)
    recalllist.append(recall)
    f1list.append(f1)
    count += 1
    ### prediction at each iteration
    prediction_y = grid.predict(prediction_x)
    predictions = numpy.add(predictions, prediction_y)
    predict_probs = grid.predict_proba(prediction_x)
    prediction_probs = numpy.add(prediction_probs, predict_probs)
testmetrics = [('Accuracy', mean(accuracylist)), ('Precision', mean(precisionlist)), ('Recall', mean(recalllist)), ('F1 Score', mean(f1list))]
print(testmetrics)
#print(predict_probs[:,1])
## Prediction
final_predictions = predictions/100
prediction_probs = prediction_probs/100
for i in range(len(final_predictions)):
    if final_predictions[i] >= 0.5:
        final_predictions[i] = 1
    if final_predictions[i] < 0.5:
        final_predictions[i] = 0


final_prediction = pandas.DataFrame(final_predictions, columns = ['Predictions'])
final_prediction['Predictions'] = final_prediction['Predictions'].astype(str)

prediction_probs = pandas.DataFrame(prediction_probs[:,1], columns = ['Prediction Probs'])
#prediction_probs['Prediction Probs'] = prediction_probs['Prediction Probs'].astype(str)

for i in range(180):
    if final_prediction.at[i, 'Predictions'] == '1.0':
        final_prediction.at[i, 'Predictions'] = 'aml'
    if final_prediction.at[i, 'Predictions'] == '0.0':
        final_prediction.at[i, 'Predictions'] = 'normal'

prediction_set = prediction_set.drop('Label', axis = 1)
prediction_set = prediction_set.join(final_prediction)

prediction_set = prediction_set.join(prediction_probs)


  
