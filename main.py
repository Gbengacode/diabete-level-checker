import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# import data set
diabete_dataset = pd.read_csv('diabetes.csv')

# number of rows


# getting the statistical measure of the data

diabete_dataset.describe()

X = diabete_dataset.drop(columns="Outcome", axis=1)
Y = diabete_dataset["Outcome"]
print(Y)

# Standardize data

scalar = StandardScaler()
standard_data = scalar.fit_transform(X)
X = standard_data
# train and test factor
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)
classifier = svm.SVC(kernel='linear')

# training the support vector machine classification
classifier.fit(x_train, y_train)

# accuracy score
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print(training_data_accuracy)

x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print(test_data_accuracy)

# predictive system
input_data = (4, 110, 92, 0, 0, 37.6,  0.191, 30)
array_data = np.asarray(input_data)
# reshape data
input_data_reshaped = array_data.reshape(1, -1)
# print(input_data_reshaped)
std_data = scalar.transform(input_data_reshaped)

#prediction
prediction = classifier.predict(std_data)
print(prediction)
# print(std_data)
