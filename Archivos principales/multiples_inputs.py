# use mlp for prediction on multi-output regression
from numpy import asarray
from numpy import mean
from numpy import std
import numpy as np 
from sklearn.datasets import make_regression
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import RepeatedKFold


dataset = np.loadtxt("planta.csv",delimiter=",")
'''
Calculamos las dimenciones del data set para generar los vetores U (steps) y el vector t
'''
###########
m = np.shape(dataset)[0]
n = np.shape(dataset)[1]

u = np.array([i*0.1 for i in range(m)])
t = np.linspace(0,10,n)
###########
X = np.meshgrid(u,t)[1]
Y = dataset.T

def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(100, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(75, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
    model.compile(loss='mae', optimizer='adam')
    return model

#evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
		# define model
		model = get_model(n_inputs, n_outputs)
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=1000)
		# evaluate model on test set
		mae = model.evaluate(X_test, y_test, verbose=0)
		# store result
		print('>%.3f' % mae)
		results.append(mae)
	return results
'''
# evaluate model
results = evaluate_model(X, Y)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(results), std(results)))
'''


# load dataset

n_inputs, n_outputs = X.shape[1], Y.shape[1]
# get model
model = get_model(n_inputs, n_outputs)
# fit the model on all data
model.fit(X, Y, verbose=0, epochs=1000)
# make a prediction for new data
row = [5 for i in range(m)]
newX = asarray([row])
yhat = model.predict(newX)
print('Predicted: %s' % yhat[0])
