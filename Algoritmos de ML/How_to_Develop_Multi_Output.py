# linear regression for multioutput regression
from sklearn.linear_model import LinearRegression
import numpy as np 


#############################################################################################
#############################################################################################

dataset = np.loadtxt("planta_T.csv",delimiter=",")
'''
Calculamos las dimenciones del data set para generar los vetores U (steps) y el vector t
'''
###########
m = np.shape(dataset)[0]
n = np.shape(dataset)[1]

u = np.array([i*0.1 for i in range(m)])
t = np.linspace(0,10,n)
###########
X = np.meshgrid(u,t)[0].T
Y = dataset

# define model
model = LinearRegression()
# fit model
model.fit(X, Y)
# make a prediction
row = [11 for i in range(n)]
yhat = model.predict([row]).T
# summarize prediction

###############
###############

# define model
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
# fit model
model.fit(X, Y)
# make a prediction
row = [11 for i in range(n)]
yhat_2 = model.predict([row]).T
# summarize prediction

###############
###############
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
# fit model
model.fit(X, Y)
# make a prediction
row = [11 for i in range(n)]
yhat_3 = model.predict([row]).T
# summarize prediction


###############
###############

# example of making a prediction with the direct multioutput regression model
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
# define dataset

# define base model
model = LinearSVR()
# define the direct multioutput wrapper model
wrapper = MultiOutputRegressor(model)
# fit the model on the whole dataset
wrapper.fit(X, Y)
# make a single prediction
row = [11 for i in range(n)]
yhat_4 = wrapper.predict([row]).T
# summarize the prediction


