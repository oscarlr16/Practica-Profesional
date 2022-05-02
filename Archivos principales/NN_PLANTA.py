from keras.models import Sequential
from keras.layers import Dense
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


seed = 7
np.random.seed(seed)

# Cargar el dataset de planta.csv
dataset = np.loadtxt("planta.csv",delimiter=",")

tiempo = [i for i in range(0,100)]
x = np.array(tiempo)
y = dataset[1,:]

x = x.reshape((len(x), 1))
y = y.reshape((len(y), 1))

scale_x = MinMaxScaler()
x = scale_x.fit_transform(x)
scale_y = MinMaxScaler()
y = scale_y.fit_transform(y)

model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(75, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid',kernel_initializer='uniform'))

model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=50, verbose=1)

y_p = model.predict(x)

# Se reinvierte la escala
x_plot = scale_x.inverse_transform(x)
y_plot = scale_y.inverse_transform(y)
y_p_plot = scale_y.inverse_transform(y_p)

# plot x vs yhat

plt.plot(x_plot,y_p_plot, label='Predicción')


plt.plot(x_plot,y_plot, label='Original')
plt.title('Entrada (x) vs salida (y)')
plt.xlabel('Entrada Variable (x)')
plt.ylabel('salida Variable (y)')

plt.legend()
plt.show()