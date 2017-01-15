import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler

#read data
dataframe = pd.read_csv('challenge_dataset.txt')
x_values = dataframe[['x']]
y_values = dataframe[['y']]

#mean normalize
x_standard_scaler = StandardScaler()
y_standard_scaler = StandardScaler()
x_norm = x_standard_scaler.fit_transform(x_values)
y_norm = y_standard_scaler.fit_transform(y_values)


#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_norm, y_norm)
#print error

print np.subtract(y_values,y_standard_scaler.inverse_transform(body_reg.predict(x_norm)));

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, y_standard_scaler.inverse_transform(body_reg.predict(x_norm)))
plt.show()