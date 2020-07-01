#Random Forest Regression

#importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')

#Separating the dependant and independant variable
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

#Fitting  Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300 , random_state=0 )
regressor.fit(X,Y)

'''#predicting a new result
y_pred=regressor.predict([[6.5]])

#Visualizing the  Regression Results
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()'''

import pickle
pickle.dump(regressor , open('salary_predictor.pkl','wb'))