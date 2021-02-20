## importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn import svm

## Loading data
df = pd.read_excel('data.xlsx') 

## Prepare data to fit
X = df.drop('NVF', axis=1)
y = df['NVF']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

## Fitting the model using best parameter from GridSearchCV 
regr = svm.SVR(C = 10000, epsilon = 0.1, gamma= 'scale', kernel= 'rbf')
regr.fit(X_train, y_train)
print("R^2 score of the model: ", regr.score(X_test, y_test))

## saving plot of actual output vs predicted output
regr_y_pred = regr.predict(X_test)
fig, ax = plt.subplots(figsize=(20,6))
ax.plot(np.array(y_test), color='orange')
ax.set(title = "Actual NVF vs Predicted NVF",
       ylabel = 'output')

ax.plot(regr_y_pred, color='blue')
ax.set()
fig.savefig('actual vs predicted.png')
plt.show();

## saving 5 png file showing correlation between each input to output(NVF)
fig, ax = plt.subplots(figsize=(5,4))
ax.scatter(df['Triaxiality'],df["NVF"], color='orange')
ax.set(title = "Triaxiality vs NVF",
       xlabel = 'Triaxiality',
       ylabel = 'NVF')
ax.set()
fig.savefig('Triaxiality vs NVF.png')
plt.show();

fig, ax = plt.subplots(figsize=(5,4))
ax.scatter(df['Lode'],df["NVF"], color='red')
ax.set(title = "Lode vs NVF",
       xlabel = 'Lode',
       ylabel = 'NVF')
ax.set()
fig.savefig('Lode vs NVF.png')
plt.show();

fig, ax = plt.subplots(figsize=(5,4))
ax.scatter(df['strain'],df["NVF"], color='green')
ax.set(title = "strain vs NVF",
       xlabel = 'strain',
       ylabel = 'NVF')
ax.set()
fig.savefig('strain vs NVF.png')
plt.show();

fig, ax = plt.subplots(figsize=(5,4))
ax.scatter(df['orientation'],df["NVF"], color='blue')
ax.set(title = "orientation vs NVF",
       xlabel = 'orientation',
       ylabel = 'NVF')
ax.set()
fig.savefig('orientation vs NVF.png')
plt.show();

fig, ax = plt.subplots(figsize=(5,4))
ax.scatter(df['Hconc'],df["NVF"], color='black')
ax.set(title = "Hconc vs NVF",
       xlabel = 'Hconc',
       ylabel = 'NVF')
ax.set()
fig.savefig('Hconc vs NVF.png')
plt.show();

## results / prediction on entire datasets and save them in a csv file
results = regr.predict(X)
# save results array as csv file
from numpy import asarray
from numpy import savetxt

# save to csv file
print("results is saving....")
savetxt('result.csv', results, delimiter=',')
print("results saved on file 'result.csv'")


##Saving the model
###As we have saved the results in a csv file we might not use the pickle. but it might help for the future use
import pickle
pickle.dump(regr, open("svm_regressor_model.pkl", "wb"))

##Loading the model
loaded_model = pickle.load(open("svm_regressor_model.pkl", "rb"))
print("Score of the loaded model: ", loaded_model.score(X_test, y_test))