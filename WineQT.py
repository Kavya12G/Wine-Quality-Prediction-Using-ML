import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

os.chdir('c:/Users/kavya/OneDrive/Documents/')

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('WineQT.csv')
print(df.head())

df.info()

print(df.isna().sum())

# Now, try to get the transpose of the summary statistics
df_describe_transposed = df.describe().T
print(df_describe_transposed)

df.isnull().sum()

for col in df.columns:
 if df[col].isnull().sum() > 0:
	 df[col] = df[col].fillna(df[col].mean())

df.isnull().sum().sum()

df.hist(bins=20, figsize=(10, 10))
plt.show()

plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()

plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()

df = df.drop('total sulfur dioxide', axis=1)

df['best quality'] = [1 if x > 5 else 0 for x in df.quality]

df.replace({'white': 1, 'red': 0}, inplace=True)

features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
	features, target, test_size=0.2, random_state=40)

print(xtrain.shape, xtest.shape)

norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)
# Train the XGBoost model
model = XGBClassifier()
model.fit(xtrain, ytrain)

# Save the trained model to a file
model_path = 'wine_model.pkl'
joblib.dump(model, model_path)
print(f"Trained model saved to {model_path}")

models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]


for i in range(3):
	models[i].fit(xtrain, ytrain)

	print(f'{models[i]} : ')
	print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
	print('Validation Accuracy : ', metrics.roc_auc_score(
		ytest, models[i].predict(xtest)))
	print()
	
y_pred = models[1].predict(xtest)

cm = confusion_matrix(ytest, y_pred)

# Plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

print(metrics.classification_report(ytest,
									models[1].predict(xtest)))












