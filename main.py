'''
Multiclass classifier based on Mahalanobis distance (https://en.wikipedia.org/wiki/Mahalanobis_distance)
Vasilios Mavroudis, 2019
Generalized version of the binary classifier shown here: https://www.machinelearningplus.com/statistics/mahalanobis-distance/
'''


from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from classifiers import MahalanobisClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

# Load and preprocess data
df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv', header=None)
#df.dropna(inplace=True)  # Drop missing values.

# Split data
samples = df.iloc[:,0:7]
labels = df.iloc[:,7]
unique_labels = np.unique(labels) #To return the correct labels when predicting
samples, new_samples, labels, new_labels = train_test_split(samples, labels, test_size=0.5, random_state=100)

# "Training"
clf = MahalanobisClassifier(samples, labels)

# Predicting
pred_probs = clf.predict_probability(new_samples)
pred_class = clf.predict_class(new_samples,unique_labels)


pred_actuals = pd.DataFrame([(pred, act) for pred, act in zip(pred_class, new_labels)], columns=['pred', 'true'])
#print(pred_actuals[:25])


truth = pred_actuals.loc[:, 'true']
pred = pred_actuals.loc[:, 'pred']
scores = np.array(pred_probs)[:, 1]
print('\nAccuracy Score: ', accuracy_score(truth, pred))
print('\nClassification Report: \n', classification_report(truth, pred))