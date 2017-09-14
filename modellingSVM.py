import cv2
import numpy as np
import pandas as pd
import seaborn as sb
import os
import shutil
from myFeatures import FeatureExtraction
from sklearn import svm
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score
import pickle

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#MENTION LABELS NB NB
class Tester():
    def run_test(self,data,modelname):
        dataset = pd.read_csv(data)
        X = dataset.ix[:,1:].values
        y = dataset.ix[:,0].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, stratify = y, random_state=40)
        # BEFORE SVM TRAINING & GRID SEARCH:
        param_grid = [
        	{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        	{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        	 ]
        #C_range = np.logspace(-5, 15,num=21,base = 10.0)#, 6)
        #print(C_range)
        #gamma_range = np.logspace(-15, 3, num=19,base = 10.0)#, 6)
        #print(gamma_range)
        #param_grid = [
        #   {'C': C_range.tolist(), 'kernel': ['linear']},
        #   {'C': C_range.tolist(), 'gamma': gamma_range.tolist(), 'kernel': ['rbf']},
        #]
        svc = svm.SVC(probability=True)
        print("SVM Grid Search")#easy.p & grid.py ??
        clf = GridSearchCV(svc, param_grid)  # ,verbose=1)
	scores = cross_val_score(clf, X, y, cv=5)
	print('Cross validation scores:',scores)
	print("\t Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        clf = clf.fit(X_train, y_train)
        print('Best score for data1:', clf.best_score_)
        print('Best C:', clf.best_estimator_.C)
        print('Best Kernel:', clf.best_estimator_.kernel)
        print('Best Gamma:', clf.best_estimator_.gamma)
        print("SVM Train")
        filename = modelname
        pickle.dump(clf, open(filename, 'wb'))
        #AFTER SVM TRAINING:
        #clf = pickle.load(open(modelname, 'rb'))
        print("SVM Predict:")
        y_pred = clf.predict(X_test)
        print("SVM Best Estimator:")
        print(clf.best_estimator_)
        print("SVM Grid Scores:")
        print(clf.grid_scores_)
        print("SVM Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]))
        print("SVM Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred, labels=range(7)))
        print("SVM Accuracy Score:")
        print(accuracy_score(y_test, y_pred, normalize=True))


    def main(self):
	
	self.ahed = FeatureExtraction()
        self.run_test('dataCsv.csv','finalized_model.sav')
	self.run_test('myHogdDataCsv.csv','myHogfinalized_model.sav')

if __name__ == '__main__': Tester().main()
