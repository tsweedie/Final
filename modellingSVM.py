import cv2
import json
import numpy as np
import pandas as pd
import seaborn as sb
import os
import sys
import shutil
import pickle
from myFeatures import FeatureExtraction
from sklearn import svm
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#MENTION LABELS NB NB
class Tester():
    def run_test(self,data,modelname):
	output = open("test.out", 'w')
	sys.stdout = output
        dataset = pd.read_csv(data)
        X = dataset.ix[:,1:].values
        y = dataset.ix[:,0].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.40, stratify = y, random_state=40)
        # BEFORE SVM TRAINING & GRID SEARCH:
        param_grid = [
        	{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        	{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        	 ]
        svc = svm.SVC(probability=True)
	print '\n######################################################################'
        print '#############################~SVM Grid Search~########################'#easy.p & grid.py ??
	print '######################################################################'
        clf = GridSearchCV(svc, param_grid)  # ,verbose=1)
	print '\t Parameter Grid:\n',param_grid
	print '\n######################################################################'
	print '##########################~SVM Cross Validation~######################'
	print '######################################################################'
	skf = StratifiedKFold(n_splits=3, random_state=None, shuffle=False)
	scores = cross_val_score(clf, X, y, cv=skf, n_jobs=-1)
	#sorted(scores.keys())
	print '\t Cross validation scores: \t',scores
	print '\t Cross validation Accuracy:\t %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
	print '\n######################################################################'
	print '#############################~SVM Train~##############################'
	print '######################################################################'
        clf = clf.fit(X_train, y_train)
	filename = modelname
        pickle.dump(clf, open(filename, 'wb'))
        print '\t Best score for classifier:\t', clf.best_score_
        print '\t Best C:\t', clf.best_estimator_.C
        print '\t Best Kernel:\t', clf.best_estimator_.kernel
        print '\t Best Gamma:\t', clf.best_estimator_.gamma
	print '\t SVM Best Estimator:\t', clf.best_estimator_
        print '\n\t SVM Grid Scores: testfile.txt\n'#,clf.cv_results_
	#file = open('testfile.txt','w+') 
	#file.write(json.dumps(clf.cv_results_)) 
        #file.close() 

        #AFTER SVM TRAINING:
	print '\n######################################################################'
        print '#############################~SVM Predict~############################'
	print '######################################################################'
	#clf = pickle.load(open(modelname, 'rb'))
        y_pred = clf.predict(X_test)
        print 'SVM Classification Report:'
        print classification_report(y_test, y_pred, target_names=["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"])
        print 'SVM Confusion Matrix:'
        print confusion_matrix(y_test, y_pred, labels=range(7))
        print '\t SVM Accuracy Score:', accuracy_score(y_test, y_pred, normalize=True)
	print 'file:', data
	output.close()


    def main(self):
	
	self.ahed = FeatureExtraction()
        self.run_test('dataCsv.csv','finalized_model.sav')
	#self.run_test('myHogdDataCsv.csv','myHogfinalized_model.sav')

if __name__ == '__main__': Tester().main()
