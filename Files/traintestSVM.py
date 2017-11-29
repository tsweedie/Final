import sys
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle


class Tester():
    def run_test(self, data, modelname):
        print '\n######################################################################'
        print '##########################~', modelname, '~#####################'
        print '######################################################################'

        dataset = pd.read_csv(data)

        X = dataset.ix[:, 1:].values
        y = dataset.ix[:, 0].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.40, stratify=y, random_state=40)

		# Grid search parameters:
		
        # C = np.logspace(-5, 15,num=21,base = 2.0)
        # gamma = np.logspace(-15, 3, num=19,base = 2.0)
        param_grid = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        ]
		
		# choose svm type: SVC - Support Vector Classification(based on libsvm)
        svc = svm.SVC(probability=True)

        print '\n######################################################################'
        print '#############################~SVM Grid Search~########################'
        print '######################################################################'
        clf = GridSearchCV(svc, param_grid)
        print '\t Parameter Grid:\n', param_grid

        print '\n######################################################################'
        print '##########################~SVM Cross Validation~######################'
        print '######################################################################'
        skf = StratifiedKFold(n_splits=3, random_state=None, shuffle=False)
        scores = cross_val_score(clf, X, y, cv=skf, n_jobs=-1)
        print '\t Cross validation scores: \t', scores
        print '\t Cross validation Accuracy:\t %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)

        print '\n######################################################################'
        print '#############################~SVM Train~##############################'
        print '######################################################################'
        clf = clf.fit(X_train, y_train)
        filename = modelname
        pickle.dump(clf, open('Files/'+filename, 'wb'))
        print '\t Best score for classifier:\t', clf.best_score_
        print '\t Best C:\t', clf.best_estimator_.C
        print '\t Best Kernel:\t', clf.best_estimator_.kernel
        print '\t Best Gamma:\t', clf.best_estimator_.gamma
        print '\t SVM Best Estimator:\t', clf.best_estimator_
        print '\n\t SVM Grid Scores: \n', clf.cv_results_

        print '\n######################################################################'
        print '#############################~SVM Predict~############################'
        print '######################################################################'
        clf = pickle.load(open('Files/'+modelname, 'rb'))
        y_pred = clf.predict(X_test)
        print 'SVM Classification Report:'
        print classification_report(y_test, y_pred,
                                    target_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
        print 'SVM Confusion Matrix:'

        print confusion_matrix(y_test, y_pred, labels = [1,2,3,4,5,6,7])
        print '\t SVM Accuracy Score:', accuracy_score(y_test, y_pred, normalize=True)
        print 'file:', data

        print '\n######################################################################'
        print '#############################~SVM Test Results~############################'
        print '######################################################################'
        for i in range(len(X_test_names)):
            print 'Subject: ',X_test_names[i],'\t Label: ',y_test[i],'\t Prediction: ',y_pred[i]

    def main(self):
        print 'Output is printed to Files/test.out'
        orig_stdout = sys.stdout
        output = open('Files/test.out', 'w+')
        sys.stdout = output
        self.run_test('Files/dataCsv.csv', 'finalized_model.sav')
        sys.stdout.close()
        sys.stdout = orig_stdout
        print 'Done ^_^'
if __name__ == '__main__': Tester().main()
