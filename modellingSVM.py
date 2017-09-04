import cv2
import numpy as np
import os
import shutil
from myFeatures import FeatureExtraction
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score
import pickle

class Tester():
    def load_images_from_folder(self, folder):
        print("Load images from CK+")
        images = []
        for filename in os.listdir(folder):
            if filename.endswith(".png"):
                img = cv2.imread(os.path.join(folder, filename))
                if img is not None:
                    images.append(img)
        return images

    def run_viola(self, size):
        print("Running Viola Jones on Images")

        pathAngry = os.path.abspath('Data/Angry')
        pathDisgust = os.path.abspath('Data/Disgust')
        pathFear = os.path.abspath('Data/Fear')
        pathHappy = os.path.abspath('Data/Happy')
        pathNeutral = os.path.abspath('Data/Neutral')  # or path
        pathSad = os.path.abspath('Data/Sad')
        pathSurprise = os.path.abspath('Data/Surprise')

        Angry = self.load_images_from_folder(pathAngry)
        Disgust = self.load_images_from_folder(pathDisgust)
        Fear = self.load_images_from_folder(pathFear)
        Happy = self.load_images_from_folder(pathHappy)
        Neutral = self.load_images_from_folder(pathNeutral)
        Sad = self.load_images_from_folder(pathSad)
        Surprise = self.load_images_from_folder(pathSurprise)

        current_directory = os.getcwd()
        shutil.rmtree(current_directory + '/Data/Viola')

        if not os.path.exists(os.path.join(current_directory, r'Data/Viola/angViola')):
            os.makedirs(os.path.join(current_directory, r'Data/Viola/angViola'))
        if not os.path.exists(os.path.join(current_directory, r'Data/Viola/disViola')):
            os.makedirs(os.path.join(current_directory, r'Data/Viola/disViola'))
        if not os.path.exists(os.path.join(current_directory, r'Data/Viola/fearViola')):
            os.makedirs(os.path.join(current_directory, r'Data/Viola/fearViola'))
        if not os.path.exists(os.path.join(current_directory, r'Data/Viola/hapViola')):
            os.makedirs(os.path.join(current_directory, r'Data/Viola/hapViola'))
        if not os.path.exists(os.path.join(current_directory, r'Data/Viola/neuViola')):
            os.makedirs(os.path.join(current_directory, r'Data/Viola/neuViola'))
        if not os.path.exists(os.path.join(current_directory, r'Data/Viola/sadViola')):
            os.makedirs(os.path.join(current_directory, r'Data/Viola/sadViola'))
        if not os.path.exists(os.path.join(current_directory, r'Data/Viola/surViola')):
            os.makedirs(os.path.join(current_directory, r'Data/Viola/surViola'))

        angViola, disViola, fearViola, hapViola, neuViola, sadViola, surViola = [], [], [], [], [], [], []
        for i in range(0, size):  # change
            if i < len(Angry):
                v1, image = self.ahed.viola_jones(Angry[i])
                angViola.append(v1)
                cv2.imwrite(current_directory + "Data/Viola/angViola/angry" + str(i) + ".png", v1)
            if i < len(Disgust):
                v2, image = self.ahed.viola_jones(Disgust[i])
                disViola.append(v2)
                cv2.imwrite(current_directory + "Data/Viola/disViola/disgust" + str(i) + ".png", v2)
            if i < len(Fear):
                v3, image = self.ahed.viola_jones(Fear[i])
                fearViola.append(v3)
                cv2.imwrite(current_directory + "Data/Viola/fearViola/fear" + str(i) + ".png", v3)
            if i < len(Happy):
                v4, image = self.ahed.viola_jones(Happy[i])
                hapViola.append(v4)
                cv2.imwrite(current_directory + "Data/Viola/hapViola/happy" + str(i) + ".png", v4)
            if i < len(Neutral):
                v5, image = self.ahed.viola_jones(Neutral[i])
                neuViola.append(v5)
                cv2.imwrite(current_directory + "Data/Viola/neuViola/neutral" + str(i) + ".png", v5)
            if i < len(Sad):
                v6, image = self.ahed.viola_jones(Sad[i])
                sadViola.append(v6)
                cv2.imwrite(current_directory + "Data/Viola/sadViola/sad" + str(i) + ".png", v6)
            if i < len(Surprise):
                v7, image = self.ahed.viola_jones(Surprise[i])
                surViola.append(v7)
                cv2.imwrite(current_directory + "Data/Viola/surViola/surprise" + str(i) + ".png", v7)

        return angViola, disViola, fearViola, hapViola, neuViola, sadViola, surViola

    def run_hog(self, angViola, disViola, fearViola, hapViola, neuViola, sadViola, surViola):
        print("Running Hog on Viola Images")
        angTrain, disTrain, fearTrain, hapTrain, neuTrain, sadTrain, surTrain = [], [], [], [], [], [], []
        angLabel, disLabel, fearLabel, hapLabel, neuLabel, sadLabel, surLabel = [], [], [], [], [], [], []

        for i in range(0, 20):
            if i < len(angViola):
                angTrain.append(self.ahed.hog_opencv(angViola[i]))
                angLabel.append(1)
            if i < len(disViola):
                disTrain.append(self.ahed.hog_opencv(disViola[i]))
                disLabel.append(2)
            if i < len(fearViola):
                fearTrain.append(self.ahed.hog_opencv(fearViola[i]))
                fearLabel.append(3)
            if i < len(hapViola):
                hapTrain.append(self.ahed.hog_opencv(hapViola[i]))
                hapLabel.append(4)
            if i < len(neuViola):
                neuTrain.append(self.ahed.hog_opencv(neuViola[i]))
                neuLabel.append(5)
            if i < len(sadViola):
                sadTrain.append(self.ahed.hog_opencv(sadViola[i]))
                sadLabel.append(6)
            if i < len(surViola):
                surTrain.append(self.ahed.hog_opencv(surViola[i]))
                surLabel.append(7)

        angTest, disTest, fearTest, hapTest, neuTest, sadTest, surTest = [], [], [], [], [], [], []
        angLabel2, disLabel2, fearLabel2, hapLabel2, neuLabel2, sadLabel2, surLabel2 = [], [], [], [], [], [], []
        for i in range(20, 84):
            if i < len(angViola):
                angTest.append(self.ahed.hog_opencv(angViola[i]))
                angLabel2.append(1)
            if i < len(disViola):
                disTest.append(self.ahed.hog_opencv(disViola[i]))
                disLabel2.append(2)
            if i < len(fearViola):
                fearTest.append(self.ahed.hog_opencv(fearViola[i]))
                fearLabel2.append(3)
            if i < len(hapViola):
                hapTest.append(self.ahed.hog_opencv(hapViola[i]))
                hapLabel2.append(4)
            if i < len(neuViola):
                neuTest.append(self.ahed.hog_opencv(neuViola[i]))
                neuLabel2.append(5)
            if i < len(sadViola):
                sadTest.append(self.ahed.hog_opencv(sadViola[i]))
                sadLabel2.append(6)
            if i < len(surViola):
                surTest.append(self.ahed.hog_opencv(surViola[i]))
                surLabel2.append(7)

        train = angTrain + disTrain + fearTrain + hapTrain + neuTrain + sadTrain + surTrain
        test = angTest + disTest + fearTest + hapTest + neuTest + sadTest + surTest
        labelTrain = angLabel + disLabel + fearLabel + hapLabel + neuLabel + sadLabel + surLabel
        labelTest = angLabel2 + disLabel2 + fearLabel2 + hapLabel2 + neuLabel2 + sadLabel2 + surLabel2

        return train, test, labelTrain, labelTest

    def main(self):
        self.ahed = FeatureExtraction()
        angViola, disViola, fearViola, hapViola, neuViola, sadViola, surViola = self.run_viola(84)  # 84 = max images in folder
        train, test, labelTrain, labelTest = self.run_hog(angViola, disViola, fearViola, hapViola, neuViola, sadViola, surViola)

        # BEFORE SVM TRAINING & GRID SEARCH:
        #param_grid = [
        #	{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        #	{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        #	 ]
        #svc = svm.SVC(probability=True)  # verbose=1)# I will also advice using verbose to see behavior of your svm and grid search
        #print("SVM Grid Search")
        #clf = GridSearchCV(svc, param_grid)  # ,verbose=1)

        #print("SVM Train")
        #clf = clf.fit(train, labelTrain)
        #filename = 'finalized_model.sav'
        #pickle.dump(clf, open(filename, 'wb'))
        #print("SVM Predict")
        #predict = clf.predict(test)

        #AFTER SVM TRAINING:
        clf = pickle.load(open('finalized_model.sav', 'rb'))
        print("SVM Predict:")
        predict = clf.predict(test)
        print("SVM Best Estimator:")
        print(clf.best_estimator_)
        print("SVM Grid Scores:")
        print(clf.grid_scores_)
        print("SVM Classification Report:")
        print(classification_report(labelTest, predict, target_names=["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]))
        print("SVM Confusion Matrix:")
        print(confusion_matrix(labelTest, predict, labels=range(7)))
        print("SVM Accuracy Score:")
        print(accuracy_score(labelTest, predict, normalize=True)) #normalize=False))

        # JUNK:
        # print(sorted(clf.cv_results_.keys()))
        # print(clf.cv_results_['params']) #cv_results is a dictionary
        # print(clf.best_index_)
        # clf = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
        # decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
        # max_iter=-1, probability=False, random_state=None, shrinking=True,
        # tol=0.001, verbose=False)
        # C_range = 10.0 ** np.arange(-4, 4)
        # gamma_range = 10.0 ** np.arange(-4, 4)
        # param_grid = [
        #    {'C': C_range.tolist(), 'kernel': ['linear']},
        #    {'C': C_range.tolist(), 'gamma': gamma_range.tolist(), 'kernel': ['rbf']},
        # ]


if __name__ == '__main__': Tester().main()
