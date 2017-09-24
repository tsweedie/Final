import os
from extractFeatures import FeatureExtraction
from modelData import MineData
import csv
import cv2
import numpy as np
import sys
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle


class test():

    def getData(self, subject):

        pathAngry = os.path.abspath('Data/TESTING/subjects/' + subject + '/Angry')
        pathDisgust = os.path.abspath('Data/TESTING/subjects/' + subject + '/Disgust')
        pathFear = os.path.abspath('Data/TESTING/subjects/' + subject + '/Fear')
        pathHappy = os.path.abspath('Data/TESTING/subjects/' + subject + '/Happy')
        pathNeutral = os.path.abspath('Data/TESTING/subjects/' + subject + '/Neutral')
        pathSad = os.path.abspath('Data/TESTING/subjects/' + subject + '/Sad')
        pathSurprise = os.path.abspath('Data/TESTING/subjects/' + subject + '/Surprise')

        Angry = self.modeldata.load_images_from_folder(pathAngry)
        Disgust = self.modeldata.load_images_from_folder(pathDisgust)
        Fear = self.modeldata.load_images_from_folder(pathFear)
        Happy = self.modeldata.load_images_from_folder(pathHappy)
        Neutral = self.modeldata.load_images_from_folder(pathNeutral)
        Sad = self.modeldata.load_images_from_folder(pathSad)
        Surprise = self.modeldata.load_images_from_folder(pathSurprise)

        v1, image = self.features.viola_jones(Angry[0])
        v2, image = self.features.viola_jones(Disgust[0])
        v3, image = self.features.viola_jones(Fear[0])
        v4, image = self.features.viola_jones(Happy[0])
        v5, image = self.features.viola_jones(Neutral[0])
        v6, image = self.features.viola_jones(Sad[0])
        v7, image = self.features.viola_jones(Surprise[0])

        featureVectors = []

        angVector = self.features.hog_opencv(v1)
        angVector = angVector.tolist()
        angVector.insert(0, 1)
        featureVectors.append(angVector)

        disVector = self.features.hog_opencv(v2)
        disVector = disVector.tolist()
        disVector.insert(0, 2)
        featureVectors.append(disVector)

        fearVector = self.features.hog_opencv(v3)
        fearVector = fearVector.tolist()
        fearVector.insert(0, 3)
        featureVectors.append(fearVector)

        hapVector = self.features.hog_opencv(v4)
        hapVector = hapVector.tolist()
        hapVector.insert(0, 4)
        featureVectors.append(hapVector)

        neuVector = self.features.hog_opencv(v5)
        neuVector = neuVector.tolist()
        neuVector.insert(0, 5)
        featureVectors.append(neuVector)

        sadVector = self.features.hog_opencv(v6)
        sadVector = sadVector.tolist()
        sadVector.insert(0, 6)
        featureVectors.append(sadVector)

        surVector = self.features.hog_opencv(v7)
        surVector = surVector.tolist()
        surVector.insert(0, 7)
        featureVectors.append(surVector)

        #with open('TestAHED/' + subject + '.csv', 'wb+') as f:
        #    writer = csv.writer(f)
        #    writer.writerows(featureVectors)
        return  np.array(featureVectors)

    def runTest(self, data, modelname, subject):
        print '\n######################################################################'
        print '##########################~', subject, '~#####################'
        print '######################################################################'
        dataset = data
        #dataset = pd.read_csv(data)
        #print dataset.shape
        X = dataset[:, 1:]
        y = dataset[:,0]

        print y

        print '\n######################################################################'
        print '#############################~SVM Predict~############################'
        print '######################################################################'
        clf = pickle.load(open(modelname, 'rb'))
        y_pred = clf.predict(X)
        print 'SVM Classification Report:'
        print classification_report(y, y_pred,
                                    target_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
        print 'SVM Confusion Matrix:'
        print confusion_matrix(y, y_pred, labels = [1,2,3,4,5,6,7])
        print '\t SVM Accuracy Score:', accuracy_score(y, y_pred, normalize=True)


    def main(self):
        print 'Starting...'
        self.features = FeatureExtraction()
        self.modeldata = MineData()
        # pathsubject32 = os.path.abspath('Data/TESTING/subjects/subject32')
        subjects = ['subject11' , 'subject22' , 'subject37' , 'subject42', 'subject44' , 'subject50' , 'subject55' , 'subject67' , 'subject71']
        print 'Output is printed to TestAHED/testing.out'
        orig_stdout = sys.stdout
        output = open('TestAHED/testing.out', 'w+')
        sys.stdout = output
        for sub in subjects:
            fv = self.getData(sub)
            self.runTest(fv, 'finalized_model.sav',sub)
            #self.runTest( 'TestAHED/' + sub + '.csv', 'finalized_model.sav')
        sys.stdout.close()
        sys.stdout = orig_stdout
        print 'Done ^_^'

if __name__ == '__main__': test().main()