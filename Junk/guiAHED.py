import cv2
import numpy as np
from myFeatures import FeatureExtraction
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
#-----------------QT------------------------------------------
import sys
from PyQt4 import QtCore, QtGui, uic
qtCreatorFile = "userinterface.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
#--------------------------------------------------------------
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

class Capture():
    def __init__(self):
        self.capturing = False
        self.c = cv2.VideoCapture(0)


    def startCapture(self):
        print "pressed start"
        self.capturing = True
        cap = self.c
        ahed = FeatureExtraction()
        loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
        target_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        #loaded_model.probability = True
        #result = loaded_model.score(X_test, Y_test)
        #print(result)
        while(self.capturing):
            ret, frame = cap.read()
            # where all the testing functions go:
            if ret:  # ret == true
                scaled, frame = ahed.viola_jones(frame)  #NB add means of cancelling out false positives
                hog = ahed.hog_opencv(scaled)
                result = loaded_model.predict(np.array([hog]))
                print(target_names[result[0]-1])
                result_proba = loaded_model.predict_proba(np.array([hog]))
                print(result_proba)

            #
            cv2.imshow("Live Video Stream", frame)
            cv2.waitKey(5)
        cv2.destroyAllWindows()

    def endCapture(self):
        print "pressed End"
        self.capturing = False

    def quitCapture(self):
        print "pressed Quit"
        cap = self.c
        cv2.destroyAllWindows()
        cap.release()
        QtCore.QCoreApplication.quit()

class Demo(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):#
        QtGui.QMainWindow.__init__(self)#
        Ui_MainWindow.__init__(self)#
        self.setupUi(self)#
        #-----------------------video-------------------------
        self.capture = Capture()
        self.startVideoButton.clicked.connect(self.capture.startCapture)
        self.endVideoButton.clicked.connect(self.capture.endCapture)
        self.quitVideoButton.clicked.connect(self.capture.quitCapture)

        self.lcd1.display(100)
        self.lcd2.display(1)
        self.lcd3.display(1)
        self.lcd4.display(1)
        self.lcd5.display(1)
        self.lcd6.display(1)
        self.lcd7.display(1)
        #-------------------------------------------------
if __name__ == '__main__':#
	app = QtGui.QApplication(sys.argv)#
	window = Demo()#
	window.show()#
	sys.exit(app.exec_())#