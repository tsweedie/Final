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
        self.capturing = False
        self.c = cv2.VideoCapture(0)
        #-----------------------video-------------------------
        #self.capture = Capture()
        self.startVideoButton.clicked.connect(self.startCapture)
        self.endVideoButton.clicked.connect(self.endCapture)
        self.quitVideoButton.clicked.connect(self.quitCapture)

        pixmap = QtGui.QPixmap('lense.png')
        pixmap = pixmap.scaled(200, 200)
        self.emojiLabel.setPixmap(pixmap)
        #self.resize(200, 200) window resize
        pixmapVideo = QtGui.QPixmap('lense.png')
        pixmapVideo = pixmapVideo.scaled(440, 440)
        self.videoLabel.setPixmap(pixmapVideo)

        self.videoLabel.show()
        self.emojiLabel.show()
        #-------------------------------------------------
    def startCapture(self):
        print "pressed start"
        self.capturing = True
        cap = self.c
        ahed = FeatureExtraction()
        loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
        target_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        fps = 0
        while(self.capturing):
            ret, frame = cap.read()
            # where all the testing functions go:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #if ret:  # ret == true
            if fps == 10:
                fps = 0

                scaled, frame = ahed.viola_jones(frame)  #NB add means of cancelling out false positives(detecting nose as eye)
                hog = ahed.hog_opencv(scaled)

                result = loaded_model.predict(np.array([hog]))
                if result[0] == 1:
                    pixmap = QtGui.QPixmap('Data/Emojis/angry.png')
                elif result[0] == 2:
                    pixmap = QtGui.QPixmap('Data/Emojis/disgust.png')
                elif result[0] == 3:
                    pixmap = QtGui.QPixmap('Data/Emojis/fear.png')
                elif result[0] == 4:
                    pixmap = QtGui.QPixmap('Data/Emojis/happy.png')
                elif result[0] == 5:
                    pixmap = QtGui.QPixmap('Data/Emojis/neutral.png')
                elif result[0] == 6:
                    pixmap = QtGui.QPixmap('Data/Emojis/sad.png')
                else:
                    pixmap = QtGui.QPixmap('Data/Emojis/surprise.png')

                pixmap = pixmap.scaled(200, 200)
                self.emojiLabel.setPixmap(pixmap)
                print(target_names[result[0] - 1])

                result_proba = loaded_model.predict_proba(np.array([hog]))
                probabilities = result_proba[0]
                self.lcd1.display(int(probabilities[0]*100))
                self.lcd2.display(int(probabilities[1]*100))
                self.lcd3.display(int(probabilities[2]*100))
                self.lcd4.display(int(probabilities[3]*100))
                self.lcd5.display(int(probabilities[4]*100))
                self.lcd6.display(int(probabilities[5]*100))
                self.lcd7.display(int(probabilities[6]*100))
            fps = fps + 1
            img =QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            pixmapVideo = QtGui.QPixmap.fromImage(img)
            self.videoLabel.setPixmap(pixmapVideo)
            QtGui.QApplication.processEvents()

    def endCapture(self):
        print "pressed End"
        self.capturing = False

    def quitCapture(self):
        print "pressed Quit"
        cap = self.c
        cap.release()
        QtCore.QCoreApplication.quit()
if __name__ == '__main__':#
	app = QtGui.QApplication(sys.argv)#
	window = Demo()#
	window.show()#
	sys.exit(app.exec_())#