import cv2
import numpy as np
from myFeatures import FeatureExtraction
import pickle
# -----------------QT------------------------------------------
import sys
from PyQt4 import QtCore, QtGui, uic
qtCreatorFile = "userinterface.ui"  # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
# --------------------------------------------------------------
class Demo(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):  #
        QtGui.QMainWindow.__init__(self)  #
        Ui_MainWindow.__init__(self)  #
        self.setupUi(self)  #
        self.capturing = False
        self.c = cv2.VideoCapture(0)

        self.startVideoButton.clicked.connect(self.startCapture)
        self.endVideoButton.clicked.connect(self.endCapture)
        self.quitVideoButton.clicked.connect(self.quitCapture)

        pixmap = QtGui.QPixmap('lense.png')
        pixmap = pixmap.scaled(256, 256)
        self.emojiLabel.setPixmap(pixmap)

        pixmapVideo = QtGui.QPixmap('lense.png')
        pixmapVideo = pixmapVideo.scaled(450, 450)
        self.videoLabel.setPixmap(pixmapVideo)

        self.videoLabel.show()
        self.emojiLabel.show()

    def startCapture(self):
        print "pressed start"
        self.capturing = True
        cap = self.c

        ahed = FeatureExtraction()
        loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
        target_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        emojis = {1: 'Data/Emojis/angry.png', 2: 'Data/Emojis/disgust.png', 3: 'Data/Emojis/fear.png',
                  4: 'Data/Emojis/happy.png', 5: 'Data/Emojis/neutral.png', 6: 'Data/Emojis/sad.png',
                  7: 'Data/Emojis/surprise.png'}
        fps = 0
        while (self.capturing):
            ret, frame = cap.read()
            # where all the testing functions go:
            if fps == 5 and ret:  # ret == true
                fps = 0
                scaled, frame = ahed.viola_jones(
                    frame)  # NB add means of cancelling out false positives(detecting nose as eye)

                if np.sum(scaled) != 0:
                    hog = ahed.hog_opencv(scaled)
                    result = loaded_model.predict(np.array([hog]))
                    pixmap = QtGui.QPixmap(emojis[result[0]])
                    pixmap = pixmap.scaled(256, 256)
                    self.emojiLabel.setPixmap(pixmap)
                    print(target_names[result[0] - 1])
                    result_proba = loaded_model.predict_proba(np.array([hog]))
                    probabilities = (result_proba[0]) * 100
                    probabilities.astype(np.int64)
                    self.lcd1.display(probabilities[0])
                    #self.lcd2.display(probabilities[1])
                    #self.lcd3.display(probabilities[2])
                    self.lcd4.display(probabilities[1])#[3])
                    self.lcd5.display(probabilities[2])#[4])
                    #self.lcd6.display(probabilities[5])
                    self.lcd7.display(probabilities[3])#[6])

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
                pixmapVideo = QtGui.QPixmap.fromImage(img)
                self.videoLabel.setPixmap(pixmapVideo)
                QtGui.QApplication.processEvents()
            fps = fps + 1

    def endCapture(self):
        print "pressed End"
        self.capturing = False

    def quitCapture(self):
        print "pressed Quit"
        cap = self.c
        cap.release()
        QtCore.QCoreApplication.quit()

if __name__ == '__main__':  #
    app = QtGui.QApplication(sys.argv)  #
    window = Demo()  #
    window.show()  #
    sys.exit(app.exec_())  #
