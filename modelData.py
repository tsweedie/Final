import cv2
import os
import shutil
from extractFeatures import FeatureExtraction
import csv
import xlsxwriter



class MineData():
    def load_images_from_folder(self, folder):
        print 'Load images from CK+', folder
        images = []
        names = []
        for filename in os.listdir(folder):
            if filename.endswith('.png'):
                img = cv2.imread(os.path.join(folder, filename))
                if img is not None:
                    images.append(img)
                    names.append(filename)
        return images,names

    def main(self):
        print 'Starting...'
        self.features = FeatureExtraction()

        pathAngry = os.path.abspath('Data/Angry')
        pathDisgust = os.path.abspath('Data/Disgust')
        pathFear = os.path.abspath('Data/Fear')
        pathHappy = os.path.abspath('Data/Happy')
        pathNeutral = os.path.abspath('Data/Neutral') 
        pathSad = os.path.abspath('Data/Sad')
        pathSurprise = os.path.abspath('Data/Surprise')

        Angry, angNames = self.load_images_from_folder(pathAngry)
        Disgust, disNames = self.load_images_from_folder(pathDisgust)
        Fear, fearNames = self.load_images_from_folder(pathFear)
        Happy, hapNames = self.load_images_from_folder(pathHappy)
        Neutral, neuNames = self.load_images_from_folder(pathNeutral)
        Sad, sadNames = self.load_images_from_folder(pathSad)
        Surprise, surNames = self.load_images_from_folder(pathSurprise)

        current_directory = os.getcwd()
        #shutil.rmtree(current_directory + '/Data/Viola')

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

        print 'Running Viola-Jones on CK+ Images'
        angViola, disViola, fearViola, hapViola, neuViola, sadViola, surViola = [], [], [], [], [], [], []
        for i in range(0, 84):
            if i < len(Angry):
                v1, image = self.features.viola_jones(Angry[i])
                angViola.append(v1)
                cv2.imwrite(current_directory + '/Data/Viola/angViola/angry' + angNames[i] + '.png', v1)
            if i < len(Disgust):
                v2, image = self.features.viola_jones(Disgust[i])
                disViola.append(v2)
                cv2.imwrite(current_directory + '/Data/Viola/disViola/disgust' + disNames[i] + '.png', v2)
            if i < len(Fear):
                v3, image = self.features.viola_jones(Fear[i])
                fearViola.append(v3)
                cv2.imwrite(current_directory + '/Data/Viola/fearViola/fear' + fearNames[i] + '.png', v3)
            if i < len(Happy):
                v4, image = self.features.viola_jones(Happy[i])
                hapViola.append(v4)
                cv2.imwrite(current_directory + '/Data/Viola/hapViola/happy' + hapNames[i] + '.png', v4)
            if i < len(Neutral):
                v5, image = self.features.viola_jones(Neutral[i])
                neuViola.append(v5)
                cv2.imwrite(current_directory + '/Data/Viola/neuViola/neutral' + neuNames[i] + '.png', v5)
            if i < len(Sad):
                v6, image = self.features.viola_jones(Sad[i])
                sadViola.append(v6)
                cv2.imwrite(current_directory + '/Data/Viola/sadViola/sad' + sadNames[i] + '.png', v6)
            if i < len(Surprise):
                v7, image = self.features.viola_jones(Surprise[i])
                surViola.append(v7)
                cv2.imwrite(current_directory + '/Data/Viola/surViola/surprise' + surNames[i] + '.png', v7)
                
        print 'Running Hog on Viola-Jones Images'
        featureVectors = []
        for i in range(0, 84):
            if i < len(angViola):
                angVector = self.features.hog_opencv(angViola[i])
                angVector = angVector.tolist()
                angVector.insert(0, 1)
                angVector.insert(0, angNames[i])
                featureVectors.append(angVector)
            if i < len(disViola):
                disVector = self.features.hog_opencv(disViola[i])
                disVector = disVector.tolist()
                disVector.insert(0, 2)
                disVector.insert(0, disNames[i])
                featureVectors.append(disVector)
            if i < len(fearViola):
                fearVector = self.features.hog_opencv(fearViola[i])
                fearVector = fearVector.tolist()
                fearVector.insert(0, 3)
                fearVector.insert(0, fearNames[i])
                featureVectors.append(fearVector)
            if i < len(hapViola):
                hapVector = self.features.hog_opencv(hapViola[i])
                hapVector = hapVector.tolist()
                hapVector.insert(0, 4)
                hapVector.insert(0, hapNames[i])
                featureVectors.append(hapVector)
            if i < len(neuViola):
                neuVector = self.features.hog_opencv(neuViola[i])
                neuVector = neuVector.tolist()
                neuVector.insert(0, 5)
                neuVector.insert(0, neuNames[i])
                featureVectors.append(neuVector)
            if i < len(sadViola):
                sadVector = self.features.hog_opencv(sadViola[i])
                sadVector = sadVector.tolist()
                sadVector.insert(0, 6)
                sadVector.insert(0, sadNames[i])
                featureVectors.append(sadVector)
            if i < len(surViola):
                surVector = self.features.hog_opencv(surViola[i])
                surVector = surVector.tolist()
                surVector.insert(0, 7)
                surVector.insert(0, surNames[i])
                featureVectors.append(surVector)
                
        with open('Files/dataCsv.csv', 'wb+') as f:
            writer = csv.writer(f)
            writer.writerows(featureVectors)
            
        workbook = xlsxwriter.Workbook('Files/dataExcel.xlsx')
        worksheet = workbook.add_worksheet()
        for row, vector in enumerate(featureVectors):
            for col, value in enumerate(vector):
                worksheet.write(row, col, value)
        workbook.close()

        print 'Running My Hog on Viola-Jones Images'
        featureVectors = []
        for i in range(0, 84):
            if i < len(angViola):
                angVector = self.features.calculate_myhog(angViola[i])
                angVector = angVector.tolist()
                angVector.insert(0, 1)
                featureVectors.append(angVector)
            if i < len(disViola):
                disVector = self.features.calculate_myhog(disViola[i])
                disVector = disVector.tolist()
                disVector.insert(0, 2)
                featureVectors.append(disVector)
            if i < len(fearViola):
                fearVector = self.features.calculate_myhog(fearViola[i])
                fearVector = fearVector.tolist()
                fearVector.insert(0, 3)
                featureVectors.append(fearVector)
            if i < len(hapViola):
                hapVector = self.features.calculate_myhog(hapViola[i])
                hapVector = hapVector.tolist()
                hapVector.insert(0, 4)
                featureVectors.append(hapVector)
            if i < len(neuViola):
                neuVector = self.features.calculate_myhog(neuViola[i])
                neuVector = neuVector.tolist()
                neuVector.insert(0, 5)
                featureVectors.append(neuVector)
            if i < len(sadViola):
                sadVector = self.features.calculate_myhog(sadViola[i])
                sadVector = sadVector.tolist()
                sadVector.insert(0, 6)
                featureVectors.append(sadVector)
            if i < len(surViola):
                surVector = self.features.calculate_myhog(surViola[i])
                surVector = surVector.tolist()
                surVector.insert(0, 7)
                featureVectors.append(surVector)
                
        with open('Files/myHogDataCsv.csv', 'wb+') as f:
            writer = csv.writer(f)
            writer.writerows(featureVectors)
            
        workbook = xlsxwriter.Workbook('Files/myHogDataExcel.xlsx')
        worksheet = workbook.add_worksheet()
        for row, vector in enumerate(featureVectors):
            for col, value in enumerate(vector):
                worksheet.write(row, col, value)
        workbook.close()
        print 'Done ^_^'


if __name__ == '__main__': MineData().main()
