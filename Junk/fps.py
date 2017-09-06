import cv2

videoCapture = cv2.VideoCapture(0)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# self.c.set(cv2.CV_CAP_PROP_FPS, 60)
frameps = videoCapture.get(cv2.CAP_PROP_FPS)
print(frameps)
featureVectors = np.empty((0))
        for i in range(0, 84):
            if i < len(angViola):
                angVector = self.ahed.hog_opencv(angViola[i])
                np.insert(featureVectors,0, np.insert(angVector,0,1),axis=0)
            if i < len(disViola):
                disVector = self.ahed.hog_opencv(disViola[i])
                np.insert(featureVectors,0, np.insert(disVector,0,2),axis=0)
            if i < len(fearViola):
                fearVector = self.ahed.hog_opencv(fearViola[i])
                np.insert(featureVectors,0, np.insert(fearVector,0,3),axis=0)
            if i < len(hapViola):
                hapVector = self.ahed.hog_opencv(hapViola[i])
                np.insert(featureVectors,0, np.insert(hapVector,0,4),axis=0)
            if i < len(neuViola):
                neuVector = self.ahed.hog_opencv(neuViola[i])
                np.insert(featureVectors,0, np.insert(neuVector,0,5),axis=0)
            if i < len(sadViola):
                sadVector = self.ahed.hog_opencv(sadViola[i])
                print(sadVector.shape)
                np.insert(featureVectors,0, np.insert(sadVector,0,6),axis=0)
            if i < len(surViola):
                surVector = self.ahed.hog_opencv(surViola[i])
                np.insert(featureVectors,0, np.insert(surVector,0,7),axis=0)

for i in range(0, 84):
    if i < len(angViola):
        angVector = self.ahed.hog_opencv(angViola[i])
        angVector = angVector.tolist()
        angVector.insert(0, 1)
        featureVectors.append(angVector)
    if i < len(disViola):
        disVector = self.ahed.hog_opencv(disViola[i])
        disVector = disVector.tolist()
        disVector.insert(0, 2)
        featureVectors.append(disVector)
    if i < len(fearViola):
        fearVector = self.ahed.hog_opencv(fearViola[i])
        fearVector = fearVector.tolist()
        fearVector.insert(0, 3)
        featureVectors.append(fearVector)
    if i < len(hapViola):
        hapVector = self.ahed.hog_opencv(hapViola[i])
        hapVector = hapVector.tolist()
        hapVector.insert(0, 4)
        featureVectors.append(hapVector)
    if i < len(neuViola):
        neuVector = self.ahed.hog_opencv(neuViola[i])
        neuVector = neuVector.tolist()
        neuVector.insert(0, 5)
        featureVectors.append(neuVector)
    if i < len(sadViola):
        sadVector = self.ahed.hog_opencv(sadViola[i])
        sadVector = sadVector.tolist()
        sadVector.insert(0, 6)
        featureVectors.append(sadVector)
    if i < len(surViola):
        surVector = self.ahed.hog_opencv(surViola[i])
        surVector = surVector.tolist()
        surVector.insert(0, 7)
        featureVectors.append(surVector)