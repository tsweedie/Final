import cv2
import numpy as np
from skimage.feature import hog
from skimage import img_as_float

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
class FeatureExtraction():
    # ------------------------------------------------------------------------------
    # ----------------------------Viola Jones Face Detection------------------------
    # ------------------------------------------------------------------------------
    def viola_jones(self, image):
        height = 56
        width = 56
        scaled = np.zeros(( width,height), dtype=np.float)
        faces = faceCascade.detectMultiScale(image, 1.3, 5)

        max_size = 0  # w*h
        X = 0
        Y = 0
        W = 0
        H = 0

        for (x, y, w, h) in faces:  # find the biggest face
            if max_size < w * h:
                max_size = w * h
                X = x
                Y = y
                W = w
                H = h
        # cater for when there are no faces in the image "max_size = 0"
        if max_size != 0:
            cv2.rectangle(image, (X, Y), (X + W, Y + H), (192, 192, 192), 2)
            roi = image[Y:Y + H, X:X + W]
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            scaled = cv2.resize(gray, (width,height))

            #eyes = eyeCascade.detectMultiScale(roi)
            #for ex, ey, ew, eh in eyes:
            #    cv2.rectangle(roi, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

            #circle_img = np.zeros((height, width), np.uint8)
            #cv2.ellipse(circle_img, (height / 2, width / 2), (24, 28), 0, 0, 360, 255, thickness=-1)
            #masked_data = cv2.bitwise_and(scaled, scaled, mask=circle_img)
            #scaled = masked_data
        #return scaled
        return scaled, image

        # ------------------------------------------------------------------------------
        # ------------------------------My HOG: Gradients-----------------------------------
        # ------------------------------------------------------------------------------
        # figure out how to create actual hogs using mag and orien image

    def gx(self, scaled):
        y, x = scaled.shape
        gx = np.zeros(scaled.shape)
        for i in range(y):
            gx[i, :] = np.convolve(scaled[i, :], [1, 0, -1], 'same')
        return gx

    def gy(self, scaled):
        y, x = scaled.shape
        gy = np.zeros(scaled.shape)
        for j in range(x):
            gy[:, j] = np.convolve(scaled[:, j], [1, 0, -1], 'same')
        return gy

    def magnitude(self, gx, gy):
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        return magnitude

    def orientation(self, gx, gy):
        orientation = (np.arctan2(gy, gx) * 180 / np.pi) % 360
        return orientation
        # def calculate_hog(self,bins,cell,block,winsize):
        # return hog

        # ------------------------------------------------------------------------------
        # -------------------------------------HOG--------------------------------------
        # ------------------------------------------------------------------------------

    def hog_opencv(self, image):

        image = img_as_float(image)  # convert unit8 tofloat64 ... dtype
        orientations = 9
        cellSize = (8, 8)  # pixels_per_cell
        blockSize = (3, 3)  # cells_per_block
        blockNorm = 'L1-sqrt'  # {'L1', 'L1-sqrt', 'L2', 'L2-Hys'}
        visualize = True  # Also return an image of the HOG.
        transformSqrt = False
        featureVector = True
        fd, hog_image = hog(image, orientations, cellSize, blockSize, blockNorm, visualize, transformSqrt,
                            featureVector)
        # cv2.imwrite('hog.jpg', hog_image)
        # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))  # Rescale histogram for better display

        return fd
