import sys
import cv2
import numpy as np
from skimage.feature import hog
from skimage import img_as_float

#faceCascade = cv2.CascadeClassifier('Files/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('Files/haarcascade_eye.xml')
faceCascade = cv2.CascadeClassifier('Files/haarcascade_frontalface_alt.xml')


class FeatureExtraction():
    # ------------------------------------------------------------------------------
    # ----------------------------Viola-Jones Face Detection------------------------
    # ------------------------------------------------------------------------------
    def viola_jones(self, image):
        height = 56
        width = 56
        scaled = np.zeros((height, width), dtype=np.float)
        faces = faceCascade.detectMultiScale(image, 1.3, 5)

        max_size = 0  # w*h
        X = Y = W = H = 0

        for (x, y, w, h) in faces:  # find the biggest face
            if max_size < w * h:
                max_size = w * h
                X = x
                Y = y
                W = w
                H = h
        # cater for when there are no faces in the image 'max_size = 0'
        if max_size != 0:
            cv2.rectangle(image, (X, Y), (X + W, Y + H), (192, 192, 192), 2)
            roi = image[Y:Y + H, X:X + W]
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            scaled = cv2.resize(gray, (height, width))

        # circle_img = np.zeros((height, width), np.uint8)
        # cv2.ellipse(circle_img, (height / 2, width / 2), (24, 28), 0, 0, 360, 255, thickness=-1)
        # masked_data = cv2.bitwise_and(scaled, scaled, mask=circle_img)
        # scaled = masked_data
        return scaled, image

    # ------------------------------------------------------------------------------
    # -----------------------------------My HOG-------------------------------------
    # ------------------------------------------------------------------------------
    # Note: gradients work only on grayscale images

    def gx(self, scaled):
        y, x = scaled.shape
        scaled = np.lib.pad(scaled, 1, 'constant', constant_values=0)
        gx = np.zeros((x, y))
        # sobelX:
        # [1	,0	,-1]
        # [2	,0	,-2]
        # [1	,0	,-1]
        for i in range(y):
            a = np.convolve(scaled[i - 1, :], [1, 0, -1], 'valid')
            b = np.convolve(scaled[i, :], [2, 0, -2], 'valid')
            c = np.convolve(scaled[i + 1, :], [1, 0, -1], 'valid')
            gx[i, :] = np.sum([a, b, c], axis=0)
        return gx

    def gy(self, scaled):
        y, x = scaled.shape
        scaled = np.lib.pad(scaled, 1, 'constant', constant_values=0)
        gy = np.zeros((x, y))
        # sobelY:
        # [-1	,-2	,-1]
        # [0	,0	,0]
        # [1	,2	,1]
        for j in range(x):
            a = np.convolve(scaled[:, j - 1], [1, 0, -1], 'valid')
            b = np.convolve(scaled[:, j], [2, 0, -2], 'valid')
            c = np.convolve(scaled[:, j + 1], [1, 0, -1], 'valid')
            gy[:, j] = np.sum([a, b, c], axis=0)
        return gy

    def magnitude(self, gx, gy):
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        return magnitude

    def orientation(self, gx, gy):
        orientation = (np.arctan2(gy, gx) / np.pi) % 2 * np.pi  # same output as cv2.cartToPolar
        return orientation

    def calculate_myhog(self, scaled):
        gx = self.gx(scaled)
        gy = self.gy(scaled)

        magnitude = self.magnitude(gx, gy)
        orientation = self.orientation(gx, gy)

        bin_n = 9
        bins = np.int32(bin_n * orientation / (2 * np.pi))

        bin_blocks = []
        mag_blocks = []
        epsilon = sys.float_info.epsilon

        blocksize = 3 #2
        cellsize = 4

        width = scaled.shape[0]
        height = scaled.shape[1]

        y = ((height - (cellsize * blocksize)) / cellsize) + 1
        x = ((width - (cellsize * blocksize)) / cellsize) + 1

        histograms = []
        for i in range(0, y, 1):  # loops through each block in a image(i,j)
            for j in range(0, x, 1):
                bin_block = bins[i * cellsize: i * cellsize + blocksize * cellsize,
                            j * cellsize: j * cellsize + blocksize * cellsize]
                mag_block = magnitude[i * cellsize: i * cellsize + blocksize * cellsize,
                            j * cellsize: j * cellsize + blocksize * cellsize]
                tempHists = []
                sumHists = np.zeros((9,))
                for m in range(0, blocksize * cellsize, cellsize):  # loops through each cell in a block(m,n)
                    for n in range(0, blocksize * cellsize, cellsize):
                        cellBin = bin_block[m:m + cellsize, n:n + cellsize]
                        cellMag = mag_block[m:m + cellsize, n:n + cellsize]
                        tempHists.append(np.bincount(cellBin.ravel(), cellMag.ravel(), bin_n))
                        sumHists = np.sum([sumHists, np.bincount(cellBin.ravel(), cellMag.ravel(), bin_n)], axis=0)

                sumHists = np.sum(sumHists)
                for hist in tempHists:
                    histograms.append(np.divide(hist, np.sqrt(np.add(np.square(sumHists), np.square(epsilon)))))

                bin_blocks.append(bin_block)
                mag_blocks.append(mag_block)
        hist = np.hstack(histograms)
        return hist

    # ------------------------------------------------------------------------------
    # ---------------------------------OpenCV HOG-----------------------------------
    # ------------------------------------------------------------------------------

    def hog_opencv(self, image):
        image = img_as_float(image)  # convert unit8 tofloat64 ... dtype
        orientations = 9
        cellSize = (4, 4)  # pixels_per_cell
        blockSize = (3, 3)  # cells_per_block
        blockNorm = 'L1-sqrt'  # {'L1', 'L1-sqrt', 'L2', 'L2-Hys'}
        visualize = True  # Also return an image of the HOG.
        transformSqrt = False
        featureVector = True
        fd, hog_image = hog(image, orientations, cellSize, blockSize, blockNorm, visualize, transformSqrt,
                            featureVector)
        # cv2.imwrite('hog.jpg', hog_image)
        # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))  
        # Rescale histogram for better display
        return fd

    def main(self):
        image = cv2.imread('Files/lense.png')
        #img1 = cv2.resize(image, (4, 8))
        #img2 = cv2.resize(image, (4, 8))
        #roi = np.concatenate((img1, img2), axis=1)
        #scaled = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
	scaled, image = self.viola_jones(image)
        print(self.calculate_myhog(scaled).shape)
	fd = self.hog_opencv(scaled)
	print(fd.shape)


if __name__ == '__main__': FeatureExtraction().main()
