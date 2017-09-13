import cv2
import sys
import numpy as np
from skimage.feature import hog
from skimage import img_as_float
from scipy import signal


class myHog():
   
        # ------------------------------------------------------------------------------
        # ------------------------------My HOG: Gradients-----------------------------------
        # ------------------------------------------------------------------------------
        # figure out how to create actual hogs using mag and orien image
	#gradients work only on grayscale images

    def gx(self, scaled):
        y, x = scaled.shape
        gx = np.zeros(scaled.shape)
	#sobelX:
	#[1	,0	,-1]
        #[2	,0	,-2]
        #[1	,0	,-1]
              
        for i in range(y):
	    if( i-1 >=0 ) and (i+1 < x):
	    	a= np.convolve(scaled[i-1, :], [1,0, -1], 'valid')
	    	b= np.convolve(scaled[i, :], [2,0, -2], 'valid')
	    	c= np.convolve(scaled[i+1, :], [1,0, -1], 'valid')
		gx[i,1 :x-1] = np.sum([a,b,c], axis=0)
	    if( i-1 < 0 ):
	    	b= np.convolve(scaled[i, :], [-2,0, 2], 'valid')
	    	c= np.convolve(scaled[i+1, :], [-1,0, 1], 'valid')
		gx[i,1 :x-1] = np.sum([b,c], axis=0)
	    if(i+1 >= x):
	    	a= np.convolve(scaled[i-1, :], [-1,0, 1], 'valid')
	    	b= np.convolve(scaled[i, :], [-2,0, 2], 'valid')
		gx[i,1 :x-1] = np.sum([a,b], axis=0)
        return gx

    def gy(self, scaled):
	y, x = scaled.shape
        gy = np.zeros(scaled.shape)
	#sobelY:
	#[-1	,-2	,-1]
        #[0	,0	,0]
        #[1	,2	,1]
 
        for j in range(x):
	    if( j-1 >=0 ) and (j+1 < y):
	    	a= np.convolve(scaled[:, j-1], [1,0, -1], 'valid')
	    	b= np.convolve(scaled[:, j], [2,0, -2], 'valid')
	    	c= np.convolve(scaled[:, j+1], [1,0, -1], 'valid')
		gy[1:y-1, j] = np.sum([a,b,c], axis=0)
	    if( j-1 < 0 ):
	    	b= np.convolve(scaled[:, j], [-2,0, 2], 'valid')
	    	c= np.convolve(scaled[:, j+1], [-1,0, 1], 'valid')
		gy[1:y-1, j] = np.sum([b,c], axis=0)
	    if(j+1 >= y):
	    	a= np.convolve(scaled[:, j-1], [-1,0, 1], 'valid')
	    	b= np.convolve(scaled[:, j], [-2,0, 2], 'valid')
		gy[1:y-1, j] = np.sum([a,b], axis=0)
        return gy

    def magnitude(self, gx, gy):
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        return magnitude

    def orientation(self, gx, gy):
        #orientation = (np.arctan2(gy, gx) * 180 / np.pi) % 360
	orientation = (np.arctan2(gy, gx) /np.pi) % 2*np.pi #same output as cv2.cartToPolar
        return orientation

    def calculate_hog(self, scaled, cellsize, blocksize, binsize):
        gx = self.gx(scaled)
    	gy = self.gy(scaled)
	#print(gx,"gx mine")
	#print(gy,"gy mine")
	#gx2 = cv2.Sobel(scaled, cv2.CV_32F, 1, 0)
    	#gy2 = cv2.Sobel(scaled, cv2.CV_32F, 0, 1)
	#print(gx2,"gx stolen")
	#print(gy2,"gy stolen")

	magnitude = self.magnitude(gx, gy) 
	orientation = self.orientation(gx, gy)
	#print(magnitude,"mag mine")
	bin_n = binsize # Number of bins
    	bins = np.int32(bin_n*orientation/(2*np.pi))
	#print("bins mine", bins)

	bin_blocks = []
    	mag_blocks = []
	epsilon = sys.float_info.epsilon
	
    	cellx = celly = cellsize
	
	width = scaled.shape[0]	
	height = scaled.shape[1]
	
	y = ((height-(cellsize*blocksize))/cellsize) + 1
	x = ((width-(cellsize*blocksize))/cellsize) + 1 
	count = 0
	histograms = []
	for i in range(0, y ,1): #loops through each block in a image(i,j)
            	for j in range(0, x ,1):
		    bin_block = bins[i*cellsize : i*cellsize+blocksize*cellsize,
					j*cellsize : j*cellsize+blocksize*cellsize]
		    mag_block = magnitude[i*cellsize : i*cellsize+blocksize*cellsize,
					j*cellsize : j*cellsize+blocksize*cellsize]
		    tempHists = []
		    sumHists = np.zeros((9,))
		    for m in range(0, blocksize*cellsize,cellsize):	#loops through each cell in a block(m,n)
			for n in range(0, blocksize*cellsize,cellsize): 
			    cellBin =bin_block[m :m+cellsize,n :n+cellsize]
			    cellMag =mag_block[m :m+cellsize,n :n+cellsize]
		    	    tempHists.append(np.bincount(cellBin.ravel(), cellMag.ravel(), bin_n))
			    sumHists =  np.sum([sumHists, np.bincount(cellBin.ravel(), cellMag.ravel(), bin_n)], axis=0)
			    
		    sumHists = np.sum(sumHists)
		    for hist in tempHists:
			histograms.append(np.divide(hist,np.sqrt(np.add(np.square(sumHists),np.square(epsilon)))))
		    
            	    bin_blocks.append(bin_block)
		    mag_blocks.append(mag_block)
		    count = count +1

	#print(count, "number of blocks")
    	hist = np.hstack(histograms)
	#print(hist, "^_^")
	#print(hist)
	return hist

    def hog_opencv(self, image):

        image = img_as_float(image)  # convert unit8 tofloat64 ... dtype
        orientations = 9
        cellSize = (2, 2)  # pixels_per_cell
        blockSize = (2, 2)  # cells_per_block
        blockNorm = 'L1-sqrt'  # {'L1', 'L1-sqrt', 'L2', 'L2-Hys'}
        visualize = True  # Also return an image of the HOG.
        transformSqrt = False
        featureVector = True
        fd, hog_image = hog(image, orientations, cellSize, blockSize, blockNorm, visualize, transformSqrt,
                            featureVector)

        return fd
    def main(self):
	image = cv2.imread('lense.png')
	img1 = cv2.resize(image, (4, 8))
	img2 = cv2.resize(image, (4, 8))
	roi = np.concatenate((img1, img2), axis=1)
	scaled = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

	
	print(self.calculate_hog(scaled, 2, 2, 9).shape)
	print(self.hog_opencv(scaled).shape)
	print(self.hog_opencv(scaled))

	#print("=========================================")
	#cell = 2
	#block = 2
	#Y = ((height-(cell*block))/cell) + 1
	#X = ((width-(cell*block))/cell) + 1 
	#for i in range(0,Y,1):
        #    	for j in range(0,X ,1):
        #    	    x = scaled[i*cell : i*cell+block*cell,j*cell : j*cell+block*cell]
	#	    print(x)
   
#if __name__ == '__main__': myHog().main()



