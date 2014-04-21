import math
import os
import copy
import numpy as np
import ctypes
import cv2
from SimpleCV import Image, Color
from Ellipse import Ellipse

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
from cython_functions import countInWindowEst, generateFlattenedDiskImg, doLineTest, normalizeImageStart

class Galaxy(object):
    def __init__(self, imgPath):
        self.id = (os.path.split(imgPath)[1]).split('.')[0]

        self.imagePath = imgPath
        self.initSuccess = True

        self._image = None
        self._flattenedDiskImage = None
        self._bulgeBlob = None
        self._imageWithoutBackground = None
        self._rotLineTestResult = None
        self._rotBoxTest = None

        #Derived properties
        self.ellipse = self._detectMainEllipse()

        #Store observations about this particular galaxy in the following dictionary
        self.observations = {}

    def getAspectRatio(self):
        if max(self.ellipse.b, self.ellipse.a) == 0:
            return 0

        result = min(self.ellipse.a, self.ellipse.b) / max(self.ellipse.b, self.ellipse.a)
        if math.isnan(result):
            result = 0
        return result

    def getFeatureEllipses(self):
        gray = cv2.cvtColor(self.flattenedDiskImg.getNumpyCv2(), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ellipses = []
        largeContours = []
        for cnt in contours:
            arcLen = cv2.arcLength(cnt, False)
            if arcLen > 20.0:
                largeContours.append(cnt)
                ellipses.append(cv2.fitEllipse(cnt))

        return ellipses

    def getBulgeImage(self):
        if self.bulgeBlob:
            return self.image.applyBinaryMask(self.bulgeBlob.image)
        else:
            return None

    #Property Methods
    def getBulgeBlob(self):
        if not self._bulgeBlob:
            bulgeBlobs = self._findGalaxyBulgeBlobs()
            if bulgeBlobs:
                self._bulgeBlob = bulgeBlobs[0]

        return self._bulgeBlob

    def getDiskImage(self):
        if self.bulgeBlob:
            return self.imageWithoutBackground - self.getBulgeImage()
        else:
            return self.imageWithoutBackground

    def getImage(self):
        if not self._image:
            self._image = Image(self.imagePath)

        return self._image

    def getImageWithoutBackground(self):
        if not self._imageWithoutBackground:
            self._imageWithoutBackground = self._removeAllButCentralGalaxyCluster()

        return self._imageWithoutBackground

    def getFlattenedDiskImg(self):
        if not self._flattenedDiskImage:

            grayImg = self.image.toGray()
            diff = grayImg - self._boxFilter(grayImg, 35)

            if not self.ellipse.error:
                res = countInWindowEst(diff.threshold(6).getGrayNumpy(), 5)
                res = self._getRotatedAndScaledGalImg(Image(res))

                if res != None:
                    res = res.getGrayNumpy()
                    res = generateFlattenedDiskImg(res, int(res.shape[0]/2), int(res.shape[0]/2), res.shape[0]/2)

                    if res != None:
                        res = normalizeImageStart(res)
                        self._flattenedDiskImage = Image(res)

        return self._flattenedDiskImage

    def getRotateBoxTest(self):
        if not self._rotBoxTest:

            grayImg = self.image.toGray()
            diff = grayImg - self._boxFilter(grayImg, 35)

            result = [0]*18

            if not self.ellipse.error and self.ellipse.a > 0 and self.ellipse.b > 0:
                res = countInWindowEst(diff.threshold(6).getGrayNumpy(), 5)
                res = self._getRotatedAndScaledGalImg(Image(res))

                if res != None:
                    res = res.getGrayNumpy()

                    img = Image(res)
                    ellipse = self.ellipse
                    center = (img.width/2, img.height/2)
                    width = 10
                    diameter = int(max(ellipse.a, ellipse.b)*2)
                    mask = np.zeros_like(img.getGrayNumpy(), dtype=np.bool)
                    mask[center[0]-width/2:center[0]+width/2, center[1]-diameter/2:center[1]+diameter/2] = True
                    mask = Image(mask).threshold(0)

                    maxI = 0
                    maxThetaVal = 0

                    values = []

                    for i, theta in enumerate(xrange(-90, 90, 10)):
                        masked = (img & mask.rotate(theta, point=(img.width/2, img.height/2))).getGrayNumpy()
                        val = np.sum(masked)/float(img.width*img.height)
                        values.append(val)

                        if val > maxThetaVal:
                            maxThetaVal = val
                            maxI = i

                    result[0:len(values)-maxI] = values[maxI:]
                    result[len(values)-maxI:] = values[0:maxI]

            self._rotBoxTest = result

        return self._rotBoxTest

    def getRotLineTestResults(self):
        if self._rotLineTestResult == None and self.flattenedDiskImg:
            self._rotLineTestResult = doLineTest(self.flattenedDiskImg.getGrayNumpy())

        return self._rotLineTestResult

    def getRotLineTestDerivative(self):
        if self.rotLineTestResult != None:
            result = [float(self.rotLineTestResult[i]) - float(self.rotLineTestResult[i+1]) for i in xrange(len(self.rotLineTestResult)-1)]
            return np.array(result, dtype=np.float)
        return None

    def getRotBias(self):
        result = 0

        if self.rotLineTestResult != None:
            leftBias = sum(self.rotLineTestResult[:len(self.rotLineTestResult)/2])
            rightBias = sum(self.rotLineTestResult[-len(self.rotLineTestResult)/2:])

            result = float(rightBias - leftBias)/len(self.rotLineTestResult)

        return result

    image = property(getImage)
    bulgeBlob = property(getBulgeBlob)
    imageWithoutBackground = property(getImageWithoutBackground)
    flattenedDiskImg = property(getFlattenedDiskImg)
    rotLineTestResult = property(getRotLineTestResults)
    rotLineTestDerivative = property(getRotLineTestDerivative)
    rotBias = property(getRotBias)
    rotBoxTestResult = property(getRotateBoxTest)

    #Private Methods
    def _getRotatedAndScaledGalImg(self, img):
        def rotateImgToPrimaryAxis(ellipse, img):
            maxAxis = max(ellipse.a, ellipse.b)
            minAxis = min(ellipse.a, ellipse.b)

            isBMax = ellipse.b == maxAxis

            if isBMax:
                result = img.rotate(ellipse.angle*(180.0/np.pi)+90.0)
            else:
                result = img.rotate(ellipse.angle*(180.0/np.pi))

            return result

        def getScaledImage(ellipse, img):
            result = None

            maxAxis = max(ellipse.a, ellipse.b)
            minAxis = min(ellipse.a, ellipse.b)

            if ellipse.ellipse_center()[0] - maxAxis > 0 and \
               ellipse.ellipse_center()[1] - minAxis > 0 and \
               maxAxis*2 < img.width and maxAxis*2 < img.height:

                result = img.crop(ellipse.ellipse_center()[0] - maxAxis,
                                  ellipse.ellipse_center()[1] - minAxis,
                                  maxAxis*2,
                                  minAxis*2)
                if result:
                    result = result.scale(width=int(maxAxis*2), height=int(maxAxis*2))

            return result

        rotImg = rotateImgToPrimaryAxis(self.ellipse, img)
        return getScaledImage(self.ellipse, rotImg)

    def _detectMainEllipse(self):
        minDist = 100000
        mainBlob = None

        for blob in self.image.dilate(2).findBlobs(minsize=200):
            dist = math.sqrt((blob.x - self.image.width/2)**2 + (blob.y - self.image.height/2)**2)
            if dist < minDist:
                minDist = dist
                mainBlob = blob

        blob = mainBlob

        ellipse = Ellipse()
        ellipse.fitToData(np.array(blob.hull()))

        return ellipse

    def _removeAllButCentralGalaxyCluster(self):
        e = self.ellipse
        img = self.image

        emask = Image(np.zeros((img.width, img.height), dtype=np.uint8))

        if e and e.a and e.b and e.a != np.nan and e.b != np.nan:
            try:
                e.drawOntoLayer(emask)
            except:
                print "Got exception while processing %s" % self.id
                pass

        emask = emask.applyLayers().floodFill((img.width/2, img.height/2), color=Color.BLUE)
        mask = emask.binarize().invert()
        return img.applyBinaryMask(mask)

    def _findGalaxyBulgeBlobs(galaxy):
        img = galaxy.imageWithoutBackground
        graynp = img.getGrayNumpy()

        stretched = np.array(graynp, dtype=np.float32)*(255.0/graynp.max())
        stretched = Image(stretched)
        return stretched.threshold(220).binarize().invert().findBlobs(minsize=4)

    def _boxFilter(self, img, dim):
        C = 1.0/dim**2
        kernel = [[C for k in xrange(dim)] for i in xrange(dim)]
        return img.convolve(kernel)

    def _rotMatrix(self, theta):
        return np.matrix([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def __getstate__(self):
        d = copy.copy(self.__dict__)

        d['_image'] = None
        d['_bulgeBlob'] = None
        d['_imageWithoutBackground'] = None

        return d








