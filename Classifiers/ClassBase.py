import numpy as np
import pickle

class ClassBase(object):
    trainingFeaturesCache = {}

    def __init__(self):
        self.areModelsTrained = False

    def predict(self, galaxy):
        return self._defaultPrediction()

    def _defaultPrediction(self):
        raise Exception('Did not implement process galaxy in Class Idenitifier')

    def train(self, verbose=False, training_data=None):
        self.areModelsTrained = True

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    def load(self, path):
        pass

    def _extractFeatures(self, galaxy, useCache=False):

        if not galaxy.id in ClassBase.trainingFeaturesCache or not useCache:

            bulgeImg = galaxy.getBulgeImage()
            if not bulgeImg:
                return None

            diskImg = galaxy.getDiskImage()

            bulgeImgBrightness = bulgeImg.getGrayNumpy()
            bulgeBrightness = np.mean(bulgeImgBrightness[bulgeImgBrightness > 25])

            diskImgBrightness = diskImg.getGrayNumpy()
            diskBrightness = np.mean(diskImgBrightness[diskImgBrightness > 25])

            galaxyClusterNp = galaxy.image.toRGB().getNumpy()[:, :, 0]
            red = np.mean(galaxyClusterNp[galaxy.image.getGrayNumpy() > 25])

            galaxyClusterNp = galaxy.image.toRGB().getNumpy()[:, :, 1]
            green = np.mean(galaxyClusterNp[galaxy.image.getGrayNumpy() > 25])

            p = galaxy.ellipse.ellipse_center()
            p[0] -= 1
            p[1] -= 1

            symDiff = np.abs((galaxy.image.getGrayNumpy() - galaxy.image.rotate(180, point=p).getGrayNumpy()))
            asymscore = float(np.sum(symDiff)) / float(np.sum(galaxy.image.getGrayNumpy()))

            imgNoBackground = galaxy.imageWithoutBackground.getGrayNumpy()
            fractionForeground = len(imgNoBackground[imgNoBackground > 25])/galaxy.ellipse.ellipse_area() if galaxy.ellipse.ellipse_area() else 1.0

            scaledImg = None
            if galaxy.flattenedDiskImg:
                scaledImg = galaxy.flattenedDiskImg.scale(50, 12).getGrayNumpy()
                if galaxy.rotBias < 0:
                    scaledImg = scaledImg[::-1]

                scaledImg = scaledImg.reshape((scaledImg.shape[0]*scaledImg.shape[1]))

            rotLineTestResult = galaxy.rotLineTestResult if galaxy.rotBias >= 0 else galaxy.rotLineTestResult[::-1]
            rotLineTestDerivative = galaxy.rotLineTestDerivative if galaxy.rotBias >= 0 else galaxy.rotLineTestDerivative[::-1]


            features =  {'R': red,
                         'G': green,
                         'bulgeIntensity': 5 * np.log(bulgeBrightness/diskBrightness),
                         'bulgeBlobFractionalArea': galaxy.bulgeBlob.area()/galaxy.ellipse.ellipse_area() if galaxy.ellipse.ellipse_area() else 0,
                         'asymScore': asymscore,
                         'rotBias': abs(galaxy.rotBias),
                         'rotLineTestResult': rotLineTestResult,
                         'rotLineTestDerivative': rotLineTestDerivative,
                         'fractionForeground': fractionForeground,
                         'scaledImg': scaledImg,
                         'rotateBoxTestResult': None}

            ClassBase.trainingFeaturesCache[galaxy.id] = features

            return features

        return ClassBase.trainingFeaturesCache[galaxy.id]
