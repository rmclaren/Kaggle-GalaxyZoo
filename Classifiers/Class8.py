from ClassBase import ClassBase
from GalaxyFactory import GalaxyFactory
from Solution import *
from config import TRAINING_SOLUTIONS_PATH, TRAINING_IMAGE_PATH

from sklearn.ensemble import RandomForestRegressor

class Class8(ClassBase):
    def __init__(self):
        super(Class8, self).__init__()

        self.ringModel = None


    def predict(self, galaxy):
        result = self._defaultPrediction()
        features = self._extractFeatures(galaxy)

        if features:
            blobs = galaxy._findGalaxyBulgeBlobs()
            if blobs and len(blobs) > 1:
                result[5] = 1.0

            if features['scaledImg'] != None:
                result[0] = self.ringModel.predict([features['scaledImg']])
        else:
            return self._defaultPrediction()

        return result

    def train(self, verbose=False, training_data=None):

        trainingDataDict = self._getTrainingData(numSamples=5000)

        X = np.array(trainingDataDict['scaledImg'], dtype=np.float32)
        y = np.array(trainingDataDict['solution_data'][0], dtype=np.float32)
        self.ringModel = RandomForestRegressor(max_depth=50)
        self.ringModel = self.ringModel.fit(X, y)

    def _getTrainingData(self, startPos=0, numSamples=5000, verbose=False):
        galaxyFactory = GalaxyFactory(isTrainingData=True)
        sr = SolutionReader(TRAINING_SOLUTIONS_PATH)
        sr.skipTo(startPos)

        galaxyIds = []
        solution_data = [[], [], [], [], [], [], []]

        rotateLineTestDeriv = []
        rotateBoxResult = []
        scaledImg = []

        galaxyIndex = 0

        trainingSol = sr.next()
        while trainingSol:
            galaxy = galaxyFactory.getGalaxyForImage(os.path.join(TRAINING_IMAGE_PATH,
                                                                  str(trainingSol.GalaxyID) + '.jpg'))

            features = self._extractFeatures(galaxy, useCache=True)

            if features != None and features['rotLineTestResult'] != None and features['scaledImg'] != None:
                galaxyIds.append(galaxy.id)
                rotateLineTestDeriv.append(features['rotLineTestDerivative'])
                scaledImg.append(features['scaledImg'])

                solution_data[0].append(trainingSol.Class8_1)
                solution_data[1].append(trainingSol.Class8_2)
                solution_data[2].append(trainingSol.Class8_3)
                solution_data[3].append(trainingSol.Class8_4)
                solution_data[4].append(trainingSol.Class8_5)
                solution_data[5].append(trainingSol.Class8_6)
                solution_data[6].append(trainingSol.Class8_7)

            galaxyIndex += 1
            if galaxyIndex > numSamples:
                break

            trainingSol = sr.next()

        return {'galaxyIds': galaxyIds,
                'rotateLineTestDeriv': rotateLineTestDeriv,
                'scaledImg': scaledImg,
                'solution_data': solution_data}



    def _defaultPrediction(self):
        return [0.1501786052312757, 0.05801758115317498, 0.1415283175699542, 0.1727477803548223, 0.27844547790811786,
                0.18250933577075995, 0.01657290201189505]