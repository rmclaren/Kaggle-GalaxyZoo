from ClassBase import ClassBase

import matplotlib.pyplot as plt

from config import TRAINING_SOLUTIONS_PATH, TRAINING_IMAGE_PATH
from Solution import *
from GalaxyFactory import GalaxyFactory

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

class Class4(ClassBase):

    def __init__(self):
        super(Class4, self).__init__()
        self.dtr0 = None
        self.dtr1 = None
        self.str0 = None
        self.str1 = None
        self.ftr0 = None
        self.ftr1 = None

    def predict(self, galaxy):
        if galaxy.rotLineTestResult != None:
            features = self._extractFeatures(galaxy)

            if features:
                dtr0Pred = self.dtr0.predict(features['rotLineTestDerivative'])
                dtr1Pred = self.dtr1.predict(features['rotLineTestDerivative'])
                str0Pred = self.str0.predict(features['scaledImg'])
                str1Pred = self.str1.predict(features['scaledImg'])

                return [self.ftr0.predict([dtr0Pred[0], str0Pred[0]]),
                        self.ftr1.predict([dtr1Pred[0], str1Pred[0]])]

        return self._defaultPrediction()

    def _defaultPrediction(self):
        return [0.4947287273461465, 0.5052712726538535]

    def train(self, verbose=False, training_data=None):
        n_estimators = 50
        n_samples = 5000

        trainingDataDict = self._getTrainingData(numSamples=n_samples)

        X = np.array(trainingDataDict['rot_line_test_deriv'], dtype=np.float32)
        y = np.array(trainingDataDict['solution_data'][0], dtype=np.float32)
        dtr0 = ExtraTreesRegressor(n_estimators=n_estimators)
        dtr0 = dtr0.fit(X, y)

        X = np.array(trainingDataDict['rot_line_test_deriv'], dtype=np.float32)
        y = np.array(trainingDataDict['solution_data'][1], dtype=np.float32)
        dtr1 = ExtraTreesRegressor(n_estimators=n_estimators)
        dtr1 = dtr1.fit(X, y)

        X = np.array(trainingDataDict['scaled_img'], dtype=np.float32)
        y = np.array(trainingDataDict['solution_data'][0], dtype=np.float32)
        str0 = ExtraTreesRegressor(n_estimators=n_estimators)
        str0 = str0.fit(X, y)

        X = np.array(trainingDataDict['scaled_img'], dtype=np.float32)
        y = np.array(trainingDataDict['solution_data'][1], dtype=np.float32)
        str1 = ExtraTreesRegressor(n_estimators=n_estimators)
        str1 = str1.fit(X, y)


        trainingDataDict = self._getTrainingData(startPos=n_samples+1, numSamples=n_samples)

        dtr0Pred = [dtr0.predict(trainingDataDict['rot_line_test_deriv'][i]) for i in range(len(trainingDataDict['rot_line_test_deriv']))]
        dtr1Pred = [dtr1.predict(trainingDataDict['rot_line_test_deriv'][i]) for i in range(len(trainingDataDict['rot_line_test_deriv']))]
        str0Pred = [str0.predict(trainingDataDict['scaled_img'][i]) for i in range(len(trainingDataDict['scaled_img']))]
        str1Pred = [str1.predict(trainingDataDict['scaled_img'][i]) for i in range(len(trainingDataDict['scaled_img']))]

        X = np.array([[dtr0Pred[i][0], str0Pred[i][0]] for i in xrange(len(dtr0Pred))], dtype=np.float32)
        y = np.array(trainingDataDict['solution_data'][0], dtype=np.float32)
        ftr0 = ExtraTreesRegressor(n_estimators=n_estimators)
        ftr0 = ftr0.fit(X, y)

        X = np.array([(dtr1Pred[i][0], str1Pred[i][0]) for i in xrange(len(dtr1Pred))], dtype=np.float32)
        y = np.array(trainingDataDict['solution_data'][1], dtype=np.float32)
        ftr1 = ExtraTreesRegressor(n_estimators=n_estimators)
        ftr1 = ftr1.fit(X, y)

        self.dtr0 = dtr0
        self.dtr1 = dtr1
        self.str0 = str0
        self.str1 = str1
        self.ftr0 = ftr0
        self.ftr1 = ftr1

        self.areModelsTrained = True

    def _getTrainingData(self, startPos=0, numSamples=5000, verbose=False):
        galaxyFactory = GalaxyFactory(isTrainingData=True)
        sr = SolutionReader(TRAINING_SOLUTIONS_PATH)
        sr.skipTo(startPos)

        galaxyIds = []
        solution_data = [[], []]

        rot_line_test = []
        rot_line_test_deriv = []
        scaled_img = []
        rot_bias = []

        galaxyIndex = 0

        trainingSol = sr.next()
        while trainingSol:
            galaxy = galaxyFactory.getGalaxyForImage(os.path.join(TRAINING_IMAGE_PATH,
                                                                  str(trainingSol.GalaxyID) + '.jpg'))

            features = self._extractFeatures(galaxy, useCache=True)

            if features != None and features['rotLineTestResult'] != None:
                galaxyIds.append(galaxy.id)

                rot_line_test.append(features['rotLineTestResult'])
                rot_line_test_deriv.append(features['rotLineTestDerivative'])
                scaled_img.append(features['scaledImg'])
                rot_bias.append(features['rotBias'])

                solution_data[0].append(trainingSol.Class4_1)
                solution_data[1].append(trainingSol.Class4_2)

            if verbose and not galaxyIndex % 10:
                print galaxyIndex

            galaxyIndex += 1
            if galaxyIndex > numSamples:
                break

            trainingSol = sr.next()

        return {'galaxyIds': galaxyIds,
                'rot_line_test': rot_line_test,
                'rot_line_test_deriv': rot_line_test_deriv,
                'scaled_img': scaled_img,
                'rot_bias': rot_bias,
                'solution_data': solution_data}

    def plot(self):
        c = Class4()
        c.train()

        sr = SolutionReader(TRAINING_SOLUTIONS_PATH)
        sr.skipTo(15000)

        galaxyFactory = GalaxyFactory(isTrainingData=True)
        trainingSol = sr.next()
        galaxy = galaxyFactory.getGalaxyForImage(os.path.join(TRAINING_IMAGE_PATH, str(trainingSol.GalaxyID) + '.jpg'))

        print c.predict(galaxy)


