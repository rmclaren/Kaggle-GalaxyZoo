from ClassBase import ClassBase

import matplotlib.pyplot as plt

from config import TRAINING_SOLUTIONS_PATH, TRAINING_IMAGE_PATH
from Solution import *
from GalaxyFactory import GalaxyFactory

from sklearn.ensemble import RandomForestRegressor

class Class5(ClassBase):

    def __init__(self):
        super(Class5, self).__init__()
        self.model1 = None
        self.model2 = None
        self.model3 = None
        self.model4 = None

    def predict(self, galaxy):
        features = self._extractFeatures(galaxy)
        if not features or galaxy.ellipse.ellipse_area() == 0:
            return self._defaultPrediction()

        featureArray = np.array([features['bulgeIntensity'],
                                 features['R'] - features['G'],
                                 features['bulgeBlobFractionalArea']], dtype=np.float32)

        model1Result = self.model1.predict(featureArray)[0]
        model2Result = self.model2.predict(featureArray)[0]
        model3Result = self.model3.predict(featureArray)[0]
        model4Result = self.model4.predict(featureArray)[0]

        return [model1Result, model2Result, model3Result, model4Result]

    def _defaultPrediction(self):
        return [0.09370659373687916, 0.4140570329904807, 0.40817431565229556, 0.08406205762034447]

    def train(self, verbose=False, training_data=None):
        if not training_data:
            trainingDataDict = self._getTrainingData(numSamples=5000)
        else:
            trainingDataDict = training_data

        X = np.array(trainingDataDict['training_data'], dtype=np.float32)

        y = np.array(trainingDataDict['solution_data'][0], dtype=np.float32)
        self.model1 = RandomForestRegressor(max_depth=50)
        self.model1 = self.model1.fit(X, y)

        y = np.array(trainingDataDict['solution_data'][1], dtype=np.float32)
        self.model2 = RandomForestRegressor(max_depth=50)
        self.model2 = self.model2.fit(X, y)

        y = np.array(trainingDataDict['solution_data'][2], dtype=np.float32)
        self.model3 = RandomForestRegressor(max_depth=50)
        self.model3 = self.model3.fit(X, y)

        y = np.array(trainingDataDict['solution_data'][3], dtype=np.float32)
        self.model4 = RandomForestRegressor(max_depth=50)
        self.model4 = self.model4.fit(X, y)

        self.areModelsTrained = True

    def _getTrainingData(self, startPos=0, numSamples=5000, verbose=False):
        galaxyFactory = GalaxyFactory(isTrainingData=True)
        sr = SolutionReader(TRAINING_SOLUTIONS_PATH)
        sr.skipTo(startPos)

        galaxyIds = []
        solution_data = [[], [], [], []]
        training_data = []
        success = []

        galaxyIndex = 0

        trainingSol = sr.next()
        while trainingSol:
            galaxy = galaxyFactory.getGalaxyForImage(os.path.join(TRAINING_IMAGE_PATH,
                                                                  str(trainingSol.GalaxyID) + '.jpg'))

            if galaxy.initSuccess:
                if galaxy.ellipse.ellipse_area() > 0:
                    features = self._extractFeatures(galaxy, useCache=True)

                    if features:
                        training_data.append([features['bulgeIntensity'],
                                              features['R'] - features['G'],
                                              features['bulgeBlobFractionalArea']])

                        solution_data[0].append(trainingSol.Class5_1)
                        solution_data[1].append(trainingSol.Class5_2)
                        solution_data[2].append(trainingSol.Class5_3)
                        solution_data[3].append(trainingSol.Class5_4)

                        galaxyIds.append(trainingSol.GalaxyID)
                        success.append(True)

                        if verbose and not galaxyIndex % 10:
                            print galaxyIndex

                        galaxyIndex += 1
                        if galaxyIndex > numSamples:
                            break

            else:
                galaxyIds.append(trainingSol.GalaxyID)
                success.append(False)

            trainingSol = sr.next()

        return {'galaxyIds': galaxyIds, 'training_data': training_data, 'solution_data': solution_data}


    def plot(self):
        print 'Plotting Class 5'

        trainingData = self._getTrainingData(numSamples=1000)

        if not self.areModelsTrained:
            self.train(verbose=True, training_data=trainingData)

        trainingData = self._getTrainingData(startPos=1000, numSamples=1000)

        plotRange = range(len(trainingData['training_data']))
        model1Results = [self.model1.predict(np.array(trainingData['training_data'][i], dtype=np.float32)) for i in plotRange]
        model1Results = [model1Results[i][0] for i in xrange(len(model1Results))]
        plt.plot(trainingData['solution_data'][0], model1Results, 'r+')

        model2Results = [self.model2.predict(np.array(trainingData['training_data'][i], dtype=np.float32)) for i in plotRange]
        model2Results = [model2Results[i][0] for i in xrange(len(model2Results))]
        plt.plot(trainingData['solution_data'][1], model2Results, 'g+')

        model3Results = [self.model3.predict(np.array(trainingData['training_data'][i], dtype=np.float32)) for i in plotRange]
        model3Results = [model3Results[i][0] for i in xrange(len(model3Results))]
        plt.plot(trainingData['solution_data'][2], model3Results, 'b+')

        model4Results = [self.model4.predict(np.array(trainingData['training_data'][i], dtype=np.float32)) for i in plotRange]
        model4Results = [model4Results[i][0] for i in xrange(len(model4Results))]
        plt.plot(trainingData['solution_data'][3], model4Results, 'm+')

        plt.title('Class 5')

        print '    Showing Class 5'

        plt.show()

