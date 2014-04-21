
import os
import math
import pickle

from Galaxy import Galaxy
from config import TEST_GALAXY_PATH, TRAINING_GALAXY_PATH

class GalaxyFactory(object):
    '''
    '''

    def __init__(self, isTrainingData=False, recomputeAll=False):
        self.isTrainingData = isTrainingData
        self.recomputeAll = recomputeAll
        self.outputPath = TRAINING_GALAXY_PATH if self.isTrainingData else TEST_GALAXY_PATH

    def getGalaxyForImage(self, imagePath):
        # galFilePath = os.path.join(self.outputPath, galId + '.dat')

        # if galId + '.dat' in os.listdir(self.outputPath) and not self.recomputeAll:
        #     galaxy = pickle.load(open(galFilePath, 'rb'))
        # else:
        #     galaxy = Galaxy(galId, imagePath)
        #     galaxy.imagePath = imagePath
        #     pickle.dump(galaxy, open(galFilePath, 'wb'))

        return Galaxy(imagePath)



