#The implementation of this class was derived from
#http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

import numpy as np
from numpy.linalg import eig, inv
import math

class Ellipse(object):

    def __init__(self):
        self.angle = None
        self.a = None
        self.b = None
        self.box = None
        self.parameters = None
        self.error = False

    def fitToData(self, data):
        '''
        param data: numpy array where [:,0] is x and [:,1] is y
        '''
        x = data[:, 0][:, np.newaxis]
        y = data[:, 1][:, np.newaxis]
        D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T, D)
        C = np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2; C[1, 1] = -1
        E, V = eig(np.dot(inv(S), C))
        n = np.argmax(np.abs(E))
        self.parameters = V[:, n]

        axes = self.ellipse_axis_length()
        self.a = axes[0]
        self.b = axes[1]
        self.angle = self.ellipse_angle_of_rotation()

        if not self.a or not self.b or self.parameters == None or np.iscomplexobj(self.parameters) or \
           math.isnan(self.a) or math.isnan(self.b) or math.isnan(self.ellipse_center()[0]) or \
           np.iscomplex(self.ellipse_center()[0]) or np.iscomplex(self.a) or np.iscomplex(self.b) or \
           np.iscomplexobj(self.angle):
            self.a = 0
            self.b = 0
            self.parameters = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.angle = 0
            self.error = True

    def ellipse_center(self):
        a = self.parameters
        b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        num = b*b-a*c
        if num != 0:
            x0=(c*d-b*f)/num
            y0=(a*f-b*d)/num
            return np.array([x0, y0])
        else:
            return np.array([0, 0])

    def ellipse_angle_of_rotation(self):
        a = self.parameters
        b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]

        return 0.5*np.arctan(2*b/(a-c))

    def ellipse_axis_length(self):
        a = self.parameters
        b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        down1 = (b*b-a*c)*((c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2 = (b*b-a*c)*((a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))

        if down1 == 0 or down2 == 0 or up/down1 < 0 or up/down2 < 0:
            res1 = 0
            res2 = 0
            self.error = 0
        else:
            res1 = np.sqrt(up/down1)
            res2 = np.sqrt(up/down2)

        return np.array([res1, res2])

    def ellipse_area(self):
        return np.pi * self.a * self.b

    def drawOntoLayer(self, img):
        R = np.arange(0, 2*np.pi, 0.01)
        center = self.ellipse_center()
        phi = self.ellipse_angle_of_rotation()

        a, b = (self.a, self.b)
        xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
        yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
        img.drawPoints([(xx[i], yy[i]) for i in xrange(len(xx))], sz=2)

    def getMask(self, img):
        mask = np.zeros((img.width, img.height), dtype=np.bool)

        R = np.arange(0, 2*np.pi, 0.01)
        center = self.ellipse_center()
        phi = self.ellipse_angle_of_rotation()
        axes = self.ellipse_axis_length()

        a, b = axes
        xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
        yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)

        mask[[(xx[i], yy[i]) for i in xrange(len(xx))]] = True

        return


