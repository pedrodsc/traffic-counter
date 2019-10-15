import numpy as np
from numpy.linalg import inv
from collections import OrderedDict

from utils import get_rects

class Kalman(object):
  def __init__(self,R,Q,dt):
    
    self.R = R # Measure noise
    self.Q = Q # Process noise
    
    # Matrix to extract Cx and Cy from vector x
    self.H = np.array([[1., 0, 0, 0, 0, 0],[0., 1., 0, 0, 0, 0]])
    # Model of the process
    self.A = np.array( [[1., 0, dt,  0, (dt**2)/2, 0],    # Cx = Cx + Vx*dt + 0.5*Ax*dt**2
                        [0, 1,  0, dt, 0,     (dt**2)/2], # Cy = Cy + Vy*dt + 0.5*Ay*dt**2
                        [0, 0,  1,  0, dt,    0],         # Vx = Vx + Ax*dt
                        [0, 0,  0,  1, 0,     dt],        # Vy = Vy + Ay*dt
                        [0, 0,  0,  0, 1,     0],         # Ax = Ax
                        [0, 0,  0,  0, 0,     1]])        # Ay = Ay

  def predict(self,trackedObject):
    
    trackedObject.x = self.A.dot(trackedObject.x) # (C.5)
    trackedObject.P = self.A.dot(trackedObject.P).dot(self.A.T) + self.Q # (C.6)
  
  def update(self,trackedObject,z):
    
    S = self.H.dot(trackedObject.P).dot(self.H.T) + self.R # (C.9)
    K = trackedObject.P.dot(self.H.T).dot(inv(S)) # (C.9)
    y = z - self.H.dot(trackedObject.x) # (C.10)
    trackedObject.x = trackedObject.x + K.dot(y) # (C.10)
    trackedObject.P = trackedObject.P - K.dot(self.H).dot(trackedObject.P) # (C.11)

class TrackedObject(object):
    def __init__(self,className,objectID,x,y):
        self.className = className
        self.objectID = objectID
        self.score = 0
        # Covariance matrix
        self.P = np.eye(6)
        self.x = np.array([[x,y,0,0,0,0]]).T

class Tracker(object):
    def __init__(self, maxMissingFrames = 20, img_shape):
        # Lista, vetor ou dicionário de objetos sendo trackeados
        self.trackedObjects = OrderedDict()
        # Numero de objetos já trackeados
        self.nextObjID = 1
        # Número de objetos sendo trackeados
        self.objectsUnderTracking = 0
        # Hiperparâmetros: Limite de frames para remover um objeto, incertezas*
        self.maxMissingFrames = maxMissingFrames
        # Não sei se as incertezas serão para cada objeto ou global
        
        # This matrix times bbox(0 to 1) gives the centroid of the bbox
        (w,h) = img_shape
        self.centroidMatrix = np.array([[w/2,0,w/2,0],[0,h/2,0,h/2]],dtype=np.float)
        pass
    
    def register(self, detectedObj):
        trackedObjects[self.nextObjID] = detectedObj
        self.nextObjID += 1
        self.objectsUnderTracking += 1
        pass
    
    def deresgister(self, objectID):
        trackedObjects.pop(objectID)
        self.objectsUnderTracking -= 1
        pass
    
    def dataAssociation(self, detectedObjects):
        # Associa os novos dados aos objetos antigos
        # Adiciona novos objetos
        
        # Parse detectedObjects
        boxes, score, classes, nums = detectedObjects
        boxes, score, classes, nums = boxes[0], score[0], classes[0], nums[0]
      # bbox, confidence, class name, number of detected objects
        
        # centroids in x,y format
        centroids = np.empty((nums,2))      
        
        for i in range(nums):
            box = boxes[i]
            centroids[i] = self.centroidMatrix.dot(box)
            
            #  |Cx|   |w/2  0  w/2 0  |     |bx1|
            #  |Cy| = | 0  h/2  0 h/2 |  *  |by1|
            #                               |bx2|
            #                               |by2|
        
        
        
        pass
    
    def update(self): ## Vai se chamar update?
        pass
        # Roda o update de todos os objetos detectados
    
