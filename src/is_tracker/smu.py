import cv2
from darkflow.net.build import TFNet
import numpy as np
from pyimagesearch.centroidtracker import CentroidTracker
import argparse
import time
import cv2
from pprint import pprint

def distance(c1,c2):
    d = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
    return d


class Coisa:
    def __init__ (self,c1,label,objId):
        self.position = c1
        self.label = label
        self.objId = objId
        self.vel = (0,0)
        self.miss = 0
        
    def setVel(vel):
        self.vel = vel
    
    def addMiss():
        self.miss = self.miss + 1
    def clearMisses():
        self.miss = 0

options = {
        'model': 'cfg/yolo.cfg',
        'load': 'bin/yolo.weights',
        'threshold': 0.4,
        'gpu': 0.7
}

tfnet = TFNet(options)

colors = [tuple(255*np.random.rand(3)) for i in range(60)]

radius = 20

capture = cv2.VideoCapture('/home/pedrovdsc/Vídeos/Blumenau1.mp4')

ct = CentroidTracker(40)

_,firstFrame = capture.read()
paths = np.zeros(firstFrame.shape,dtype=np.uint8)

actualObjectsInFrame = []
pastObjectsInFrame = []

showPath = True
showBox = True

'''

    Passos do algorítimo
    
    1.Obter as coordenadas dos objetos
    2.Comparar com as coordenadas de objetos anteriores
    3.Caso a distancia entre dois objetos seja menor que RAIO, considere os dois o mesmo objeto
    4.Se algum objeto não tiver par, adicioná-lo à lista.
    5.Caso algum objeto não apareça, adicionar 1 ao número de vezes que ele desaparece
    6.Caso o objeto apareça, zerar o contador
    7.Execultar uma rotina de limpeza. Caso algum objeto esteja sumido por X frames, apagá-lo


'''

while (capture.isOpened()):

    stime = time.time()

    _, frame = capture.read()
    result = tfnet.return_predict(frame)
    
    rects = []
    
    for detectedObject in result:
        
        label = detectedObject['label']
        
        if label == 'car':
        
            topLeft = (detectedObject['topleft']['x'],detectedObject['topleft']['y'])
            bottomRight = (detectedObject['bottomright']['x'],detectedObject['bottomright']['y'])
            
            rects.append((topLeft[0],topLeft[1],bottomRight[0],bottomRight[1]))
            
            center = (int((topLeft[0] + bottomRight[0])/2),int((topLeft[1] + bottomRight[1])/2))
            if showBox:
                cv2.putText(frame,'{:.2f}'.format(detectedObject['confidence']),(topLeft[0],topLeft[1]-5), cv2.FONT_HERSHEY_COMPLEX, 0.5,(200,0,200),1,cv2.LINE_AA)
                frame = cv2.rectangle(frame,topLeft,bottomRight,(0,200,255), 2)
            
            actualObjectsInFrame.append(center)

    
    
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "{}".format(objectID)
        cv2.putText(frame, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 240, 0), 2)
        #cv2.circle(frame, (centroid[0], centroid[1]), 3, (0, 255, 0), -1)

    if showPath:
        pathsGray = cv2.cvtColor(paths,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(pathsGray, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask)

        frame = cv2.bitwise_and(frame,frame,mask = mask)
        frame = cv2.add(frame,paths)
        
    # show the output frame
    
    key = cv2.waitKey(1) & 0xFF
    ttime = time.time() - stime
    cv2.putText(frame,'FPS {:.2f}'.format(1/ttime),(10,30), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(frame,'Total {}'.format(ct.nextObjectID),(10,60), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,0,255),1,cv2.LINE_AA)
    cv2.imshow('Imagem processada',frame)

    pprint(ct.objects)
    
    k =  cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('s'):
        showPath = not showPath
    elif k == ord('d'):
        showBox = not showBox
    else:
        continue
    
capture.release()
cv2.destroyAllWindows()
