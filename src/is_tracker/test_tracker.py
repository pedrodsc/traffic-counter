import cv2
import numpy as np
import time
from pprint import pprint

from is_wire.core import Channel, Subscription, Message
from is_msgs.image_pb2 import ObjectAnnotations, Image

from tracker import *
from utility import load_options, to_image

#####

class particle():
    def __init__(self, x, y):
        self.pos = np.array([x,y])
        self.vel = np.array([0.,0])
    
    def move(self):
        self.pos += self.vel
        self.vel += np.random.randn(2)

class particleGenerator():
    def __init__(self, population_size = 10, canvas = (800,600)):
        self.population = []
        for i in range(population_size):
            x = np.random.rand()*canvas[0]
            y = np.random.rand()*canvas[1]
            
            self.population.append(particle(x,y))
    def move(self):
        for p in self.population:
            p.move()
    
    def positions(self):
        return [p.pos for p in self.population]

#####

img_height = 600
img_width = 800

radius = 5
color = (0,0,255)

kalmanFilter = Kalman()
particles = particleGenerator()
trackedList = []
for idx, p in enumerate(particles.positions()):
    trackedList.append(TrackedObject('Dot',idx,p.pos[1],p.pos[2]))

def main():
    trackerOptions = load_options()
    broker = trackerOptions.broker
    print("broker: {}".format(broker))
    channel = Channel(broker)

    while True:
        img_to_draw = np.ones((img_height,img_width,3))*255
        # TODO 
        # get objects in new frame
        # predict
        # associate objects based on prediction
        # update kalman

        particles.move()
        
            
        tracker_rendered = Message()
        tracker_rendered.pack(to_image(img_to_draw))
        channel.publish(tracker_rendered, 'Tracker.'+str(trackerOptions.camera_id)+'.Teste')
        print("image sent")
        
        time.sleep(0.2)
                
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass

# for p in particles.population:
#             img_to_draw = cv2.circle(img_to_draw, (int(p.pos[0]),int(p.pos[1])), radius, color, -1)
#             kalmanFilter.predict(p)
#             kalmanFilter.update(p)
#             cv2.putText(img_to_draw, '+', (int(p.pos[0]),int(p.pos[1])), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)