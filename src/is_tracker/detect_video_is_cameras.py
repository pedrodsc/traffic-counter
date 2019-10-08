# TODO Criar um Yolo.1.BBox com todos os objetos detectados
# Usar protobuf??

import time
import cv2
import tensorflow as tf
import numpy as np

from is_wire.core import Channel, Subscription, Message
from is_msgs.image_pb2 import Image
from pprint import pprint

from yolov3_tf2.models import (YoloV3, YoloV3Tiny)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
from pyimagesearch.centroidtracker import CentroidTracker
from image_tools import to_image
from utils import load_options, get_np_image, get_rects

config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

centroidTracker = CentroidTracker(20) # Descarta um objeto após 20 frames sem reaparecer

def main():
    
    trackerOptions = load_options()
    pprint('#####################################################')
    pprint(trackerOptions)
    
    if trackerOptions.YOLO.tiny:
        yolo = YoloV3Tiny()
    else:
        yolo = YoloV3()
        
    yolo.load_weights(trackerOptions.YOLO.weights)
    print('weights loaded')

    class_names = [c.strip() for c in open(trackerOptions.YOLO.classes).readlines()]
    print('classes loaded')

    times = []

    broker = trackerOptions.broker
    channel = Channel(broker)
    
    subscription = Subscription(channel)

    camera_frame = "CameraGateway."+str(trackerOptions.camera_id)+".Frame"
    subscription.subscribe(topic=camera_frame)
    
    while True:
        t1 = time.time()
        
        msg = channel.consume()
        img = msg.unpack(Image)
        
        t2 = time.time()
        
        img = get_np_image(img)
        img_to_draw = img
        
        img = tf.expand_dims(img, 0)
        img = transform_images(img, trackerOptions.YOLO.size)

        t3 = time.time()
        boxes, scores, classes, nums = yolo.predict(img)
        t4 = time.time()

        for i in range(nums[0]):
            #print('\t{}, {}, {}'.format(class_names[int(classes[0][i])], np.array(scores[0][i]),np.array(boxes[0][i])))
            
        rects = get_rects(img_to_draw.shape[0:2], (boxes, scores, classes, nums))
        
        objects = centroidTracker.update(rects)
        
        t5 = time.time()
        
        img_to_draw = draw_outputs(img_to_draw, (boxes, scores, classes, nums), class_names)
        
        for (objectID, centroid) in objects.items():
            text = "{}".format(objectID)
            cv2.putText(img_to_draw, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 240, 0), 4)
        
        t6 = time.time()
        
        msg_time = 'consume {:.4f}'.format(t2 - t1)
        yolo_time = 'yolo {:.4f}'.format(t4 - t3)
        tracker_time = 'tracker {:.4f}'.format(t5 - t4)
        frame_time = 'total {:.4f}'.format(t6 - t1)
        
        cv2.putText(img_to_draw, msg_time, (40,40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (10, 10, 10), 2)
        cv2.putText(img_to_draw, yolo_time, (40,60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (10, 10, 10), 2)
        cv2.putText(img_to_draw, tracker_time, (40,80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (10, 10, 10), 2)
        cv2.putText(img_to_draw, frame_time, (40,100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (10, 10, 10), 2)
        
        # Publica BBox - protótipo
        
        #yolo_bbox_msg = Message()
        #yolo_bbox_msg.pack(object annotation)
        # O object annotation contem o nome da classe, confiança, x1,y1,x2,y2 (0 <= x e y <= 1)
        # O Vetor de bbox
        
        # Olhar o object_anottation em https://github.com/labviros/is-skeletons-detector/blob/master/src/is_skeletons_detector/skeletons.py
        
        #channel.publish(yolo_bbox_msg, 'Yolo.'+str(trackerOptions.camera_id)+'.BBox')
        
        ##
        
        # Publica imagem ##
        yolo_rendered = Message()
        yolo_rendered.pack(to_image(img_to_draw))
        channel.publish(yolo_rendered, 'Yolo.'+str(trackerOptions.camera_id)+'.Frame')
        
        ##
        
        
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
