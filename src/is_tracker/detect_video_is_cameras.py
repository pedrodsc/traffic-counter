# TODO 
# 
# 1. Criar um "10.10.2.1:30300/Tracker.1.BBox" com todos os objetos detectados
# 2. Trocar o centroidTracker para o Tracker com filtro de Kalman
# 3.Editar o Yolo para retornar um detectedObjects mais legivel do que o atual Yolo.predict()
#   possivelmente criando uma função mãe para chamar (Yolo.detect()??)
# 4. Trocar os nomes das instancias de Yolo para detector e tracker onde for aplicavel
# Leia a linha 103

import time
import cv2
import tensorflow as tf
import numpy as np

from is_wire.core import Channel, Subscription, Message
from is_msgs.image_pb2 import ObjectAnnotations, Image
from pprint import pprint

from yolov3_tf2.models import (YoloV3, YoloV3Tiny)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
from pyimagesearch.centroidtracker import CentroidTracker

from utility import load_options, get_np_image, get_rects, to_object_annotations, to_image

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
    print("broker: {}".format(broker))
    channel = Channel(broker)
    
    subscription = Subscription(channel)

    topic = "CameraGateway."+str(trackerOptions.camera_id)+".Frame"
    print("topic: {}".format(topic))
    
    subscription.subscribe(topic=topic)
    print("subscribed")
    
    while True:
        
        t1 = time.time()
        msg = channel.consume()
        print("msg consumed")
        img = msg.unpack(Image)
        
        t2 = time.time()
        
        img = get_np_image(img)
        img_to_draw = img
        
        img = tf.expand_dims(img, 0)
        img = transform_images(img, trackerOptions.YOLO.size)

        t3 = time.time()
        boxes, scores, classes, nums = yolo.predict(img)
        t4 = time.time()

       # for i in range(nums[0]):
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
        
        # TODO
        # Publica BBox - protótipo
        
        #yolo_bbox_msg = Message()
        #yolo_bbox_msg.pack(object annotation)
        # O object annotation contem o nome da classe, confiança, x1,y1,x2,y2
        # 
        # ObjectAnnotations <= ObjectAnnotation {
                                #   label <= classe
                                #   id <= id dado pelo tracker
                                #   score <= confiança
                                #   region.BoundingPoly <= x1,y1,x2,y2
                                #   keypoints <= ?? Centroid? }
        # resolution.height <= img.height
        # resolution.width <= img.width
        # frame_id <= numero do frame atual desde o inicio do programa
        #             associar isso de alguma forma com o msg.created_at
        #             por enquanto.
        # 
        # Olhar o object_anottation em https://github.com/labviros/is-skeletons-detector/blob/master/src/is_skeletons_detector/skeletons.py
        
        #channel.publish(yolo_bbox_msg, 'Tracker.'+str(trackerOptions.camera_id)+'.BBox')
        
        ##
        
        # Publica imagem ##
        yolo_rendered = Message()
        yolo_rendered.pack(to_image(img_to_draw))
        channel.publish(yolo_rendered, 'Tracker.'+str(trackerOptions.camera_id)+'.Frame')
        
        ##
        
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
