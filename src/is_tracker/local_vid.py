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
#from pyimagesearch.centroidtracker import CentroidTracker

from utility import load_options, get_np_image, get_rects, to_object_annotations, to_image
from tracker import Tracker, TrackedObject

config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

cap = cv2.VideoCapture('road.mp4')

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
    _,img = cap.read()
    print((img.shape[1],img.shape[0]))
    tracker = Tracker((img.shape[1],img.shape[0]))

    while True:
        
        t1 = time.time()
        _,img = cap.read()
        
        t2 = time.time()
        
        img = get_np_image(img)
        img_to_draw = img
        
        img = tf.expand_dims(img, 0)
        img = transform_images(img, trackerOptions.YOLO.size)

        t3 = time.time()
        detections_list = yolo.predict(img)
        tracker.update(detections_list)
        t4 = time.time()

        # for i in range(nums[0]):
            # print('\t{}, {}, {}'.format(class_names[int(classes[0][i])], np.array(scores[0][i]),np.array(boxes[0][i])))
        boxes, scores, classes, nums = detections_list
        
        t5 = time.time()
        
        img_to_draw = draw_outputs(img_to_draw, (boxes, scores, classes, nums), class_names)
        
        for (object_ID, obj) in tracker.tracked_objects.items():
            text = "{}".format(object_ID)
            cv2.putText(img_to_draw, text, (int(obj.x[0]), int(obj.x[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 240, 0), 4)
            if not obj.missing:
                cv2.putText(img_to_draw, text, (int(obj.z[0]), int(obj.z[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (240, 0, 0), 4)
        
        t6 = time.time()
        
        msg_time = 'consume {:.4f}'.format(t2 - t1)
        yolo_time = 'yolo {:.4f}'.format(t4 - t3)
        tracker_time = 'tracker {:.4f}'.format(t5 - t4)
        frame_time = 'total {:.4f}'.format(t6 - t1)
        
        cv2.putText(img_to_draw, msg_time, (40,40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (10, 10, 10), 2)
        cv2.putText(img_to_draw, yolo_time, (40,60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (10, 10, 10), 2)
        cv2.putText(img_to_draw, tracker_time, (40,80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (10, 10, 10), 2)
        cv2.putText(img_to_draw, frame_time, (40,100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (10, 10, 10), 2)
        cv2.putText(img_to_draw, 'X', (40,120), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 240, 0), 2)
        cv2.putText(img_to_draw, 'Z', (40,140), cv2.FONT_HERSHEY_COMPLEX, 0.8, (240, 0, 0), 2)
        
        # Publica imagem ##
        yolo_rendered = Message()
        yolo_rendered.pack(to_image(img_to_draw))
        channel.publish(yolo_rendered, 'Tracker.'+str(trackerOptions.camera_id)+'.Frame')
        
        ##
        # time.sleep(0.1)
        
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        print('Saiu aqui')
