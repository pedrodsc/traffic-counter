import time
import cv2
import tensorflow as tf
import numpy as np
import threading
from pprint import pprint

from is_wire.core import Channel, Subscription, Message
from is_msgs.image_pb2 import ObjectAnnotations, Image

from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.dataset import transform_images

from utility import (load_options, get_np_image, get_rects, to_object_annotations,
    to_image, draw_outputs, draw_outputs2, draw_path, filter_classes, pack_and_publish, atomic_draw_and_publish)
from tracker import Tracker, TrackedObject 

class Is_Tracker(object):
    def __init__(self, Options):
        self.saved_model = './is-tracker/src/serving/yolov3/1'
        self.Options = Options
        pprint('#####################################################')
        pprint(self.Options)

        self.class_names = [c.strip() for c in open(self.Options.YOLO.classes).readlines()]
        print('classes loaded')
        self.broker = self.Options.broker
        print(f'broker: {self.broker}')
        self.channel_pub = Channel(self.broker)
        self.channel_sub = Channel(self.broker)
        self.subscription = Subscription(self.channel_sub)

        print(f'topic: {self.Options.consume_topic}')
        
        self.subscription.subscribe(topic=self.Options.consume_topic)
        print('subscribed')
        print('waiting for messages...')

        msg = self.channel_sub.consume()
        print('message consumed')

        img = msg.unpack(Image)
        img = get_np_image(img)

        print((img.shape[1],img.shape[0]))
        self.tracker = Tracker((img.shape[1],img.shape[0]),self.Options.TrackerOptions)
        self.filter_set = set(self.Options.TrackerOptions.filter)

        self.model = tf.saved_model.load(self.saved_model)
        self.infer = self.model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    def run(self):
        
        t1 = time.time() # t1

        msg = self.channel_sub.consume()
        img = msg.unpack(Image)
        
        t2 = time.time() # t2
        
        img = get_np_image(img)
        img_to_draw = img
        
        img = tf.expand_dims(img, 0)
        img = transform_images(img, self.Options.YOLO.size)
        #detections_list = yolo.predict(img)
        
        outputs = self.infer(img)
        boxes, scores, classes, nums = outputs["yolo_nms_0"][0], outputs[
        "yolo_nms_1_1"][0], outputs["yolo_nms_2_2"][0], outputs["yolo_nms_3_3"][0]
        boxes = np.asarray(boxes, dtype=float)
        scores = np.asarray(scores, dtype=float)
        classes = np.asarray(classes, dtype=int)
        nums = int(nums)
        detections_list = [boxes, scores, classes, nums]
        detections_list = filter_classes(detections_list,self.filter_set,self.class_names)
        
        t3 = time.time() #t3

        self.tracker.update(detections_list)

        t4 = time.time() # t4
        
        # img_to_draw = draw_outputs2(img_to_draw, self.tracker.tracked_objects.items(),
        #                 self.class_names, self.Options.DrawingParams)
        
        t5 = time.time() # t5
        
        # Publica imagem ##
        
        thread = threading.Thread(target=atomic_draw_and_publish,args=[img_to_draw, self.tracker.tracked_objects.items(),
                        self.class_names, self.Options.DrawingParams, self.channel_pub, self.Options.publish_topic])
        thread.start()
        #atomic_draw_and_publish()
        #pack_and_publish(img_to_draw, channel, Options.publish_topic)

        t6 = time.time() # t6

        print(f'Loop: {(t6-t1)*1000:.1f}ms\tCons: {(t2-t1)*1000:.1f}ms\tInfer: {(t3-t2)*1000:.1f}ms\tTrack: {(t4-t3)*1000:.1f}ms\t\
        Draw&Pub: {(t6-t5)*1000:.1f}ms')
        #Draw: {(t5-t4)*1000:.1f}ms\tPub: {(t6-t5)*1000:.1f}ms')