# TODO 
# 
# 1. Criar um '10.10.2.1:30300/Tracker.1.BBox' com todos os objetos detectados
# 2. Editar o Yolo para retornar um detectedObjects mais legivel do que o atual Yolo.predict()
#   possivelmente criando uma função mãe para chamar (Yolo.detect()??)
#  OBS: O ponto 2 é debatível visto que agora eu estou usando o tf.saved_model.load e infer()
# Leia a linha 126

import time
import cv2
import tensorflow as tf
import numpy as np
from pprint import pprint

from is_wire.core import Channel, Subscription, Message
from is_msgs.image_pb2 import ObjectAnnotations, Image

from yolov3_tf2.models import (YoloV3, YoloV3Tiny)
from yolov3_tf2.dataset import transform_images

from utility import (load_options, get_np_image, get_rects, to_object_annotations,
    to_image, draw_outputs)
from tracker import Tracker, TrackedObject 

config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

saved_model = './is-tracker/src/serving/yolov3/1'

def main():
    
    trackerOptions = load_options()
    pprint('#####################################################')
    pprint(trackerOptions)
    
    # if trackerOptions.YOLO.tiny:
    #     yolo = YoloV3Tiny()
    # else:
    #     yolo = YoloV3()
        
    # yolo.load_weights(trackerOptions.YOLO.weights)
    # print('weights loaded')

    class_names = [c.strip() for c in open(trackerOptions.YOLO.classes).readlines()]
    print('classes loaded')

    times = []

    broker = trackerOptions.broker
    print(f'broker: {broker}')
    channel = Channel(broker)
    
    subscription = Subscription(channel)

    topic = f'CameraGateway.{trackerOptions.camera_id}.Frame'
    print(f'topic: {topic}')
    
    subscription.subscribe(topic=topic)
    print('subscribed')
    
    msg = channel.consume()
    print('msg consumed')
    img = msg.unpack(Image)
    img = get_np_image(img)

    print((img.shape[1],img.shape[0]))
    tracker = Tracker((img.shape[1],img.shape[0]))

    model = tf.saved_model.load(saved_model)
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

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
        #detections_list = yolo.predict(img)
        
        outputs = infer(img)
        boxes, scores, classes, nums = outputs["yolo_nms_0"], outputs[
        "yolo_nms_1_1"], outputs["yolo_nms_2_2"], outputs["yolo_nms_3_3"]

        
        detections_list = [boxes, scores, classes, nums]

        t4 = time.time()

        tracker.update(detections_list)
        t5 = time.time()
        
        img_to_draw = draw_outputs(img_to_draw, (boxes, scores, classes, nums), class_names)
        
        for (object_ID, obj) in tracker.tracked_objects.items():
            text = "{}".format(object_ID)
            cv2.putText(img_to_draw, text, (int(obj.x[0]), int(obj.x[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 240, 0), 4)
            if not obj.missing:
                cv2.putText(img_to_draw, text, (int(obj.z[0]), int(obj.z[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (240, 0, 0), 4)


        msg_time = 'consume {:.4f}'.format(t2 - t1)
        yolo_time = 'yolo {:.4f}'.format(t4 - t3)
        tracker_time = 'tracker {:.4f}'.format(t5 - t4)
        
        
        
        cv2.putText(img_to_draw, msg_time, (40,40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (10, 10, 10), 1)
        cv2.putText(img_to_draw, yolo_time, (40,60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (10, 10, 10), 1)
        cv2.putText(img_to_draw, tracker_time, (40,80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (10, 10, 10), 1)
        
        t6 = time.time()
        # O tempo de desenhar o 'draw_time' e o 'frame_time' não são contabilizados pq sim
        draw_time = 'draw {:.4f}'.format(t6 - t5)
        frame_time = 'total {:.4f}'.format(t6 - t1)
        cv2.putText(img_to_draw, draw_time, (40,100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (10, 10, 10), 1)
        cv2.putText(img_to_draw, frame_time, (40,120), cv2.FONT_HERSHEY_COMPLEX, 0.8, (10, 10, 10), 1)

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
        channel.publish(yolo_rendered, f'Tracker.{trackerOptions.camera_id}.Frame')
        
        t7 = time.time()
        print(f'Loop time: {(t7-t1)*1000:.1f}ms')
        
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
