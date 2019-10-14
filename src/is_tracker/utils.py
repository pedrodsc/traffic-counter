from google.protobuf.json_format import Parse

from options_pb2 import TrackerOptions
from is_msgs.image_pb2 import Image

import numpy as np
import cv2
import sys

def load_options():
    op_file = sys.argv[1] if len(sys.argv) > 1 else '/home/is-tracker/etc/conf/options.json' 
    try:
        with open(op_file, 'r') as f:
            try:
                op = Parse(f.read(), TrackerOptions())
                return op
            except Exception as ex:
                print('Unable to load options from \'{}\'. \n{}'.format(op_file, ex))
    except Exception as ex:
        print('Unable to open file \'{}\''.format(op_file))

def get_np_image(input_image):
    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image

def get_rects(img_shape, outputs):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img_shape)
    wh = np.append(wh,wh)
    rect = boxes * wh
    rect = rect.flatten()
    rect = np.trim_zeros(rect)
    rect = np.reshape(rect,(int(rect.shape[0]/4),4))
    
    return rect

def to_object_annotations(detectedObjects, im_width, im_height, frame_id):
        
        objAnno = ObjectAnnotations()
        
        for detObj in detectedObjects:
            obj = obsAnno.objects.add()
            #
            obj.label = detectedObjects.label
            obj.id = detectedObjects.id
            obj.score = detectedObjects.confidence
            bbox = obj.region.add()
            
            bboxTL = bbox.vertices.add()
            bboxTL.x = detObj.BBox.x1
            bboxTL.y = detObj.BBox.y1
            
            bboxBR = bbox.vertices.add()
            bboxBR.x = detObj.BBox.x2
            bboxBR.y = detObj.BBox.y2
            

        obs.resolution.width = im_width
        obs.resolution.height = im_height
        obs.frame_id = frame_id
        
        return obs
