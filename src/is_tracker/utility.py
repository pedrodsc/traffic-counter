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
        print('Unable to open file \'{}\'. Opening default configuration.'.format(op_file))
        with open(/home/is-tracker/etc/conf/options.json, 'r') as f:
            try:
                op = Parse(f.read(), TrackerOptions())
                return op
            except Exception as ex:
                print('Unable to load options from \'{}\'. \n{}'.format(op_file, ex))

def get_np_image(input_image):
    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image

def to_image(input_image, encode_format='.jpeg', compression_level=0.8):
    if isinstance(input_image, np.ndarray):
        if encode_format == '.jpeg':
            params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * (100 - 0) + 0)]
        elif encode_format == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression_level * (9 - 0) + 0)]
        else:
            return Image()
        cimage = cv2.imencode(ext=encode_format, img=input_image, params=params)
        return Image(data=cimage[1].tobytes())
    elif isinstance(input_image, Image):
        return input_image
    else:
        return Image()

def get_rects(img_shape, boxes):
    boxes = boxes[0]
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

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 1)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    return img