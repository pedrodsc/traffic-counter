from google.protobuf.json_format import Parse

from options_pb2 import Options
from is_wire.core import Message
from is_msgs.image_pb2 import Image

import numpy as np
import cv2
import sys
from pprint import pprint

def load_options(options_file = '/opt/is-tracker/options.json'):
    try:
        with open(options_file, 'r') as f:
            try:
                op = Parse(f.read(), Options())
                return op
            except Exception as ex:
                print(f'Unable to load options from \'{options_file}\'. \n{ex}')
    except Exception as ex:
        print(f'Unable to open file \'{options_file}')
        print('Opening default configuration file.')
        default_op = '/home/is-tracker/etc/conf/options.json'
        with open(default_op, 'r') as f:
            try:
                op = Parse(f.read(), Options())
                return op
            except Exception as ex:
                print(f'Unable to load default configuration file.\'{default_op}\'. \n{ex}')
                print(f'Aborting')
                sys.exit()

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

def atomic_draw_and_publish(img, tracked_objects, class_names, DrawingParams, channel, topic):
    img = draw_outputs2(img, tracked_objects, class_names, DrawingParams)
    pack_and_publish(img, channel, topic)

def pack_and_publish(img, channel, topic):
    message = Message()
    message.pack(to_image(img))
    channel.publish(message,topic)

def draw_outputs(img, detected_objects, class_names, thickness = 1, score = True):
    boxes, objectness, classes, nums = detected_objects
    #boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), thickness)
        if score:
            img = cv2.putText(img, '{} {:.4f}'.format(
                class_names[int(classes[i])], objectness[i]),
                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        else:
            img = cv2.putText(img, '{}'.format(
                class_names[int(classes[i])]),
                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
    return img

def draw_outputs2(img, tracked_objects, class_names, DrawingParams):
    wh = np.flip(img.shape[0:2])

    show_boxes = DrawingParams.show_boxes
    show_paths = DrawingParams.show_paths
    show_ids = DrawingParams.show_ids
    show_names = DrawingParams.show_names
    show_scores  = DrawingParams.show_scores

    box_color = tuple(DrawingParams.box_color)
    missing_box_color = tuple(DrawingParams.missing_box_color)
    path_color = tuple(DrawingParams.path_color)
    id_color = tuple(DrawingParams.id_color)
    name_color = tuple(DrawingParams.name_color)

    box_thickness = DrawingParams.box_thickness
    path_thickness = DrawingParams.path_thickness
    id_size = DrawingParams.id_size
    name_size = DrawingParams.name_size

    for (object_ID, obj) in tracked_objects:
        x1y1 = tuple((obj.bbox[0:2]).astype(np.int32))
        x2y2 = tuple((obj.bbox[2:4]).astype(np.int32))
        
        # Draw BBox
        if show_boxes:
            if obj.missing:
                bbox_color = missing_box_color
            else:
                bbox_color = box_color
            img = cv2.rectangle(img, x1y1, x2y2, bbox_color, box_thickness)
        
        # Draw Path
        if show_paths:
            for i in range(len(obj.path)-1):
                x1 = (int(obj.path[i][0]),int(obj.path[i][1]))
                x2 = (int(obj.path[i+1][0]),int(obj.path[i+1][1]))
                img = cv2.line(img, x1, x2, path_color, path_thickness)
        # Draw Name
        if show_names:
            if show_scores:
                text = '{} {:.4f}'.format(class_names[obj.class_id], obj.score)
            else:
                text = '{}'.format(class_names[obj.class_id])
            img = cv2.putText(img, text,
                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, name_size, name_color, 2)
        # Draw id
        if show_ids:
           img = cv2.putText(img, str(object_ID), (int(obj.z[0]), int(obj.z[1])),
                cv2.FONT_HERSHEY_COMPLEX, id_size, id_color, 2)
    return img

def draw_path(img, tracked_objects, thickness = 1, color = (0,255,0)):
    
    for (object_ID, obj) in tracked_objects:
        for i in range(len(obj.path)-1):
            x1 = (int(obj.path[i][0]),int(obj.path[i][1]))
            x2 = (int(obj.path[i+1][0]),int(obj.path[i+1][1]))
            img = cv2.line(img, x1, x2, color, thickness)

    return img

def filter_classes(detected_objects, filter_set, class_names):
    # if nums == 0:
    #     return detected_objects

    filter_ids = []
    for x in filter_set:
        try:
            filter_ids.append(class_names.index(x))
        except ValueError:
            print(f'{x} is not a valid class.')
    filter_ids = set(filter_ids)
    
    boxes, objectness, classes, nums = detected_objects
    mask = [x in filter_ids for x in classes][:nums]


    boxes = boxes[:nums][mask]
    objectness = objectness[:nums][mask]
    classes = classes[:nums][mask]
    nums = mask.count(True)
    return [boxes, objectness, classes, nums]

