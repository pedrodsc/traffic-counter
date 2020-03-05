import cv2
import numpy as np

from is_wire.core import Channel, Subscription, Message
from is_msgs.image_pb2 import ObjectAnnotations, Image

from utility import load_options, get_np_image, get_rects, to_object_annotations, to_image

options = load_options()
print('#####################################################')
print(options)

broker = options.broker
print(f'broker: {broker}')
channel = Channel(broker)

subscription = Subscription(channel)

topic = f'CameraGateway.{options.camera_id}.Frame'
print(f'topic: {topic}')

subscription.subscribe(topic=topic)
print('subscribed')

msg = channel.consume()
print('msg consumed')

img = msg.unpack(Image)

img = get_np_image(img)

img_name = f'/home/is-tracker/etc/data/Camera_{options.camera_id}.jpg'
cv2.imwrite(img_name,img)

print(f'Image saved at {img_name}')