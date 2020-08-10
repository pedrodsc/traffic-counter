import threading
import tensorflow as tf
from time import sleep
from pprint import pprint

from is_wire.core import Channel, Subscription, Message

from options_pb2 import Options
from utility import (load_options)
from is_tracker import Is_Tracker

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

def foo():
    print('tick')
    sleep(1)

def get_config(channel,config):
    msg = channel.consume()
    config[0] = msg.unpack(Options)
    
if __name__ == '__main__':
    
    options = load_options()
    broker = options.broker
    channel = Channel(broker)
    subscription = Subscription(channel)
    subscription.subscribe(topic='Tracker.Config')
    #pprint(options.TrackerOptions.filter)
    it = Is_Tracker(options)
    config = [options]
    while True:
        it.run()
        # thread = threading.Thread(target=get_config,args=[channel,config])
        # thread.start()
        # pprint(config[0])
        # foo()
