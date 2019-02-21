import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
'''
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import utils
import pprint
import os
'''
from keras.models import load_model
import time
import airsim
'''
parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument('model', type=str, help='Path to model h5 file. Model should be on the same path.')
args = parser.parse_args()
'''
model = load_model('./model.h5')

'''sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
'''
client = airsim.CarClient()
client.confirmConnection()
print('Connected')
client.enableApiControl(True)

MAX_SPEED = 20
MIN_SPEED = 5
speed_limit = MAX_SPEED

car_state = client.getCarState()
print(car_state.speed)
print(car_state.gear)

steering_angle_g=0
throttle_g=0

while True:
    if True:
        steering_angle = steering_angle_g
        throttle = throttle_g
        speed = car_state.speed
        image = client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.Scene)])
        try:
            image = np.asarray(image)       
            #image = utils.preprocess(image) 
            image = np.array([image])       
            steering_angle_g = float(model.predict(image, batch_size=1))
            
            if speed > speed_limit:
                speed_limit = MIN_SPEED  
            else:
                speed_limit = MAX_SPEED
            
            throttle_g = 1.0 - steering_angle_g**2 - (speed/speed_limit)**2  #Magic Number!!!

            print('{} {} {}'.format(steering_angle_g, throttle_g, speed))
            send_control(steering_angle_g, throttle_g)
        except Exception as e:
            print(e)

        
    else:
        
        sio.emit('manual', data={}, skip_sid=True)

'''
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)
'''

def send_control(steering_angle, throttle):
    car_controls = airsim.CarControls()
    car_controls.throttle = throttle
    car_controls.steering = steering_angle
    if throttle<0:
	    car_controls.handbrake=True
    else:
	    car_controls.handbrake=False
    client.setCarControls(car_controls)


