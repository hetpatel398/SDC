import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
from PIL import Image
from keras.models import load_model
import time
import airsim
from io import BytesIO
import base64
import utils

def send_control(steering_angle, throttle):
    car_controls = airsim.CarControls()
    car_controls.throttle = throttle
    car_controls.steering = steering_angle
    if throttle<0.15:
	    car_controls.handbrake=True
    else:
	    car_controls.handbrake=False
    client.setCarControls(car_controls)


model = load_model('./model.h5')

client = airsim.CarClient()
client.confirmConnection()
print('Connected')
client.enableApiControl(True)

MAX_SPEED = 10
MIN_SPEED = 2
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
        responses = client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.Scene)])  
        airsim.write_file(os.path.normpath('.\image.png'), responses[0].image_data_uint8)
        image = Image.open('.\image.png')
        try:
            image = np.asarray(image)       
            image = utils.preprocess(image) 
            image = np.array([image])       
            steering_angle_g = float(model.predict(image, batch_size=1))
            
            if speed > speed_limit:
                speed_limit = MIN_SPEED  
            else:
                speed_limit = MAX_SPEED
            
            throttle_g = 0.65 - steering_angle_g**2 - (speed/speed_limit)**2  #Magic Number!!!

            print('{} {} {}'.format(steering_angle_g, throttle_g, speed))
            send_control(steering_angle_g, throttle_g)
        except Exception as e:
            print(e)

        
    else:
        
        sio.emit('manual', data={}, skip_sid=True)

