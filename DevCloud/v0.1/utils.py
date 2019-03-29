import cv2, os
import numpy as np
import matplotlib.image as mpimg


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 144, 256, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def crop(image):
    return image[:, :, :] # remove the sky and the car front


def resize(image):
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    choice = np.random.choice(3)
    return load_image(data_dir, center), steering_angle


def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    #image, steering_angle = random_flip(image, steering_angle)
    #image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    #image = random_shadow(image)
    #image = random_brightness(image)
    return image, steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            image = load_image(data_dir, center) 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers

