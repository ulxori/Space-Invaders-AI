import numpy as np
from skimage.color import rgb2gray
from skimage import transform

#INPUT_SHAPE = (70,70,1)
#TRANSFRORM_SIZE = [70, 70]

INPUT_SHAPE = (80,80,1)
TRANSFRORM_SIZE = [80, 80]

def prepare_frame(image):
    image = rgb2gray(image)
    image= image[8:-12,4:-12]
    image = image/255
    image = transform.resize(image, TRANSFRORM_SIZE)
    image = np.array(image).reshape(INPUT_SHAPE)
    return image