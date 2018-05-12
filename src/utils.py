
from settings import Settings
import numpy as np

def reshape_mnist_images(mnist_images):

    new_images = []
    
    for i in range(len(mnist_images)):
        new_images.append(np.reshape(mnist_images[i], (Settings.INPUT_SHAPE[0], Settings.INPUT_SHAPE[1])))
    
    return(np.array(new_images))