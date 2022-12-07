"""Useful functions and variables to be shared across the core modules"""

import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def reshape_split(image: np.ndarray):
    """Take an input image and slice it into a 7x7 array of subimages
    """

    img_height, img_width, channels = image.shape
    tile_height, tile_width = img_height//7, img_width//7
    
    tiled_array = image.reshape(7,
                                tile_height,
                                7,
                                tile_width,
                                channels)
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array

def save_tiles(image: np.ndarray, dir_path: str = os.path.join(BASE_PATH, 'tiles_tmp')):
    """Take an input image, slice it into a 7x7 array, and save each subimage
    """

    tiled_image = reshape_split(image)
    counter = 0
    for i in range(7):
        for j in range(7):
            counter+=1
            im_to_save = cv2.cvtColor(tiled_image[i, j], cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(dir_path, f"img{i}_{j}.png"), im_to_save)
    return counter

def image_tile(image, categories):
    my_dpi=300.
    print(image.shape)
    # Set up figure
    fig=plt.figure(figsize=(float(image.shape[1])/my_dpi,float(image.shape[0])/my_dpi),dpi=my_dpi)
    ax=fig.add_subplot(111)
    # Remove whitespace from around the image
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    # Set the gridding interval: here we use the major tick interval
    myIntervalx=image.shape[1]/7
    myIntervaly=image.shape[0]/7
    locx = plticker.MultipleLocator(base=myIntervalx)
    locy = plticker.MultipleLocator(base=myIntervaly)
    ax.xaxis.set_major_locator(locx)
    ax.yaxis.set_major_locator(locy)

    # Add the grid
    ax.grid(which='major', axis='both', linestyle='-')

    # Add the image
    ax.imshow(image)

    # Add some labels to the gridsquares
    bgcolors = ['blue','green', 'yellow', 'orange', 'red', 'black']
    colors = ['w', 'w', 'black', 'black', 'w', 'w']
    for j in range(7):
        y=myIntervaly/2+j*myIntervaly
        for i in range(7):
            x=myIntervalx/2.+float(i)*myIntervalx
            bgc = bgcolors[int(categories[i+j*7])]
            c = colors[int(categories[i+j*7])]
            ax.text(x, y, categories[i+j*7], color=c, backgroundcolor=bgc, ha='center', va='center')
    return fig