import numpy as np
import pydicom
from PIL import Image
import glob
import os
import pydicom.uid
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2

#Function to read dicom files
def get_names(path):
    names = []
    for _, _, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.dcm']:
                names.append(filename)
    return names


#Function to convert dicom files to jpg
def convert_dcm_jpg(cdir, name):
    im = pydicom.dcmread(cdir + '/' + name)
    im = im.pixel_array.astype(float)
    rescale_image = (np.maximum(im, 0)/im.max())*255
    final_image = np.uint8(rescale_image)

    final_image = Image.fromarray(final_image)
    return final_image
