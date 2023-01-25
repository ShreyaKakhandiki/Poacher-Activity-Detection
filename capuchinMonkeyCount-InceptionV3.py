#!/usr/bin/env python
# coding: utf-8

# ## Process picture to find out if animals & humans are present

# In[1]:


# megadetector code from cameratraps team

import json
import os
import sys
import time
import copy
import warnings
import itertools

from datetime import datetime
from functools import partial

import humanfriendly

from detection.run_tf2_detector import ImagePathUtils, TFDetector
import visualization.visualization_utils as viz_utils

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf

print('TensorFlow version:', tf.__version__)
print('tf.test.is_gpu_available:', tf.test.is_gpu_available())


def process_images(im_files, tf_detector, confidence_threshold):
    """Runs the MegaDetector over a list of image files.

    Args
    - im_files: list of str, paths to image files
    - tf_detector: TFDetector (loaded model) or str (path to .pb model file)
    - confidence_threshold: float, only detections above this threshold are returned

    Returns
    - results: list of dict, each dict represents detections on one image
        see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
    """
    if isinstance(tf_detector, str):
        start_time = time.time()
        tf_detector = TFDetector(tf_detector)
        elapsed = time.time() - start_time
        print('Loaded model (batch level) in {}'.format(humanfriendly.format_timespan(elapsed)))

    results = []
    for im_file in im_files:
        results.append(process_image(im_file, tf_detector, confidence_threshold))
    return results


def process_image(im_file, tf_detector, confidence_threshold):
    """Runs the MegaDetector over a single image file.

    Args
    - im_file: str, path to image file
    - tf_detector: TFDetector, loaded model
    - confidence_threshold: float, only detections above this threshold are returned

    Returns:
    - result: dict representing detections on one image
        see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
    """
    print('Processing image {}'.format(im_file))
    try:
        image = viz_utils.load_image(im_file)
    except Exception as e:
        print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
        result = {
            'file': im_file,
            'failure': TFDetector.FAILURE_IMAGE_OPEN
        }
        return result

    try:
        result = tf_detector.generate_detections_one_image(
            image, im_file, detection_threshold=confidence_threshold)
    except Exception as e:
        print('Image {} cannot be processed. Exception: {}'.format(im_file, e))
        result = {
            'file': im_file,
            'failure': TFDetector.FAILURE_TF_INFER
        }
        return result

    return result


def chunks_by_number_of_chunks(ls, n):
    """Splits a list into n even chunks.

    Args
    - ls: list
    - n: int, # of chunks
    """
    for i in range(0, n):
        yield ls[i::n]


def load_and_run_detector_batch(model_file, image_file_names, checkpoint_path=None,
                                confidence_threshold=0, checkpoint_frequency=-1,
                                results=None, n_cores=0):
    """
    Args
    - model_file: str, path to .pb model file
    - image_file_names: list of str, paths to image files
    - checkpoint_path: str, path to JSON checkpoint file
    - confidence_threshold: float, only detections above this threshold are returned
    - checkpoint_frequency: int, write results to JSON checkpoint file every N images
    - results: list of dict, existing results loaded from checkpoint
    - n_cores: int, # of CPU cores to use

    Returns
    - results: list of dict, each dict represents detections on one image
    """
    if results is None:
        results = []

    already_processed = set([i['file'] for i in results])

    if n_cores > 1 and tf.test.is_gpu_available():
        print('Warning: multiple cores requested, but a GPU is available; parallelization across GPUs is not currently supported, defaulting to one GPU')

    # print("image_file_names", image_file_names)
    # print("length of image_file_names", len(image_file_names))

    
    # If we're not using multiprocessing...
    if n_cores <= 1 or tf.test.is_gpu_available():

        # Load the detector
        start_time = time.time()
        tf_detector = TFDetector(model_file)
        elapsed = time.time() - start_time
        print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))

        # Does not count those already processed
        count = 0

        im_file = image_file_names

        result = process_image(im_file, tf_detector, confidence_threshold)
        results.append(result)

        # checkpoint
        if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:
            print('Writing a new checkpoint after having processed {} images since last restart'.format(count))
            with open(checkpoint_path, 'w') as f:
                json.dump({'images': results}, f)

    else:
        # when using multiprocessing, let the workers load the model
        tf_detector = model_file

        print('Creating pool with {} cores'.format(n_cores))

        if len(already_processed) > 0:
            print('Warning: when using multiprocessing, all images are reprocessed')

        pool = workerpool(n_cores)

        image_batches = list(chunks_by_number_of_chunks(image_file_names, n_cores))
        results = pool.map(partial(process_images, tf_detector=tf_detector,
                                   confidence_threshold=confidence_threshold), image_batches)

        results = list(itertools.chain.from_iterable(results))

    # results may have been modified in place, but we also return it for backwards-compatibility.
    return results


def write_results_to_file(results, output_file, relative_path_base=None):
    """Writes list of detection results to JSON output file. Format matches
    https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format

    Args
    - results: list of dict, each dict represents detections on one image
    - output_file: str, path to JSON output file, should end in '.json'
    - relative_path_base: str, path to a directory as the base for relative paths
    """
    if relative_path_base is not None:
        results_relative = []
        for r in results:
            r_relative = copy.copy(r)
            r_relative['file'] = os.path.relpath(r_relative['file'], start=relative_path_base)
            results_relative.append(r_relative)
        results = results_relative

    final_output = {
        'images': results,
        'detection_categories': TFDetector.DEFAULT_DETECTOR_LABEL_MAP,
        'info': {
            'detection_completion_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'format_version': '1.0'
        }
    }
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=1)
    print('Output file saved at {}'.format(output_file))


# ## Import libraries for imagenet model and visualization

# In[2]:


import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import visualization.visualization_utils as viz_utils
from PIL import Image


# call tf.keras.applications.mobilenet.MobileNet() to obtain pretrained MobileNet with weights that were saved from being trained on ImageNet images. 
# assign this model to the variable mobile. Note: mobilenet_v3 not yet available on keras
#mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()
#print(mobile)

# try Inception v3 instead of mobilenet
from tensorflow.keras.applications.inception_v3 import InceptionV3
mobile = InceptionV3(weights='imagenet')
print(mobile)


# ## Set input and output file paths, load image

# In[3]:


# change image and output files here during demo

model_file = "/Users/Bimba/md_v4.1.0.pb"
checkpoint_path = '/Users/Bimba/checkpoint.json'

# image_file = '/home/protego/bearProtect/input/demo.jpeg'
# output_file = '/home/protego/bearProtect/output/demo.json'

image_file = "/Users/Bimba/practicePics/CapuchinMonkeys/Capuchin1.JPG"
output_file = '/Users/Bimba/practicePics/CapuchinMonkeys/images.json'

image1 = Image.open(image_file) 
imgplot = plt.imshow(image1)
titlestr = image_file
plt.title(titlestr)
plt.show()


# ## Run the megadetector

# In[4]:


results = []
start_time = time.time()

results = load_and_run_detector_batch(model_file,
                                          image_file,
                                          checkpoint_path=checkpoint_path,
                                          confidence_threshold=TFDetector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
                                          checkpoint_frequency=-1,
                                          results=results,
                                          n_cores=0)

elapsed = time.time() - start_time
print('Finished inference in {}'.format(humanfriendly.format_timespan(elapsed)))

relative_path_base = None
write_results_to_file(results, output_file, relative_path_base=relative_path_base)

print('Done!')


# ## Parse json file to get number, type & box coordinates of detections

# In[5]:



# read json output
f = open(output_file)
data = json.load(f)

#for image_data in data['images']: 
image_data = data['images']
#print('image_data:',image_data) 


image_filename = image_data[0]['file']
print('filename: ', image_filename)
#print('image_data:',image_data)
# get number of detections made by cameratraps
numDet = len(image_data[0]['detections'])
print('num det:',numDet)

# Closing file 
f.close() 

detection_categories_str = ('Animal', 'Human', 'Vehicle')
detection_bounding_box = []
detection_confidence = []
detection_category = []
detection_params = []

for n in range(numDet):
    detection_params = image_data[0]['detections'][n]
    detection_category.append(int(detection_params['category']))
    detection_confidence.append(detection_params['conf'])
    detection_bounding_box.append(detection_params['bbox'])
print('detection_category:')
print(detection_category)


# ## Plot all cropped images of detections

# In[6]:



# Opens an image in RGB mode 
im = Image.open(image_file) 
# Shows the image in image viewer 
#im.show() 
imgplot = plt.imshow(im)
plt.show()
# Size of the image in pixels (size of orginal image) 
# (This is not mandatory) 
width, height = im.size 
print('image size:')
print(width, height)

imCrop = []
for n in range(numDet):
    print('n:',n)
    print('detection_category:',detection_category[n])
    print('detection_category:',detection_categories_str[detection_category[n]-1])
    # Setting the points for cropped image 
    bounding_box_n = detection_bounding_box[n]
    left = int(bounding_box_n[0]*width)
    top = int(bounding_box_n[1]*height)
    right = int(bounding_box_n[2]*width)+left
    bottom = int(bounding_box_n[3]*height)+top
    print('detection_bounding_box:')
    print(left,top,right,bottom)

    # Cropped image of above dimension 
    # (It will not change orginal image) 
    imCrop.append(im.crop((left, top, right, bottom))) 
    # Shows the image in image viewer 
    imgplot = plt.imshow(imCrop[n])
    titlestr = detection_categories_str[detection_category[n]-1] + ', Confidence:' + str(detection_confidence[n]*100) + '%'
    plt.title(titlestr)
    plt.show()
    


# ## Predict animal species for each animal detection

# In[7]:


# prepare_image() that accepts an image file, and processes the image to get it in a format that the model expects. We’ll be passing each of our images to this function before we use MobileNet to predict on it, so let’s see what exactly this function is doing.

def prepare_image(detection_cropped):
    #img_path = '/home/protego/bearProtect/prediction/'
    #img = image.load_img(img_path + file, target_size=(224, 224))
    newsize = (299, 299)  
    img = detection_cropped.resize(newsize)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    #return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
    return tf.keras.applications.inception_v3.preprocess_input(img_array_expanded_dims)


# ## Count the number of Capuchins

# In[10]:


capuchin_count = 0
for n in range(numDet):
    # only predict species if detection confidence level is above 60%
    if detection_confidence[n] >= 0.65:
        imgplot = plt.imshow(imCrop[n])
        titlestr = 'Category: ' + detection_categories_str[detection_category[n]-1] + ', Confidence:' + str(detection_confidence[n]*100) + '%'
        plt.title(titlestr)
        plt.show()
        preprocessed_image = prepare_image(imCrop[n])
        prediction = mobile.predict(preprocessed_image)
        results = imagenet_utils.decode_predictions(prediction)
        print(results)
        #print(len(results[0][1]))
        #print(results[0][0][1])
        if results[0][0][1] == 'capuchin':
            capuchin_count = capuchin_count + 1


# In[11]:


print('Total Number of Capuchins in Image:',capuchin_count)


# In[ ]:




