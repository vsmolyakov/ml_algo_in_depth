import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler 
from keras.callbacks import EarlyStopping

import os
import random
from PIL import Image
from scipy.spatial import distance
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(42)

SAVE_PATH = "/content/drive/MyDrive/Colab Notebooks/data/"
DATA_PATH = "/content/drive/MyDrive/data/101_ObjectCategories/"

def get_closest_images(acts, query_image_idx, num_results=5):

    num_images, dim = acts.shape
    distances = []
    for image_idx in range(num_images):
        distances.append(distance.euclidean(acts[query_image_idx, :], acts[image_idx, :]))
    #end for    
    idx_closest  = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
  
    return idx_closest

def get_concatenated_images(images, indexes, thumb_height):

    thumbs = []
    for idx in indexes:
        img = Image.open(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), int(thumb_height)), Image.ANTIALIAS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)

    return concat_image

if __name__ == "__main__":

    num_images = 5000
    images = [os.path.join(dp,f) for dp, dn, filenames in os.walk(DATA_PATH) for f in filenames \
              if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    images = [images[i] for i in sorted(random.sample(range(len(images)), num_images))]

    #CNN encodings
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    activations = []
    for idx, image_path in enumerate(images):
        if idx % 100 == 0:
            print('getting activations for %d/%d image...' %(idx,len(images)))
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        activations.append(features.flatten().reshape(1,-1))

    # reduce activation dimension
    print('computing PCA...')
    acts = np.concatenate(activations, axis=0)
    pca = PCA(n_components=300)
    pca.fit(acts)
    acts = pca.transform(acts)

    print('image search...') 
    query_image_idx = int(num_images*random.random())
    idx_closest = get_closest_images(acts, query_image_idx)
    query_image = get_concatenated_images(images, [query_image_idx], 300)
    results_image = get_concatenated_images(images, idx_closest, 300)

    plt.figure()
    plt.imshow(query_image)
    plt.title("query image (%d)" %query_image_idx)
    plt.show()
    #plt.savefig('./figures/query_image.png')

    plt.figure()
    plt.imshow(results_image)
    plt.title("result images")
    plt.show()
    #plt.savefig('./figures/result_images.png')