from PIL import Image 
import numpy as np 
from pathlib import Path 
import keras.utils as image
from keras.applications.vgg16 import VGG16, preprocess_input 
from keras.models import Model
import numpy as np 
from feature_extraction import FeatureExtractor

if __name__ == "__main__" :
    fe = FeatureExtractor()
    for img_path in sorted(Path("./static/img").glob("*.jpg")) : 
        feature = fe.extract(img=Image.open(img_path))
        print(type(feature),feature.shape)
        feature_path = Path("./static/feature/test") / (img_path.stem + ".npy") 
        print(feature_path)
        np.save(feature_path,feature)