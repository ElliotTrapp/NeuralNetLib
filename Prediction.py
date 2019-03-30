'''
Prediction.py
Elliot Trapp
18/11/15

Prediction tool specifically for image data
'''
# Predict func
from keras.applications.inception_v3 import preprocess_input as incep_preprocess
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.applications.mobilenet import preprocess_input as mobilenet_preprocess

from keras.preprocessing import image
from keras.models import load_model

def PredictImage(model, test_img, HEIGHT=150, WIDTH=150):
    img = image.load_img(test_img, target_size=(HEIGHT, WIDTH))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    print('model:\n', model)
    print('model.name:\n', model.name)
    
    if 'inception' in model.name:
        x = incep_preprocess(x)
    elif 'vgg16' in model.name:
        x = vgg16_preprocess(x)
    elif 'mobilenet' in model.name:
        x = mobilenet_preprocess(x)
    else:
        assert(False)
    
    preds = model.predict(x)
    print(preds[0])
    print("Labels:\n",train_generator.class_indices)
    return preds[0]