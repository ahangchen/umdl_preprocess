import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from keras import Input
from keras.applications.resnet50 import preprocess_input
from keras.engine import Model
from keras.layers import Flatten
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50


def img_infos(dir_path, net):
    print 'generate for' + dir_path
    features, pids = [], []
    for image_name in sorted(os.listdir(dir_path)):
        if '.txt' in image_name:
            continue
        if 's' in image_name or 'f' in image_name:
            # market && duke
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1][1])
        elif 's' not in image_name:
            # grid
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1])
        else:
            continue
        image_path = os.path.join(dir_path, image_name)
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = net.predict(x).reshape(-1)
        features.append(np.concatenate((feature, feature, feature[:868])))
        pids.append(person)
    return features, pids


def dataset_feature(data_dir):
    train_dir, probe_dir, gallery_dir = data_dir + '/train', data_dir + '/probe', data_dir + '/test'
    base_model =ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    x = base_model.output
    x = Flatten(name='flatten')(x)
    print 'model prepared'
    model = Model(inputs=[base_model.input], outputs=[x])
    train_features, train_pids =img_infos(train_dir, model)
    probe_features, probe_pids =img_infos(probe_dir, model)
    gallery_features, gallery_pids =img_infos(gallery_dir, model)
    return train_features + probe_features + gallery_features, train_pids, probe_pids, gallery_pids

