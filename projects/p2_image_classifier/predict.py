import sys
import argparse
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json


def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)

def process_image(image):
    image_size=224
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image,(image_size,image_size))
    image = tf.cast(image,tf.float32)
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k):
    img_obj = Image.open(image_path)
    img_np = np.asarray(img_obj)
    processed_img = process_image(img_np)
    expanded_img = np.expand_dims(processed_img, axis=0)
    predicted = model.predict(expanded_img)
    tnsr = tf.math.top_k(predicted, top_k, sorted=True)
    probs = tnsr.values.numpy()[0]
    classes = [str(class_ + 1) for class_ in tnsr.indices.numpy()[0]]
    return probs, classes


if __name__ == '__main__':
    commands_parser = argparse.ArgumentParser()
    commands_parser.add_argument('image_path', action="store")
    commands_parser.add_argument('model', action="store")
    commands_parser.add_argument('--top_k', action="store", dest="topK", type=int)
    commands_parser.add_argument('--category_names', action="store", dest="category_names")

    parser_obj = commands_parser.parse_args()

    topK = parser_obj.topK

    if parser_obj.image_path == None:
        print("Please Enter Image path")
        exit()      
    elif parser_obj.model == None:
        print("Please Enter Model path")
        exit()
    elif parser_obj.topK == None:
        topK = 3
    else:
        pass
    print(parser_obj.image_path)
    class_names = json.load(open(parser_obj.category_names))
#     img = Image.open(parser_obj.image_path)
#     np_img = np.asarray(img)
    flowers_model = load_model(parser_obj.model)
    probs, classes = predict(parser_obj.image_path, flowers_model, topK)
    for img_label, prob in list(zip(classes, probs)):
        print("Flower Name: "+class_names[img_label]+ " and its probability is:  "+ str(prob))
