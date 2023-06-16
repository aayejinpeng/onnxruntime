import os
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import sys

def get_image(path, show=False):
    with Image.open(path) as img:
        img = np.array(img.convert('RGB'))
    if show:
        plt.imshow(img)
        plt.axis('off')
    return img

def preprocess(img):
    img = img / 255.
    img = cv2.resize(img, (256, 256))
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0 : y0+224, x0 : x0+224, :]
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def predict(model_path, label_path, img_path):
    model = onnx.load(model_path)
    session = ort.InferenceSession(model.SerializeToString(),providers=['DnnlExecutionProvider'])

    with open(label_path, 'r') as f:
        labels = [l.rstrip() for l in f]

    img = get_image(img_path, show=False)
    img = preprocess(img)
    ort_inputs = {session.get_inputs()[0].name: img}
    preds = session.run(None, ort_inputs)[0]
    preds = np.squeeze(preds)
    a = np.argsort(preds)[::-1]
    print('class=%s ; probability=%f' %(labels[a[0]],preds[a[0]]))

if len(sys.argv) != 4:
    print("Usage: python script.py model_path label_path img_path")
    sys.exit(1)

model_path = sys.argv[1]
label_path = sys.argv[2]
img_path = sys.argv[3]

predict(model_path, label_path, img_path)
