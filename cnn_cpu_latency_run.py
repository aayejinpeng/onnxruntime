import os
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import sys
import time
from onnx import version_converter
from onnx import shape_inference

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

def predict(model_path, label_path, img_path, batch_size, thread_num):
    model = onnx.load(model_path)
    model = version_converter.convert_version(model, 19)

    for node in model.graph.node:
        if node.op_type == 'Reshape':
            print(node)
            node.op_type = 'Flatten'
            node.input.remove('OC2_DUMMY_1')
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = 'N'
    for i in range(len(model.graph.value_info)):
        vi = model.graph.value_info[i]
        if vi.type.tensor_type.shape.dim[0].dim_value == 1:
            vi.type.tensor_type.shape.dim[0].dim_param = 'N'
    model = shape_inference.infer_shapes(model)
    # print(model.graph.value_info)

    sess_options = ort.SessionOptions()

    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(model.SerializeToString(),providers=['DnnlExecutionProvider'],provider_options=[{'num_of_threads':thread_num}])
    
    with open(label_path, 'r') as f:
        labels = [l.rstrip() for l in f]

    img = get_image(img_path, show=False)
    img = preprocess(img)
    
    img = np.tile(img, (batch_size, 1, 1, 1))
    print(img.shape)
    print(session.get_inputs()[0].name)
    ort_inputs = {session.get_inputs()[0].name: img}

    times = []
    for i in range(128):
        session.run(None, ort_inputs)
    
    for i in range(1024):
        start = time.time()
        session.run(None, ort_inputs)
        end = time.time()
        times.append(end-start)
    times.sort()
    print(times)

    #90% average time
    print('90% average time: ', np.mean(times[0:921]))
    #90% average time, per batch
    print('90% average time, per batch: ', np.mean(times[0:921])/batch_size)
    #99% average time
    print('99% average time: ', np.mean(times[0:1010]))
    #99% average time, per batch
    print('99% average time, per batch: ', np.mean(times[0:1010])/batch_size)
    #top 1 time
    print('top 1 time: ', times[0])


if len(sys.argv) != 6:
    print("Usage: python script.py model_path label_path img_path batch_size thread_num")
    sys.exit(1)

model_path = sys.argv[1]
label_path = sys.argv[2]
img_path = sys.argv[3]
batch_size = int(sys.argv[4])
thread_num = int(sys.argv[5])

predict(model_path, label_path, img_path, batch_size, thread_num)
