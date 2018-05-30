import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import cv2
import h5py
from keras.models import load_model
from keras import __version__ as keras_version

file = sys.argv[-1]

if file == 'submit.py':
  print ("Error loading video")
  quit

# Define encoder function
def encode(array):
	pil_img = Image.fromarray(array)
	buff = BytesIO()
	pil_img.save(buff, format="PNG")
	return base64.b64encode(buff.getvalue()).decode("utf-8")

video = skvideo.io.vread(file)

answer_key = {}

model_file = '/home/workspace/Example/model_with_weight.h5'
f = h5py.File(model_file, mode='r')
model_version = f.attrs.get('keras_version')
keras_version = str(keras_version).encode('utf8')
if model_version != keras_version:
	print('You are using Keras version ', keras_version, ', but the model was built using ', model_version)

model = load_model(model_file)


width = 800
top = 220
bottom = 540
height = bottom - top
classes = 3

def normalized(rgb):
	if len(rgb.shape) != 3:
		raise RuntimeError('normalized: input data is not 3 dimensional data')
	if rgb.shape[2] != 3:
		raise RuntimeError('normalized: input data is not in RGB color')
        
	n_image = rgb[top:bottom,:,:]
	norm=np.zeros((n_image.shape[0], n_image.shape[1], 3), np.float32)
	norm[:,:,0]=cv2.equalizeHist(n_image[:,:,0])
	norm[:,:,1]=cv2.equalizeHist(n_image[:,:,1])
	norm[:,:,2]=cv2.equalizeHist(n_image[:,:,2])

	return norm

def getroad(prediction):
	tmp = np.reshape(np.where(prediction[0,:,1] > 0.5, 1, 0), (height, width)).astype(np.uint8)
	road = np.zeros(shape=(600, 800)).astype(np.uint8)
	road[top:bottom,:]=tmp
	road[:250,:] = 0
	return road

def getcars(prediction):
	tmp = np.reshape(np.where(prediction[0,:,2] > 0.5, 1, 0), (height, width)).astype(np.uint8)
	car = np.zeros(shape=(600, 800)).astype(np.uint8)
	car[top:bottom,:]=tmp
	car[:230,:] = 0
	return car

# Frame numbering starts at 1
frame = 1

for rgb_frame in video:
	
	image_array = np.asarray(normalized(rgb_frame))
	predicted = model.predict(image_array[None, :, :, :], batch_size=1)
	binary_road_result = getroad(predicted)
	binary_car_result = getcars(predicted)

	answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
    
    # Increment frame
	frame+=1

# Print output in proper json format
print (json.dumps(answer_key))
