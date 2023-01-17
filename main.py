from picamera2 import Picamera2
import time
import numpy
import cv2
import math

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1920, 1080)})
picam2.configure(config)
picam2.start()
time.sleep(1)
image = picam2.capture_image("main")
 
# 1 1080p x4channels (1920,1080,4)
open_cv_image = numpy.array(image) 

# 2 1080p RGB (1920, 1080, 3)
open_cv_image = open_cv_image[:, :, :3].copy() 
print(open_cv_image.shape)
# crop
# Take the closest lower integer division resolution
width = open_cv_image.shape[1]
height = open_cv_image.shape[0]
open_cv_image = open_cv_image[0:(224 * math.floor(height/224)), 0:(224 * math.floor(width/224))]
print(open_cv_image.shape)
# tiling (224,224,3, x) - x je pocet tilov
#tiles = open_cv_image.reshape(224,224,3,64)
#print(tiles.shape)
tiled_array = open_cv_image.reshape(height//224, 224, width//224, 224, 3)
tiled_array = tiled_array.swapaxes(1,2)
tiled_array = tiled_array.reshape(-1,224,224, 3)
#tiled_array = numpy.moveaxis(tiled_array, 3, 1)
print(tiled_array.shape)

#cv2.imwrite('test.png', tiled_array)
#cv_blob_image = cv2.imread('test.png')
blob = cv2.dnn.blobFromImages(images=tiled_array, scalefactor=0.01, size=(56, 56), mean=(104, 117, 123))
print(blob.shape)
model = cv2.dnn.readNetFromONNX('resnet50-v2-7.onnx')

model.setInput(blob)
outputs = model.forward()
print(outputs.shape)

# read the ImageNet class names
with open('synset.txt', 'r') as f:
    image_net_names = f.read().split('\n')
# final class names (just the first word of the many ImageNet names for one image)
class_names = [name.split(',')[0] for name in image_net_names]


for i in range(outputs.shape[0]):
    final_outputs = outputs[i]
    # make all the outputs 1D
    final_outputs = final_outputs.reshape(1000, 1)
    # get the class label
    label_id = numpy.argmax(final_outputs)
    # convert the output scores to softmax probabilities
    probs = numpy.exp(final_outputs) / numpy.sum(numpy.exp(final_outputs))
    # get the final highest probability
    final_prob = numpy.max(probs) * 100.
    cv2.imwrite('./tiles/result_image' + str(i) + '.jpg', tiled_array[i])
    cv_blob_image = cv2.imread('./tiles/result_image' + str(i) + '.jpg')
    # map the max confidence to the class label names
    out_name = class_names[label_id]
    out_text = f"{out_name}, {final_prob:.3f}"
    # put the class name text on top of the image
    cv2.putText(cv_blob_image, out_text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite('./tiles/result_image-alt' + str(i) + '.jpg', cv_blob_image)

'''
for i in range(tiled_array.shape[0]):
    model.setInput(tiled_array[i])
    outputs = model.forward()
'''    



'''
img_shape = open_cv_image.shape
tile_size = (224, 224)
offset = (224, 224)
result_array = []
for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
    for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
        cropped_img = open_cv_image[offset[1]*i:min(offset[1]*i+tile_size[1], img_shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])]
        result_array.append(cropped_img)
'''

