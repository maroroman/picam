from picamera2 import Picamera2
import time
import numpy
import cv2
import math

picam2 = Picamera2()
picam2.start()
time.sleep(1)
image = picam2.capture_image("main")
 
open_cv_image = numpy.array(image) 
# Convert RGB to BGR 
open_cv_image = open_cv_image[:, :, ::-1].copy() 

img_shape = open_cv_image.shape
tile_size = (64, 48)
offset = (64, 48)

for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
    for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
        cropped_img = open_cv_image[offset[1]*i:min(offset[1]*i+tile_size[1], img_shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])]
        # Debugging the tiles
        cv2.imwrite("./tiles/debug_" + str(i) + "_" + str(j) + ".png", cropped_img)
