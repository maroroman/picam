## Goal
Implement classification of items in video tiles captured by a raspberry pi camera

## Outcomes
A script has been created that is capable of splitting a video feed from the raspberry pi through the use of picamera2

it uses Resnet50 to classify the contents of each video tile.

Due to computing constraints, each frame takes about 2.5 seconds to process

## Implementation

### Camera module

The camera is accessed through the picamera2 module which is compatible with raspi 64 bit systems

The only configuration set is the image size: 

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (1920, 1080)})
    picam2.configure(config)
    picam2.start()

### Image processing

Each image is processed using numpy functions, specifically reshape.

First it is put into a numpy array, where the fourth channel (alpha) is ignored.

Then the image is cropped and reshaped to fit the shape required by the Resnet50 model, a.k.a it is tiled into an array of 224x224 images

    open_cv_image = open_cv_image[0:(224 * math.floor(height/224)), 0:(224 * math.floor(width/224))]

    tiled_array = open_cv_image.reshape(height//224, 224, width//224, 224, 3)
    tiled_array = tiled_array.swapaxes(1,2)
    tiled_array = tiled_array.reshape(-1,224,224, 3)

The array is fed through a function called blobFromImages which adds a specific scalefactor and mean to each image, these values should correspond to the required values by Resnet50. The result is a blob of 32 images that are fed into the model.

Finally The output is taken and class names are assigned based on the model ouput and displayed on the tiles themselves.

## Approach

1. Tested the camera using `libcamera-still -o test.jpg`
2. Attempted usage of OpenCV for capturing raspi camera output, This output however, was empty. picamera library is incompatible with any raspi of the 64-bit version, therefore the much newer picamera2 module had to be used.
3. Basic code was assembled to test picamera2 functionality, including creating a PIL image stream to use in real time classification
4. Input tiling was investigated, in the end numpy.reshape was used, based on this tutorial: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
5. For classification, Resnet50 was chosen as it was the first google result when typing Resnet ONNX
6. Resnet onnx file was retrieved along with its synset
7. For classification the following tutorial was used: https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/#guide-to-image-classification
8. Tile re-stitching was attempted but aborted due to time constraints
9. Switched model to MobileNet, reaching speeds about 5 times faster per frame.
10. Discovered that printing shape significantly slows down the entire process, speeding up the program
11. Optimizing the writing function gained about 0.2 seconds
