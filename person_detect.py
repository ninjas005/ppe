import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

capture = cv2.VideoCapture(0)

# Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Load the input image.
image = mp.Image.create_from_file(IMAGE_FILE)

# Detect objects in the input image.
detection_result = detector.detect(image)

# Process the detection result. In this case, visualize it.
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
cv2_imshow(rgb_annotated_image)


def main():
    global capture

    while 1:
        _, frame = capture.read()



if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()


