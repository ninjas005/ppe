from flask import Flask, Response, redirect, url_for
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from typing import Tuple, Union
import mediapipe as mp
import numpy as np
import time
import cv2
import sys
import math

capture = cv2.VideoCapture(0)
app = Flask(__name__)

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 255)  # red


def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int, image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualize.
    Returns:
        Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                           width, height)
            color, thickness, radius = (0, 255, 0), 1, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image


def main1():
    cap = cv2.VideoCapture(0)

    # Initialise Mediapipe poses
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # LOOP
    while 1:
        _, frame = cap.read()
        flipped = cv2.flip(frame, flipCode=1)
        frame1 = cv2.resize(flipped, (640, 480))
        rgb_img = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_img)
        print(result.pose_landmarks)

        # X,Y coordinate Details on single location in this case the Nose Location.
        try:
            print('X Coords are', result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * 640)
            print('Y Coords are', result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * 480)
        except:
            pass

        # Draw the framework of body onto the processed image and then show it in the preview window
        mp_draw.draw_landmarks(frame1, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow("frame", frame1)

        # At any point if the | q | is pressed on the keyboard then the system will stop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()


def main2():
    global capture
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    def get_face_mesh(image):
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            return image
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            # print(' face_landmarks:', face_landmarks)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
            # print('%d facemesh_landmarks'%len(face_landmarks.landmark))
        return annotated_image

    font = cv2.FONT_HERSHEY_SIMPLEX
    if capture.isOpened() == False:
        print("Unable to read camera feed")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while capture.isOpened():
        s = time.time()
        ret, img = capture.read()
        if not ret:
            print('WebCAM Read Error')
            sys.exit(0)

        annotated = get_face_mesh(img)
        e = time.time()
        fps = 1 / (e - s)
        cv2.putText(annotated, 'FPS:%5.2f' % (fps), (10, 50), font, fontScale=1, color=(0, 255, 0), thickness=1)
        # cv2.imshow('webcam', annotated)
        _, jpeg = cv2.imencode('.jpg', annotated)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        if _:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    capture.release()


def main():
    base_options = python.BaseOptions(model_asset_path='detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if not capture.isOpened():
        print("Unable to read camera feed")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while capture.isOpened():
        s = time.time()
        _, frame = capture.read()
        if not _:
            print('WebCam Read Error')
            sys.exit()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Detect Faces in the input frame
        detection_result = detector.detect(mp_image)

        # Visualise the result
        image_copy = np.copy(mp_image.numpy_view())
        annotated_image = visualize(image_copy, detection_result)
        rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        fps = 1 / (time.time() - s)
        cv2.putText(annotated_image, 'FPS:%5.2f' % fps, (10, 50), font, fontScale=1, color=(0, 255, 0), thickness=1)
        _, jpeg = cv2.imencode('.jpg', annotated_image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        if _:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    capture.release()


@app.route('/')
def index():
    return redirect(url_for('video_feed'))


@app.route('/feed')
def video_feed():
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=False)
    cv2.destroyAllWindows()
