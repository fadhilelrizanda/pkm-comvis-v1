from tools import generate_detections as gdet
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import preprocessing, nn_matching
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from core.config import cfg
from tensorflow.python.saved_model import tag_constants
from core.yolov4 import filter_boxes
import core.utils as utils
from absl.flags import FLAGS as fflags
from absl import app, flags, logging
import tensorflow as tf
import time
import os
from collections import deque
import math

import json

import websocket
import time

try:
    import thread
except ImportError:
    import _thread as thread


def on_message(ws, message):
    print(message)


def on_error(ws, error):
    print(error)


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")


# Changed Variable
total_k_kecil = int(0)


def on_open(ws):
    def run(*args):
        global total_k_kecil
        ws.send(json.dumps({"event": "vehicle_1", "data": {
                "vehicle": total_k_kecil, "iot_token": "057110891c54543e9b645c8aaa65d25ea6c27f5d8e620fafb98e5aff9fbbf7f3"}}))
        time.sleep(1)
        ws.close()
        print("thread terminating...")
    thread.start_new_thread(run, ())


# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices(
    'GPU')  # Check GPU Device
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# deep sort imports
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'mp4v',
                    'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', True, 'dont show video output')
flags.DEFINE_boolean('info', True, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', True, 'count objects being tracked on screen')


def intersect(current_point, prev_point, point_line_1, point_line_2):
    return ccw(current_point, prev_point, point_line_2) != ccw(prev_point, point_line_1, point_line_2) and ccw(current_point, prev_point, point_line_1) != ccw(current_point, prev_point, point_line_2)


def ccw(current_point, prev_point, point_line_1):
    return (point_line_1[1]-current_point[1])*(prev_point[0]-current_point[0]) > (prev_point[1]-current_point[1])*(point_line_1[0]-point_line_1[0])


def vector_vehicle(point_A, Point_B):
    x = point_A[0] - Point_B[0]
    y = point_A[1] - Point_B[1]
    return math.degrees(math.atan2(y, x))


def main(_argv):
    current_time = time.time()
    global total_k_kecil
    # Websocket
    websocket.enableTrace(True)
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 0.5

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(fflags)
    input_size = fflags.size
    video_path = fflags.video

    # For Gate (Counter)
    pts = [deque(maxlen=30) for _ in range(1000)]
    counter = []
    kendaraan_kecil_count = []
    kendaraan_besar_count = []
    memory = {}
    # temporary memory for storing counted IDs
    already_counted = deque(maxlen=50)
    up_count = int(0)
    down_count = int(0)
    line_1_point_x = int(396)
    line_1_point_y = int(366)
    line_2_point_x = int(870)
    line_2_point_y = int(368)

    line_1_point_x_2 = int(337)
    line_1_point_y_2 = int(406)
    line_2_point_x_2 = int(935)
    line_2_point_y_2 = int(411)

    # load tflite model if flag is set
    if fflags.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=fflags.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(
            fflags.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if fflags.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*fflags.output_format)
        out = cv2.VideoWriter(fflags.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num += 1
        print('Frame_2 #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if fflags.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(
                output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if fflags.model == 'yolov3' and fflags.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=fflags.iou,
            score_threshold=fflags.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        # allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if fflags.count:
            cv2.putText(frame, "Objects being tracked: {}".format(
                count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox,
                      score, class_name, feature in zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(
            boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            if track.track_id not in memory:
                memory[track.track_id] = deque(maxlen=2)

                # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(
                bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(
                len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),
                        (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255, 255, 255), 2)

            center = (int(((bbox[0]) + (bbox[2]))/2),
                      int(((bbox[1])+(bbox[3]))/2))
            pts[track.track_id].append(center)

            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64/float(j+1))*2)
                cv2.line(frame, (pts[track.track_id][j-1]),
                         (pts[track.track_id][j]), color, thickness)

            center_y = int(((bbox[1])+(bbox[3]))/2)
            center_x = int(((bbox[0])+(bbox[2]))/2)
            current_point = [center_x, center_y]
            memory[track.track_id].append(current_point)
            previous_point = memory[track.track_id][0]

            line = [(line_1_point_x, line_1_point_y),
                    (line_2_point_x, line_2_point_y)]

            line2 = [(line_1_point_x_2, line_1_point_y_2),
                     (line_2_point_x_2, line_2_point_y_2)]
            cv2.line(frame, (line_1_point_x, line_1_point_y),
                     (line_2_point_x, line_2_point_y), (0, 255, 0), thickness=4)

            cv2.line(frame, (line_1_point_x_2, line_1_point_y_2),
                     (line_2_point_x_2, line_2_point_y_2), (0, 255, 255), thickness=4)

            if intersect(current_point, previous_point, line[0], line[1]) and track.track_id not in already_counted:
                if class_name == 'kendaraan_kecil' or class_name == 'kendaraan_besar':
                    time_speed = time.time()
                    dict_speed = {}
                    dict_speed[track.track_id] = time_speed
                    counter.append(int(track.track_id))
                    if class_name == 'kendaraan_kecil':
                        kendaraan_kecil_count.append(int(track.track_id))
                    if class_name == 'kendaraan_besar':
                        kendaraan_besar_count.append(int(track.track_id))
                    # counter.append(int(track.track_id))

                    already_counted.append(track.track_id)

                    angle = vector_vehicle(current_point, previous_point)

                    if angle > 0:
                        down_count += 1

                    if angle < 0:
                        up_count += 1

            if intersect(current_point, previous_point, line2[0], line2[1]):
                speed = 5 (time.time() - dict_speed[track.track_id])
                print("Speed : " + str(speed))

            if len(memory) > 50:
                del memory[list(memory)[0]]

        # if enable info flag then print details about each track
            if fflags.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                    str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        total_count = len(set(counter))
        total_k_kecil = len(set(kendaraan_kecil_count))
        total_k_besar = len(set(kendaraan_besar_count))
        print(total_k_kecil)

        cv2.putText(frame, "Total Kendaraan: " +
                    str(total_count), (0, 100), 0, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Kendaraan Kecil: " +
                    str(total_k_kecil), (0, 150), 0, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Kendaraan Besar: " +
                    str(total_k_besar), (0, 200), 0, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Kendaraan Up: " +
                    str(up_count), (0, 250), 0, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Kendaraan Down: " +
                    str(down_count), (0, 300), 0, 1, (255, 255, 255), 2)
        cv2.putText(frame, "FPS : " + str(int(fps)),
                    (0, 50), 0, 1, (0, 0, 255), 2)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # if time.time() - current_time > 20:
        #     ws = websocket.WebSocketApp("wss://sipejam-restfullapi.herokuapp.com",
        #                                 on_open=on_open,
        #                                 on_message=on_message,
        #                                 on_error=on_error,
        #                                 on_close=on_close)
        #     ws.run_forever()
        #     current_time = time.time()
        if not fflags.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if fflags.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
