import cv2
import numpy as np
from tqdm import trange
import argparse 
import time

from math import sqrt
from typing import Tuple
from collections import namedtuple


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))


# parse command line args
parser = argparse.ArgumentParser(
    description='Track football players from video.')

parser.add_argument(
    'input_video_file',
    help='Path to the input video file.')
parser.add_argument(
    '-o', '--output-filename',
    metavar='FILENAME',
    help='Name of the output file. [default = output.mp4]',
    default='output.mp4')

args = parser.parse_args()


# max centroids distance to consider as the same player
dist_threshold = 20.0

# eliminate weak predictions
min_prob = 0.2
nms_thresh = 0.5

# yolov3 paths
weights_path = 'external/yolov3.weights'
config_path = 'external/yolov3.cfg'

# load network with opencv's dnn
network = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = network.getLayerNames()

# get output layers
output_layer_names = [layer_names[i[0] - 1] for i in network.getUnconnectedOutLayers()]


# load video and define output settings
video = cv2.VideoCapture(args.input_video_file)

four_cc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(args.output_filename, four_cc, 24.0, (960, 540))

num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


print('Processing frames...')

start_time = time.time()


# 0 used for invalid IDs
nextIdAvailable = 1

centroids = []
player_ids = []


# 1st frame

ret, frame = video.read()
frame = cv2.resize(frame, (960, 540))

blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

network.setInput(blob)
network_output = network.forward(output_layer_names)

bounding_boxes = []
confidence_values = []

img_h, img_w = frame.shape[:2]

for result in network_output:
    
    for detection in result:

        scores = detection[5:]
        top_1_class = np.argmax(scores)
        confidence = scores[top_1_class]

        # only allow people and with enough confidence
        if (0 == top_1_class) and (confidence > min_prob):

            bounding_box = detection[0:4] * np.array([img_w, img_h, img_w, img_h])

            x_center, y_center, box_w, box_h = bounding_box.astype(int)

            x_min = x_center - (box_w // 2)
            y_min = y_center - (box_h // 2)

            bounding_boxes.append([int(x_min), int(y_min), int(box_w), int(box_h)])
            confidence_values.append(float(confidence))

# remove duplicate detections with Non-Maximum Suppression
results = cv2.dnn.NMSBoxes(bounding_boxes, confidence_values, min_prob, nms_thresh)

# draw results
for i in results.flatten():
    
    (x, y, w, h) = bounding_boxes[i]

    c_x = x + (w / 2)
    c_y = y + (y / 2)
    
    centroids.append((c_x, c_y))
    player_ids.append(nextIdAvailable)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(frame, str(nextIdAvailable), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    nextIdAvailable += 1
    
output_video.write(frame)


# next frames

for _ in trange(num_frames-1):
        
    centroids_new = []
    ids_new = []

    ret, frame = video.read()
    frame = cv2.resize(frame, (960, 540))

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    network.setInput(blob)
    network_output = network.forward(output_layer_names)

    bounding_boxes = []
    confidence_values = []

    img_h, img_w = frame.shape[:2]

    for result in network_output:
        
        for detection in result:

            scores = detection[5:]
            top_1_class = np.argmax(scores)
            confidence = scores[top_1_class]

            # only allow people and with enough confidence
            if (0 == top_1_class) and (confidence > min_prob):

                bounding_box = detection[0:4] * np.array([img_w, img_h, img_w, img_h])

                x_center, y_center, box_w, box_h = bounding_box.astype(int)

                x_min = x_center - (box_w // 2)
                y_min = y_center - (box_h // 2)

                bounding_boxes.append([int(x_min), int(y_min), int(box_w), int(box_h)])
                confidence_values.append(float(confidence))

    
    # remove duplicate detections with Non-Maximum Suppression
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidence_values, min_prob, nms_thresh)

    # draw results
    for i in results.flatten():
        
        (x, y, w, h) = bounding_boxes[i]
        
        # exclude giant bounding boxes
        if (w < 100) and (h < 200):
        
            c_x = x + (w / 2)
            c_y = y + (y / 2)
            
            dist = np.inf
            temp_id = 0
                    
            for j in range(len(centroids)):
            
                temp_dist = euclidean_distance((c_x, c_y), centroids[j])
                            
                if temp_dist < dist:
                    dist = temp_dist
                    temp_id = player_ids[j]
                    
            centroids_new.append((c_x, c_y))
            
            if dist < dist_threshold:
                ids_new.append(temp_id)
            else:
                ids_new.append(nextIdAvailable)
                nextIdAvailable += 1
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, str(ids_new[-1]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    output_video.write(frame)
    
    centroids = centroids_new
    player_ids = ids_new


elapsed_time = time.time() - start_time

print('...Done!')
print('Elapsed time: {:.3f} s. Seconds per frame: {:.3f} s.'.format(elapsed_time, elapsed_time/num_frames))


video.release()
output_video.release()
