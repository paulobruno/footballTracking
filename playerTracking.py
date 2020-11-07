import cv2
import numpy as np
from tqdm import trange
import argparse 


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


# eliminate weak predictions
min_prob = 0.2
nms_thresh = 0.5

# configure weights
weights_path = 'external/yolov3.weights'
config_path = 'external/yolov3.cfg'

# load network with opencv dnn
network = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = network.getLayerNames()

# get output layers
output_layer_names = [layer_names[i[0] - 1] for i in network.getUnconnectedOutLayers()]


video = cv2.VideoCapture(args.input_video_file)

four_cc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(args.output_filename, four_cc, 24.0, (960, 540))

num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


print('Processing frames...')

for _ in trange(num_frames):

    ret, frame = video.read()

    frame = cv2.resize(frame, (960, 540))

    # get blob from image
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # get the network outputs
    network.setInput(blob)
    network_output = network.forward(output_layer_names)

    # save the bounding boxes
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
                
        # bounding boxes
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    output_video.write(frame)

print('...Done!')


video.release()
output_video.release()

