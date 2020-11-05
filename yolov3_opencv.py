import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


# eliminate weak predictions
min_prob = 0.2


# load input image
input_img = cv2.imread('temp_gol_1.jpg')

# load COCO labels
#labels = open('external/coco.names').read().strip().split('\n')

# configure weights
weights_path = 'external/yolov3.weights'
config_path = 'external/yolov3.cfg'

# get blob from image
blob = cv2.dnn.blobFromImage(input_img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# load network with opencv dnn
network = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = network.getLayerNames()

# get output layers
output_layer_names = [layer_names[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# get the network outputs
network.setInput(blob)
network_output = network.forward(output_layer_names)

# save the bounding boxes
bounding_boxes = []
img_h, img_w = input_img.shape[:2]

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

            bounding_boxes.append([x_min, y_min, box_w, box_h])


# draw results
for (x, y, w, h) in bounding_boxes:
    
    # bounding boxes
    cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # centroids
    cv2.circle(input_img, (x + w // 2, y + h // 2), 2, (0, 255, 255), 2)


# show the result
cv2.imshow('teste', input_img)
cv2.waitKey()
cv2.destroyAllWindows()
