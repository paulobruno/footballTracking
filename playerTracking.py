import cv2
import numpy as np


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


video = cv2.VideoCapture('videos/temp_gol_1.mp4')

ret, frame = video.read()

while frame is not None:

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


    cv2.imshow('teste', frame)
    cv2.waitKey(1)

    ret, frame = video.read()


video.release()
cv2.destroyAllWindows()
