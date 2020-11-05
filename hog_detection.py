import cv2


# load input image
input_img = cv2.imread('temp_gol_1.jpg')

# start hog classifier
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 

# detect players
bounding_boxes, _ = hog.detectMultiScale(input_img, winStride=(4, 4), padding=(0, 0), scale=1.05)


# draw results
for (x, y, w, h) in bounding_boxes:
    
    # bounding boxes
    cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # centroids
    cv2.circle(input_img, (x + w // 2, y + h // 2), 2, (0, 0, 255), 2)


# show the result
cv2.imshow('teste', input_img)
cv2.waitKey()
cv2.destroyAllWindows()
