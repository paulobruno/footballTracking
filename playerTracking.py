import cv2
from time import sleep


video = cv2.VideoCapture('videos/temp_gol_1.mp4')

ret, frame = video.read()

hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 

while frame is not None:

    frame = cv2.resize(frame, (960, 540))

    players, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(0, 0), scale=1.05)

    for i in range(len(players)):
        p_x, p_y, p_w, p_h = players[i]
        cv2.rectangle(frame, (p_x, p_y), (p_x + p_w, p_y + p_h), (0, 255, 0), 3)

    cv2.imshow('teste', frame)
    cv2.waitKey(1)

    ret, frame = video.read()


video.release()
cv2.destroyAllWindows()
