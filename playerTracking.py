import cv2

video = cv2.VideoCapture('demo-s.mp4')

ret, frame = video.read()

while frame is not None:

    cv2.imshow('teste', frame)    
    cv2.waitKey(33)
    
    ret, frame = video.read()

video.release()
cv2.destroyAllWindows()
