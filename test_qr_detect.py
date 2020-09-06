import cv2
import numpy as np
import recognise


def video_reader():
    cam = cv2.VideoCapture(0)  # включаем камеру
    while True:
        _, img = cam.read()
        recognise.recognise_qr(img, trace=True)

        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()


video_reader()
