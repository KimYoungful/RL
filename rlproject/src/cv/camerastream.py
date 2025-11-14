import cv2


class CameraStream:
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(src)




    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to read from camera")
        return frame
    


    def release(self):
        self.cap.release()