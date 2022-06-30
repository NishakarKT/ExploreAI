import cv2
import threading
from django.http import HttpResponseNotFound, StreamingHttpResponse

net = cv2.dnn.readNet("data/objects/yolov4-tiny.weights",
                      "data/objects/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

classes = []
with open("data/objects/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        try:
            (class_ids, scores, bboxes) = model.detect(
                image, confThreshold=0.3, nmsThreshold=.4)
            for class_id, score, bbox in zip(class_ids, scores, bboxes):
                (x, y, w, h) = bbox
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)
                cv2.putText(image, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            pass
        except:
            pass
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def index(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        return HttpResponseNotFound("Objects need a camera!")
