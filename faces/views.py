import cv2
import threading
from .utils import FaceRec
from django.shortcuts import redirect, render
from .forms import FacesUserForm
from django.http import HttpResponseNotFound, StreamingHttpResponse

fr = FaceRec()

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
            face_locations, face_names = fr.detect_known_faces(image)
            for loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = loc[0], loc[1], loc[2], loc[3]
                cv2.putText(image, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        except:
            pass
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def load_encodings():
    fr.load_encoding_images()
load_encodings()

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
        return HttpResponseNotFound("Faces need a camera!")
        
def register(request):
    if request.method == "POST":
        form = FacesUserForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            fr.load_encoding_images()
            return redirect('faces:index')
    return render(request, "faces/register.html", {})
