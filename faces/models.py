import cv2
import pickle
import importlib
import numpy as np
from PIL import Image
from django.db import models

face_recognition = importlib.import_module("face_recognition")

# Create your models here.


class FacesUser(models.Model):
    name = models.CharField(max_length=100)
    image = models.ImageField(upload_to="faces/images")

    def __str__(self):
        return str(self.id)
    
    def save(self, *args, **kwargs):
        pil_img = Image.open(self.image)
        cv_img = np.array(pil_img)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        try:
            img_encoding = face_recognition.face_encodings(rgb_img)[0]
            data = {"encodings": [img_encoding], "names": [self.name]}
            f = open("data/faces/encodings.pickle", "wb")
            f.write(pickle.dumps(data))
            f.close()
        except:
            pass
        super().save(*args, **kwargs)

