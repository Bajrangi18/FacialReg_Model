import face_recognition
import imutils
import pickle
import os.path
import cv2
from upload_Firebase import checkName
import time


cascPathface = "D:\iSchoolConnect\Python\haarcascade_frontalface_alt2.xml"
cascPath = "D:\iSchoolConnect\Python\haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)
faceCascade1 = cv2.CascadeClassifier(cascPath)

data = pickle.loads(open('face_enc', "rb").read())

while(os.path.exists("trainSet\webcamImage.jpg")):
 time.sleep(2)
 image = cv2.imread("trainSet\webcamImage.jpg")
 print(type(image))
 time.sleep(2)

 if image is not None:
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

     faces = faceCascade1.detectMultiScale(
         gray,
         scaleFactor=1.1,
         minNeighbors=5,
         minSize=(30, 30),
         flags=cv2.CASCADE_SCALE_IMAGE
     )

     if len(faces) != 0:
         rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         #faces = faceCascade.detectMultiScale(gray,
         #                                     scaleFactor=1.1,
         #                                     minNeighbors=5,
         #                                     minSize=(60, 60),
         #                                     flags=cv2.CASCADE_SCALE_IMAGE)

         encodings = face_recognition.face_encodings(rgb)
         names = []

         for encoding in encodings:
             matches = face_recognition.compare_faces(data["encodings"],
                                                      encoding)
             name = "Unknown"

             if True in matches:

                 matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                 counts = {}

                 for i in matchedIdxs:
                     name = data["names"][i]

                     counts[name] = counts.get(name, 0) + 1

                     name = max(counts, key=counts.get)

                 names.append(name)
                 # checkName(''.join(names))
                 print(checkName(''.join(names)))
             else:
                 continue;

     else:
         continue;
