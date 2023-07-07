import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
import urllib.request
video_capture = cv2.VideoCapture(0)
#url="http://192.168.4.1"

Anil_image = face_recognition.load_image_file("C:/Users/RISHABH DIXIT/Desktop/Project/Photos/Anil.jpeg")
Anil_encoding = face_recognition.face_encodings(Anil_image)[0]

Himanshu_image = face_recognition.load_image_file("C:/Users/RISHABH DIXIT/Desktop/Project/Photos/Himanshu.jpeg")
Himanshu_encoding = face_recognition.face_encodings(Himanshu_image)[0]

Nishant_image = face_recognition.load_image_file("C:/Users/RISHABH DIXIT/Desktop/Project/Photos/Nishant.jpeg")
Nishant_encoding = face_recognition.face_encodings(Nishant_image)[0]

Pradumn_image = face_recognition.load_image_file("C:/Users/RISHABH DIXIT/Desktop/Project/Photos/Pradumn.jpeg")
Pradumn_encoding = face_recognition.face_encodings(Pradumn_image)[0]

Sir_image = face_recognition.load_image_file("C:/Users/RISHABH DIXIT/Desktop/Project/Photos/Sir.jpeg")
Sir_encoding = face_recognition.face_encodings(Sir_image)[0]

Swati_image = face_recognition.load_image_file("C:/Users/RISHABH DIXIT/Desktop/Project/Photos/Swati.jpeg")
Swati_encoding = face_recognition.face_encodings(Swati_image)[0]

known_face_encoding = [
Anil_encoding,
Himanshu_encoding,
Nishant_encoding,
Pradumn_encoding,
Sir_encoding,
 Swati_encoding 
]

known_faces_names = [
"Anil Yadav",
"Himanshu Sharma",
"Nishant Singh",
"Pradumn Singh",
"Puneet Sir",
"Swati"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name="Unknown"
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
 
            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
 
                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
 
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
cv2.destroyAllWindows()
f.close()
