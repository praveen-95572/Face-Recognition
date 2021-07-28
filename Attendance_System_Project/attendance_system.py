import numpy as np
import cv2 as cv
import os
#import face_recognition
from datetime import datetime


path='images'
images=[]
personName=[]
myList = os.listdir(path)
#print(myList)

for cur_img in myList:
     current_img = cv.imread(f'{path}/{cur_img}')
     images.append(current_img)
     personName.append(os.path.splitext(cur_img)[0])

#print(personName)


def face_encodings(images):             #dlib(in face_recognition) used to encode face    HOG algo is used

     encodeList=[]
     for img in images:
          img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
          encode = face_recognition.face_encodings(img)[0]
          encodeList.append(encode)
     return encodeList

def attendance(name):
     with open('Attendance.csv' , 'r+') as f:
          myDataList = f.readlines()
          nameList = []
          for line in myDataList:
               entry = line.split(',')
               nameList.append(entry[0])
          if name not in nameList:
               time_now = datetime.now()
               tStr = time_now.strftime('%H:%M:%S')
               dStr = time_now.strftime('%d%m%Y')
               f.writelines(f'\n{name},{tStr},{dStr}')
     

encodeList = face_encodings(images)
print("All Encodings Complete")

cap=cv.VideoCapture(0)        # 0 -> lappy camera   1-> external camera

while True:
     ret , frame=cap.read()
     faces = cv.resize(frame,(0,0), None , 0.25, 0.25)
     faces = cv.cvtColor(faces, cv.CVTCOLOR_BGR2RGB)

     facesCurrentFrame = face_recognition.face_locations(faces)
     encodesCurrentFrame = face_recognition.face_encodings(face , facesCurrentFrame)

     for encodeFace , faceLoc in zip(encodesCurrentFrame , facesCurrentFrame):
          matches = face_recognition.compare_faces(encodeList , encodeFace)
          faceDis = face_recognition.face_distance(encodeList , encodeFace)

          matchIndex = np.argmin(faceDis)

          if matches[matchIndex]:
               name = personNames[matchIndex].upper()

               y1,x2,y2,x1 = faceLoc
               y1,x2,y2,x1 = y1*4 , x2*4 , y2*4, x1*4

               cv.rectangle(frame , (x1,y1) ,(x2,y2) ,(0 , 255 , 0), 2)
               cv.rectangle(frame , (x1,y1 - 35) ,(x2,y2) ,(0 , 255 , 0), cv.FILLED)
               cv.putText(frame , name , (x1+6 , y2-6),cv.FI+ONT_HERSHEY_COMPLEX , 1, (255 ,255, 255),2)
               attendance(name)

     cv.imshow("WEBCAM" , frame)
     if cv.waitKey(1) == 13:
          break

cap.release()
cv.destroyAllWindows()
          
               
     
