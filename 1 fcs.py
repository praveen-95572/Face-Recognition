import cv2 as cv
import numpy as np

face_classifier = cv.CascadeClassifier('C:/Users/PRAVEEN/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')



def face_extractor(img):

     gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
     faces = face_classifier.detectMultiScale(gray , 1.3,5)

     if faces is():
          return None

     for(x,y,w,h) in faces:
          cropped_face = img[y:y+h , x:x+w]

     return cropped_face

     



cap=cv.VideoCapture(0)
count=0                                                               #for photocapture


while True:
     ret , frame= cap.read()
    
     if face_extractor(frame) is not None:
          count+=1
          face = cv.resize(face_extractor(frame) , (200,200))

          face = cv.cvtColor(face , cv.COLOR_BGR2GRAY)

          file_name_path = 'W:/WORK____________/python/openCv/OpenCV-master/faces/user'+str(count)+'.jpg'                      #save sample faces

          cv.imwrite(file_name_path,face)

          cv.putText(face , str(count) , (50,50) ,cv.FONT_HERSHEY_COMPLEX , 1 , (0,255,0) , 2)
          cv.imshow('Face Cropper' ,face)

     else:
          print("Face not found ")
          pass

     if cv.waitKey(1) == 13 or count==50:
          break



cap.release()
cv.destroyAllWindows()
print("Collecting Samples Complete !!!")
          
