import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time
folder = 'ImagesAttendance'
images = []
classNames = []
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (450,12)
fontScale              = 0.5
fontColor              = (255,255,255)
lineType               = 2

# load pictures from folder and label
for filename in os.listdir(folder):
    path = os.path.join(folder,filename)
    for name in os.listdir(path):
        img = cv2.imread(os.path.join(path,name))
        if img is not None:
            images.append(img)
            classNames.append(filename)

#encoding images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#write name and datetime to file csv
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%d/%m/%Y, %H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

#encoding images
encodeListKnown = findEncodings(images)
print('Encoding Complete')

#connect to webcam
cap = cv2.VideoCapture(0)

width = int(cap.get(3))
height = int(cap.get(4))

#set time each video time in seconds
capture_duration = 900
count = 0
while True:
    img_counter = 0
    count += 1 
    now1 = datetime.now()
    start_time = time.time()
    
    # if the face is unknown, unknown video face will save to 'UnkniwnVideo' folder 
    filename = "Unknown_" + str(now1.strftime('%d %m %Y - %H %M %S' ))  + '.avi'
    path1 = 'UnknownVideo'
    out = cv2.VideoWriter(os.path.join(path1 , filename), cv2.VideoWriter_fourcc(*'XVID'), 1, (640, 480))
    
    # Save video into AllVideo, each 900s = 15 minutes
    filename2 = 'All_' + str(now1.strftime('%d %m %Y - %H %M %S' )) + '.avi'
    path2 = 'AllVideo'
    out2 = cv2.VideoWriter(os.path.join(path2 , filename2), cv2.VideoWriter_fourcc(*'XVID'), 20, (640, 480))
    while( int(time.time() - start_time) < capture_duration ): #after 900s will break 
        success, img = cap.read()
        if success==True:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            now2 = datetime.now()
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
            cv2.putText(img, str(now2), bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            out2.write(img)
            cv2.imshow('Webcam', img)
            cv2.waitKey(1)
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    now = datetime.now()
                    name = classNames[matchIndex].upper()
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    #cv2.putText(img,str(now),(450,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 0),2)
                    markAttendance(name)
                else:
                    now = datetime.now()
                    path = 'Unknown'
                    name = 'Unknown'
                    
                    # if the face is unknown, take picture automatically
                    img_name = 'opencv_frame_%d_%d.png' % (count,img_counter)
                    cv2.imwrite(os.path.join(path , img_name), img)                    
                    img_counter += 1
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    out.write(img)
                    markAttendance(name)
                    print("Warining Unknown Person")
        else:
            break
            
cap.release()
out.release()
out2.release()
cv2.destroyAllWindows()
