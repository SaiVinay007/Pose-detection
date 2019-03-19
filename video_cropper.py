import os
import cv2
import glob
video = "11.webm"
cap = cv2.VideoCapture(video)
duration = cap.get(7)/1000
time = 0
i = 0
while(time<duration):
    if(time<60):
        time_str = "00:00:" + str(time)
    else:
        time_str = "00:" + str(int(time/60)) +":" + str(int(time%60))
    os.system("ffmpeg -i " + video +" -ss " + time_str +" -t 00:00:6 -vcodec copy -acodec copy slice_" + str(i)+".webm")
    i+=1
    time+=7


    





