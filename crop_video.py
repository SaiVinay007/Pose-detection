import cv2
import argparse
import skvideo.io
ret = True
parser = argparse.ArgumentParser()

parser.add_argument("filename")
parser.add_argument("start_time",type = int)
parser.add_argument("end_time",type = int)

args = parser.parse_args()


#print(args.start_time)
#print(args.end_time)

if(args.filename == None):
    filename = 0
else:
    filename = args.filename
cap =  cv2.VideoCapture(filename)

count = 0
while(ret):

	ret , frame = cap.read()
	count+=1

duration = cap.get(7)/1000

cap.release()

fps = (count-1)/duration

start_frame = int(args.start_time*fps)
end_frame = int(args.end_time*fps)

print(start_frame)
print(end_frame)
print("The output video wil have {} frames".format(end_frame-start_frame))
#out = cv2.VideoWriter("output.WebM",-1,20.0,(640,480))

#cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cap.release()

#for i in range(count):
#    ret,frame = cap.read()
#    if (i> start_frame and i <= end_frame):
        #out.write(frame)

#        cv2.imshow('frame',frame)

videodata = skvideo.io.vread(filename)
video_crop = videodata[start_frame:end_frame]

print(video_crop.shape)




