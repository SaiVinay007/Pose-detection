import cv2
import argparse
ret = True
parser = argparse.ArgumentParser()

parser.add_argument("filename")
args = parser.parse_args()

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

print(duration)

print(count/duration)

print("Total frames = {}".format(count))
    
    
