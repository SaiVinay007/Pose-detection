import os
list_video_name = os.listdir("../dataset/Zero/")

for i in range(len(list_video_name)):
    fil = open("Zero_txt.txt","a")
    fil.write(list_video_name[i])
    fil.write(",")
    fil.write("0")
    fil.write("\n")

fil.close()
