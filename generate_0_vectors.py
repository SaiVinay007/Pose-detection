import argparse
import logging
import time
import skvideo.io
import cv2
import numpy as np
import os
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

if __name__ == '__main__':
    ret_val = True
    cv2.namedWindow('tf-pose-estimation result',cv2.WINDOW_NORMAL)
    cv2.namedWindow("original",cv2.WINDOW_NORMAL)
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', default=0)
    #parser.add_argument('--start_time',type = int,default = 0)
    #parser.add_argument('--end_time',type = int, required = True)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    count = 0
    
    list_videos = os.listdir("/home/ayush/Downloads/zero")
    list_videos = ["10.webm"]
    for j, video_name in enumerate(list_videos):
        cam = cv2.VideoCapture(0)
        cv2.namedWindow('tf-pose-estimation result',cv2.WINDOW_NORMAL)
        ret_val , image = cam.read()
        for i in range(2000):
            ret_val,image = cam.read()
        #duration = cam.get(7)/1000
        if(ret_val == False):
            print("The video {} doesn't exist\n\n".format(video_name))
            continue
        count_1 = 0
        count = 0
        i = 0
        while(i<100):
            #print(i)
            ret_val, image = cam.read()
            if(ret_val == False):
                i = 101
                print("The video file :{} has been processed , moving on to the next video\n".format(video_name))
                continue     
            #print(image.shape)
            #frame = image.copy()

            #logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

            #logger.debug('postprocess+')
            _,centers = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            #angles_of_body = e.get_angles(humans,centers)

            #angle_vector = e.get_final_angle_vector(angles_of_body)
            #print(angle_vector)
            #print("\n")
            #if(len(angle_vector) ==1):   ## len(angle_vector) represents no. of humans detected in the frame
            #    count+=1
            #    if(i==0):
            #        out_vector = angle_vector[1]
            #    else:
            #        out_vector = np.hstack((out_vector,angle_vector[1]))

            #print(out_vector)
        
            #if(count>=40):
            #    logger.debug("Video name :{} , slice_number :{} is now saved\n\n".format(video_name,count_1))
            #    print("\n The numpy array saved has the shape of {} \n".format(out_vector.T.shape))
                #np.save("./dataset_numpy/One/class_1"+"video_"+video_name+"_"+str(count_1)+".npy",out_vector.T)
            #    count_1+=1
            #    count = 0
            #    i=0
            #    continue
            #logger.debug('show+')
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            #cv2.imshow("original" , frame)
    
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
            #logger.debug('finished+')
            #i+=1

    #cv2.destroyAllWindows()

        #if(count<40):
        #    print("This video {}  won't work , try something else , count has value of {} ".format(filename,count))
        #else:
        #    print("This video worked ")
        #    np.save("./dataset_numpy/Zero/video_"+str(filenumber)+".npy",out_vector)
        #    filenumber +=1
        #print(out_vector.shape)

