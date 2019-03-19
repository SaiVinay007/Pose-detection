import argparse
import logging
import time
import skvideo.io
import cv2
import numpy as np
import os
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from keras.models import load_model


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
    #cv2.namedWindow('tf-pose-estimation result',cv2.WINDOW_NORMAL)
    #cv2.namedWindow("original",cv2.WINDOW_NORMAL)
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', default=0,type = int)
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
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('tf-pose-estimation result',cv2.WINDOW_NORMAL)
    #ret_val , image = cam.read()
    count = 0
    i = 0
    out_vector = {}
    model_LSTM = load_model("./dataset_numpy/first_model.h5")

    print("model loaded\n\n")
    our_counter = 0
    accuracy_counter = 0
    counter_1 = 0
    while(ret_val):
        #print(i)
        ret_val, image = cam.read()
        if(ret_val == False):
            break
        #print(image.shape)
        #frame = image.copy()

        #logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        #logger.debug('postprocess+')

        image,centers = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        angles_of_body = e.get_angles(humans,centers)

        angle_vector = e.get_final_angle_vector(angles_of_body)

        #print("\n\nThe no. of humans is {}\n\n".format(len(angle_vector)))
        for k in range(len(angle_vector)):   ## len(angle_vector) represents no. of humans detected in the frame
            #print("our_counter = {}".format(our_counter))
            if(our_counter ==40):
                break
            if(our_counter == 0 ):
                out_vector[k] = angle_vector[k+1]
            elif(k >= len(out_vector)):
                out_vector[k] = angle_vector[k+1]
            else:
                out_vector[k] = np.hstack((out_vector[k],angle_vector[k+1]))
            our_counter+=1
        
        if(our_counter == 40):
            for k in range(len(out_vector)):
                out_vector[k] = out_vector[k].reshape((out_vector[k].shape[1],out_vector[k].shape[0]))
            for k in range(len(out_vector)):
                print("\nour_vector[{}]'s shape is {} \n".format(k,out_vector[k].shape))
            for k in range(len(out_vector)):
                out_vector[k] = out_vector[k].reshape((1,out_vector[k].shape[0],out_vector[k].shape[1]))
                if(out_vector[k].shape[1] == 40):
                    accuracy_counter +=1
                    y = model_LSTM.predict(out_vector[k])
                    print("\n\n y :{}\n".format(y))
                    if(y[0][1] < 0.7):
                        y[0][1] = 0
                    class_pred = np.argmax(y)
                    
                    if(class_pred ==1):
                        counter_1+=1
                    print("class_pred : {} for k = {}".format(class_pred,k))
            our_counter = 0
            out_vector = {}
            #print("\n\n\nForty frames are read and hopefully predicted")
            #print("\n\nRefreshing the our_counter and out_vector variables now \n\n")

        
        

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
    
    cv2.destroyAllWindows()
    class_1_pred = (counter_1)/(accuracy_counter)

    print("\n\n {}".format(class_1_pred))
    


