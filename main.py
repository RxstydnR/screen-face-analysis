import os
import time
import cv2
import numpy as np
import datetime
import csv
import argparse

import mss
import pyautogui as gui
from mtcnn import MTCNN

from tensorflow.keras.models import load_model

from model import facenet
from utils import get_window_points, choose_main_face
from analysis import FacialExpressionModel,get_emotion, write_emotion



"""
    # Window画面領域の指定
    x1,y1,x2,y2 = get_window_points()

    # The screen part to capture
    region = {
        "top": y1,
        "left": x1,
        "width": abs(x2-x1),
        "height": abs(y2-y1),
        "mon": 1, # Main PC screen
    }
    ratio = 1 # 1/2 = 0.5倍

    xml_dir = "./haarcascade"
    cascade_path = "haarcascade_frontalface_alt.xml"
    # cascade_path = "haarcascade_frontalface_alt2.xml"
    # cascade_path = "haarcascade_frontalface_alt_tree.xml"
    # cascade_path = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(os.path.join(xml_dir,cascade_path))

    face_width = 80
    face_height = 80

    SAVE_DIR = "Results"
    frame_SAVEPATH  = os.path.join(SAVE_DIR,"frames")
    face_SAVEPATH   = os.path.join(SAVE_DIR,"faces")
    result_SAVEPATH = os.path.join(SAVE_DIR,"results")
    os.makedirs(frame_SAVEPATH, exist_ok=False)
    os.makedirs(face_SAVEPATH, exist_ok=False)
    os.makedirs(result_SAVEPATH, exist_ok=False)

    csv_filename = os.path.join(SAVE_DIR,"face_features.csv")
    csv_columns = [
        "",
        "",
        "",
        "",
    ]

    # ファイル，1行目(カラム)の作成
    with open(csv_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['time','x','y','z'])

    # model predictに1枚では0.05〜0.01秒かかる
    # 1秒ごとに計測待ちフラグを立てる。
    # 計測された時刻から1秒明ける??

    while(True):

        start = datetime.datetime.now()
        file_name = start.strftime("%Y_%m%d_%H-%M-%S-%f")+".jpg"  # "%Y年%m月%d日%H時%M分%S秒%f"

        # Capture Frame
        with mss.mss() as sct:

            # Grab the data
            img = sct.grab(region)

            # Save to the picture file
            frame = np.array(img)

        h,w,c = frame.shape
        frame = cv2.resize(frame,(int(w//ratio),int(h//ratio)))
        cv2.imwrite(os.path.join(frame_SAVEPATH,file_name), frame)

        # Detect face
        facerects,rejectLevels,_ = cascade.detectMultiScale3(frame, scaleFactor=1.1, minNeighbors=5, minSize=(60,60),outputRejectLevels=1)
        
            scaleFactor: 
                How much the image size is reduced at each image scale. 
                This value is used to create the scale pyramid. 
                To detect faces at multiple scales in the image (some faces may be closer to the foreground, 
                    and thus be larger, other faces may be smaller and in the background, thus the usage of varying scales). 
                A value of 1.05 indicates that we are reducing the size of the image by 5% at each level in the pyramid.

            minNeighbors: 
                How many neighbors each window should have for the area in the window to be considered a face. 
                The cascade classifier will detect multiple windows around a face. 
                This parameter controls how many rectangles (neighbors) need to be detected for the window to be labeled a face.

            minSize: 
                A tuple of width and height (in pixels) indicating the window’s minimum size. 
                Bounding boxes smaller than this size are ignored. 
                It is a good idea to start with (30, 30) and fine-tune from there.
        

        # 顔を1つ以上認識していた場合
        if len(rejectLevels)>0:
            facerest_ind = np.argmax(rejectLevels)

            (x, y, w, h) = facerects[facerest_ind]

            # w,h は固定しても良いかも???
            center_x, center_y = x+int(w//2), y+int(y//2)

            x = center_x - int(face_width//2)
            y = center_y - int(face_height//2)

            face_img = frame[y:y+face_height,x:x+face_width,:]        
            # cv2.imwrite(os.path.join(face_SAVEPATH,file_name), face_img)

            # draw the face bounding box on the image
            cv2.rectangle(frame, (x, y), (x + face_width, y + face_height), (0, 255, 0), 1)

            # Face feature extraction with FaceNet.
            face_img = face_img[:,:,::-1]
            features = facenet_extract(face_img)
            print_feature_info(features)

            # Write down features to file.
            with open(os.path.join(SAVE_DIR,"face_features.csv"), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(features)  # time と featureをまとめて保存しないといけない
                # ファイル，1行目(カラム)の作成

            # print(f"{time.time()-start} [sec]")

        # Display the result frame
        cv2.imshow('openCV caputure',frame)
        # cv2.imwrite(os.path.join(result_SAVEPATH,file_name), frame)

        # Quit when 'q' key pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # When everything done, release the capture
    cv2.destroyAllWindows()
"""
        

def main():
    
    while(True):

        atTime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
        img_name = atTime+".jpg"

        # Capture Frame
        with mss.mss() as sct:

            # Grab the data
            img = sct.grab(region)
            # Remove alpha axis
            frame = np.array(img)[:,:,:-1]

        # Reshape frame
        h,w,c = frame.shape
        frame = cv2.resize(frame,(int(w//ratio),int(h//ratio)))

        # Face Detection
        faces = detector.detect_faces(frame)

        try:
            # When detected at least one face
            if len(faces)>0:

                # Extract only one face
                face_ind = choose_main_face(faces)
                face_info = faces[face_ind]

                # Make face bounding-box
                (x, y, w, h) = face_info["box"]

                center_x, center_y = x+int(w//2), y+int(h//2)
                length = max(w,h)

                x = center_x - int(length//2)
                y = center_y - int(length//2)
                w = length
                h = length 

                # Extract face image
                face_img = frame[y:y+h,x:x+w,:].copy()

                # Draw the face bounding box on the image
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

                # Face feature extraction with a feature extractor.
                emotion = get_emotion(model,face_img)
                write_emotion(PATH=csv_filename, contents=[atTime,emotion])

                # Save frames
                cv2.imwrite(os.path.join(face_SAVEPATH,img_name), face_img)
                cv2.imwrite(os.path.join(result_SAVEPATH,img_name), frame)
        except:
            print("Error happend.")


        # Display the result frame
        cv2.imshow('openCV caputure',frame)
        # cv2.moveWindow('openCV caputure', 100, 200) # gui.size()
        cv2.imwrite(os.path.join(frame_SAVEPATH,img_name), frame)

        # Quit when 'q' key pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Face Detection and Analysis on Real-time Screen Images') 
    parser.add_argument('--PATH_SAVEDIR', type=str, default=".", help='path to save directory') 
    parser.add_argument('--ratio', type=float, default=2.0, help='image shrink ratio (default: 1/2)')   
    args = parser.parse_args() 

    # Get location of window
    x1,y1,x2,y2 = get_window_points()

    # The screen part to capture
    region = {
        "top": y1,
        "left": x1,
        "width": abs(x2-x1),
        "height": abs(y2-y1),
        "mon": 1, # Main PC screen
    }

    # Image shrink ratio
    ratio = args.ratio

    # Make save folders
    start = datetime.datetime.now()
    SAVE_DIR = os.path.join(args.PATH_SAVEDIR,start.strftime("%Y-%m-%d_%H:%M:%S"))
    
    frame_SAVEPATH  = os.path.join(SAVE_DIR,"frames")
    face_SAVEPATH   = os.path.join(SAVE_DIR,"faces")
    result_SAVEPATH = os.path.join(SAVE_DIR,"results")
    os.makedirs(frame_SAVEPATH, exist_ok=False)
    os.makedirs(face_SAVEPATH, exist_ok=False)
    os.makedirs(result_SAVEPATH, exist_ok=False)

    # Make .csv file
    csv_filename = os.path.join(SAVE_DIR,"face_features.csv")
    csv_columns = [
        "datetime",
        "emotions",
    ]
    with open(csv_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)

    # Face Detection Model
    detector = MTCNN()

    # Facial Feature Extraction Model
    model = FacialExpressionModel("FER/model.json", "FER/weights.h5")

    # model = facenet()
    # model.load_weights('./OpenFace/model/nn4.small2.v1.h5')

    main()

