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

