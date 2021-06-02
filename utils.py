import numpy as np
import pyautogui as gui
import cv2

def get_window_points():
    _ = input("Upper Left: Place the cursor in the upper left corner of the area you want and press the Enter key.")
    x1, y1 = gui.position()
    print(f'Upper Left = x:{x1} y:{y1}')
    
    _ = input("Lower Right: Place the cursor in the lower right corner of the area you want and press the Enter key.")
    x2, y2 = gui.position()
    print(f'Lower Right = x:{x2} y:{y2}')
        
    return x1,y1,x2,y2


def choose_main_face(faces):

    confs=[]
    for i in range(len(faces)):
        confs.append(faces[i]["confidence"])
    face_ind = np.argmax(confs)
    
    return face_ind




