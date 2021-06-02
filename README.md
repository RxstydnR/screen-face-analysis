# Face Detection and Analysis on Real-time Screen Images.

This repository is a simple implementation mainly using OpenCV and Keras for real-time facial expression analysis.
It detects a person's face on a PC screen and predicts the emotion of that face. 



<img src="/Users/macbookpro/Desktop/Facenet/demo.png" alt="demo" style="zoom:40%;" />



## Quick start

Before run this code, install all necessary package. See "Setup" below.

```shell
python main.py
```



## Brief introduction to the algorithm

### Flow

1. Get a screen location. (pyautogui)
2. Take a screen-shots of the screen. (mss)
3. Face Detection. (OpenCV or Keras)
4. Face Analysis. (Keras)
5. => loop 2 ~ 5

###1. Get a screen location.

At first, you will be asked to specify the location of the screen shot. Place the cursor on the upper left and lower right corner of the window and press Enter key.

Note that, when you run this file `main.py` from Terminal, Terminal have to be main active window to Enter key. That is after moving the cursor, press Enter key without clicking on any part outside the terminal area.

Multiple displays are also available. It is possible to specify the location of the not main display. 

### 2. Take a screen-shots of the screen.

[mss](https://github.com/BoboTiG/python-mss) is an ultra fast screenshots module in pure python using ctypes. The screen images are caputured using mss.

### 3. Face Detection.

[MTCNN](https://github.com/ipazc/mtcnn) provides the MTCNN face detector which is a Keras implementation. The model's accuracy is better than [OpenCV's CascadeClassifier](https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/) but FPS performance is inferior.

This model outputs face bounding box and key points info. (more details in [mtcvv github](https://github.com/ipazc/mtcnn))

### 4. Face Analysis. 

Use The CNN pre-trained model provided [here](https://github.com/mayurmadnani/fer) to analysis a facial emotion. This Facial Expression Recognition model is inputted a gray image of the (48,48,1) size and outputs the predicted emotion.

The results of that prediction are saved in a csv file.



## Setup

**Getting Window Location**

```shell
pip install pyautogui
```

**Fast Screen-shots**

```shell
pip install mss
```
**Face Detection**

```python
pip install mtcnn
```

**Face Analysis**

Download this [FER - Facial Expression Recognition](https://github.com/mayurmadnani/fer) to get  `model.json` and `weights.h5` to use the pre-trained facial expressino recognition model. Save them to folder named "FER".

**OpenCV Face Recognition** (if you use, but need to change code.)

To use the OpenCV Face Recognition , necessary to download opencv xml file [here](https://github.com/opencv/opencv/tree/master/data/haarcascades). 



## Future works

### Facial Action Coding System [wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units)

> Facial Action Coding System (FACS) is a system to taxonomize human facial movements by their appearance on the face. Movements of individual facial muscles are encoded by FACS from slight different instant changes in facial appearance. Using FACS it is possible to code nearly any anatomically possible facial expression, deconstructing it into the specific Action Units (AU) that produced the expression. It is a common standard to objectively describe facial expressions.

Emotional analysis alone is not enough to analyze facial expressions, and even deeper information can be obtained by analyzing **Facial Action Units**, but an implementation that can easily run the Facial Action Coding System has not been provided [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) provides only matlab codes). 

**When easily available Facial Action Coding System in python is come out somewhere, I will add Facial Action Units Analysis part using some facial feature extractor such as FaceNet .**



## Application Window Tracking

If your computer OS is Windows OS, Application Window Tracking is available with [*winAPI*](https://pypi.org/project/pywin32/) & [*ahk*](https://pypi.org/project/ahk/). 

- [Pythonでウィンドウのリアルタイム描画と特徴検出](https://qiita.com/meznat/items/fa871ab88310b4198a18)

On the other hand, Mac OS doesn't support any API for Application Window Tracking.

#####How to do Window Trancking on Mac, but still no way.
  - [numpy - MacOS: Using python to capture screenshots of a specific window - Stack Overflow](https://stackoverflow.com/questions/62707662/macos-using-python-to-capture-screenshots-of-a-specific-window)
  - [macos - How to get active window title using Python in Mac? - Stack Overflow](https://stackoverflow.com/questions/28815863/how-to-get-active-window-title-using-python-in-mac/37368813)
  - [Obtain list of all window titles on macOS from a Python script - Stack Overflow](https://stackoverflow.com/questions/53237278/obtain-list-of-all-window-titles-on-macos-from-a-python-script)
  - [asweigart/PyGetWindow: A simple, cross-platform module for obtaining GUI information on applications' windows.](https://github.com/asweigart/pygetwindow)
  - [macos - How to get active window title using Python in Mac? - Stack Overflow](https://stackoverflow.com/questions/28815863/how-to-get-active-window-title-using-python-in-mac/37368813)
  - [Python and Applescript path of running application using a partial app name? - Stack Overflow](https://stackoverflow.com/questions/66810125/python-and-applescript-path-of-running-application-using-a-partial-app-name)
  - [objective c - Finding the Current Active Window in Mac OS X using Python - Stack Overflow](https://stackoverflow.com/questions/373020/finding-the-current-active-window-in-mac-os-x-using-python/25214024)
  - [Obtain Active window using Python - Stack Overflow](https://stackoverflow.com/questions/10266281/obtain-active-window-using-python)

