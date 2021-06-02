from tensorflow.keras.models import model_from_json
import cv2
import numpy as np
import csv

class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        # print("Model loaded from disk")
        # self.loaded_model.summary()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
    

def get_emotion(model,img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(48,48))
    img = img[np.newaxis, :, :, np.newaxis]
    
    # img = img.asarray("float32")/255. # 正規化必要??
    emotion = model.predict_emotion(img)
    
    return emotion
    
    
def write_emotion(PATH,contents):
    # Write down features to file.
    with open(PATH, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(contents)  