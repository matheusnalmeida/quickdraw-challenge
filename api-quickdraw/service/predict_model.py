# Imports
import numpy as np
import cv2
from keras.models import load_model

model = load_model('model/model_cnn.h5')

def predict_image_class(image_path: str) -> None:
    classes = [
    'airplane',
    'banana',  
    'bee',
    'coffee cup',
    'crab',
    'guitar',
    'hamburger',
    'rabbit',
    'truck',
    'umbrella']
    inputimg = cv2.imread(image_path)
    imagem = cv2.resize(inputimg,(28, 28), interpolation=cv2.INTER_AREA)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    inputd = np.array(imagem)
    inputd = inputd.astype(np.float32) / 255.0
    inputd = inputd.reshape((28, 28, 1))

    inp = np.array([inputd])
    predicao = model.predict(inp)

    predicoes_values = list(predicao[0])
    max_index = predicoes_values.index(max(predicoes_values))
    return classes[max_index]