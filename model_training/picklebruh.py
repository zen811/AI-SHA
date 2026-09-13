import pandas as pd
import numpy as np
import pickle
with open('./AI-SHA/Gesture_model.pkl', 'rb') as file:
    model_pack = pickle.load(file) 

xgb_classifier = model_pack['model']
scalar = model_pack['scaler']
label_enc = model_pack['label_encoder']

def Gesture_checker(raw_data):
    
    scaled_data = scalar.transform(raw_data)
    
    numeric_prediction = xgb_classifier.predict(scaled_data)
    
    text_prediction = label_enc.inverse_transform(numeric_prediction)
    
    return text_prediction[0]

empt=[]
vals=[]
for _ in range(225):
    empt.append(_)  #replace with index of the landmark in order pose left hand right hand
                    #add append to add x y z valus of the index
    vals.append(np.random.rand())

print(vals)
Image_frame_data = pd.DataFrame([vals],columns=([0]*225))
result = Gesture_checker(Image_frame_data)
# print(f"Predicted Gesture: {result}")