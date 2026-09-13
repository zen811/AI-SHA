from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import LabelEncoder

import xgboost as xgb 
import pandas as pd
import pickle



df = pd.read_csv('./AI-SHA/model_training/gestures/handgestures.csv')
x = df.drop(df.columns[-1], axis=1)
y = df.iloc[:, -1:]
print(df)



X_train, X_test, Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=69)
X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.5,random_state=69)


label_enc= LabelEncoder()
Y_train =label_enc.fit_transform(Y_train)
Y_val=label_enc.transform(Y_val)
Y_test=label_enc.transform(Y_test)


scalar = StandardScaler()
X_train= scalar.fit_transform(X_train)
X_val= scalar.transform(X_val)
X_test=scalar.transform(X_test)

xgb_classifier=xgb.XGBClassifier(random_state=69)
xgb_classifier.fit(X_train,Y_train)
Y_pred=xgb_classifier.predict(X_val)
test_accuracy= accuracy_score(Y_val,Y_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")
print("\n Classification Report: \n", classification_report(Y_val,Y_pred))
print("Confusion Matrix: \n", confusion_matrix(Y_val,Y_pred))


model_pack = {'model': xgb_classifier,'scaler': scalar,'label_encoder': label_enc}


with open('Gesture_model.pkl', 'wb') as file:
    pickle.dump(model_pack, file)
    

