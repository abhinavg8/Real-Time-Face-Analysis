import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

model_pred = tf.keras.models.load_model('CKmodel.h5')
model_pred.compile(loss='categorical_crossentropy',
             optimizer='adam',
                metrics=['accuracy'])

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

exp = ['Angry','Happy','Sad','Surprise']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']
gender_list = ['Male', 'Female']

def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return(age_net, gender_net)

age_net , gender_net = load_caffe_models()

def detect_face(img):
    flag = False
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img) 
    
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 2)
        crop_img = face_img[y:y+h,x:x+w]
        flag = True
        crop = crop_img.copy()
        crop = cv2.resize(crop,(48,48))
        crop = crop.reshape(1,48,48,3)
        crop = crop.astype('float64')
        pred = model_pred.predict(crop)
        
        n1 = int(np.random.randint(0,256,1))
        n2 = int(np.random.randint(0,256,1))
        n3 = int(np.random.randint(0,256,1))

        text = exp[pred.argmax()]
        cv2.putText(face_img,text,(x,y-30),fontFace = cv2.FONT_ITALIC,
                        fontScale = 2,color=[n1,n2,n3],thickness=5) 
        
        
        crop_img = cv2.resize(crop_img,(227,227))
        blob = cv2.dnn.blobFromImage(crop_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        text = 'Age : ' + age
        cv2.putText(face_img,text,(x-20,y+h+50),fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1,color=[0,220,0],thickness=3)
        
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        text = 'Gender : '+ gender
        cv2.putText(face_img,text,(x+w+2,y+int(h/2)+20),fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1,color=[0,0,255],thickness=3)

    if(flag):
        return face_img
    else:
        return face_img
    

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('video_out.mp4', cv2.VideoWriter_fourcc(*'XVID'),15, (width, height))

cap = cv2.VideoCapture(0) 

while True: 
    
    ret, frame = cap.read() 
    
    frame = detect_face(frame)
    
    writer.write(frame)

    cv2.imshow('Video Face Detection', frame) 
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
        
cap.release() 
writer.release()
cv2.destroyAllWindows()