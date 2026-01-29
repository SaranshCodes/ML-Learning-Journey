import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

st.title('X-Ray Image Classifier')
img_size =100
pretrained_model_path = 'model_10.h5'
categories= ['Normal', 'Pneumonia']

def load_model():
    return tf.keras.models.load_model(pretrained_model_path)

model = load_model()
print("Model loaded successfully.")

def load_classifier():
    st.subheader("Uplaod a X-Ray Image to detect if it is Normal or Pneumonia")
    file=st.file_uploader(label=" ",type =['jpeg'])
    
    if file!=None:
        img=tf.keras.preprocessing.image.load_img(file, target_size= (img_size,img_size), color_mode="grayscale")
        new_array =tf.keras.preprocessing.image.img_to_array(img)
        new_array = new_array.reshape(-1, img_size,img_size, 1)
        st.image(file)
        st.write('')
        st.write('')
        
        if st.button("Predict"):
            # MAKING PREDICTION
            preds=''
            prediction = model.predict(new_array/255.0)
            print(prediction)
            print(round(prediction[0][0]))
            preds= categories[int(round((prediction[0][0])))] + '-' + str(round(prediction[0][0]*100,2))
            st.write(preds)
def main():
    load_classifier()

if __name__ == '__main__':
    main()
            