import streamlit as st
from PIL import Image
from keras_preprocessing.image import img_to_array
from src.trainer import model1
import numpy as np
from skimage.transform import resize


def predict_image(image_input, classifier):
    classifier = model1
    x = img_to_array(image_input)
    predict_modified = image.x
    predict_modified = predict_modified / 255
    predict_modified = np.expand_dims(predict_modified, axis=0)
    result = classifier.predict(predict_modified)
    return result


def main():

    st.title('-- Cat Dog Image Classification App --')
    st.write('\n')
    st.header('Input Image')
    st.text('')
    st.sidebar.title('Upload Image')

    # Disabling warning
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # loading image cat or dog
    image_load = st.sidebar.file_uploader("Cat and Dog image ", type=['png', 'jpg', 'jpeg'],
                                          accept_multiple_files=False)

    if image_load is not None:
        u_img = Image.open(image_load)
        st.image(u_img, 'Uploaded Image', use_column_width=True)
        st.write('')

    col1, col2 = st.beta_columns(2)
    with col1:
        pred_button = st.button("Predict")

    with col2:
        if pred_button:
            predicted = predict_image(image_load, model1)
            if predicted[0][0] >= 0.5:
                prediction = 'dog'
                probability = predicted[0][0]
                print("probability = " + str(probability))
            else:
                prediction = 'cat'
                probability = 1 - predicted[0][0]
                print("probability = " + str(probability))
                print("Prediction = " + prediction)


if __name__ == '__main__':
    main()
