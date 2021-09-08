import PIL.Image
import streamlit as st


def main():
    # Title of
    st.title('-- Cat Dog Image Classification App --')
    st.write('\n')
    st.header('Input Image')
    st.text('')

    st.sidebar.title('Upload Image')

    # Disabling warning
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # loading image cat or dog
    image_load = st.sidebar.file_uploader('Cat or Dog Image', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

    if st.sidebar.button(" Click Here to Classify "):
        if image_load is not None:
            image = PIL.Image.open(image_load).convert('RGB')
            st.image(image, use_column_width=True)
            (W, H) = image.size
            st.write('Image size : weight {} - height {}'.format(W, H))
            st.write('')


if __name__ == '__main__':
    main()
