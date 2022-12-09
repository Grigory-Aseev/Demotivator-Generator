import streamlit as st
from demotivation import generate_demotivator
from os.path import exists

RESULT_DIR = "..\\images\\result.jpg"


def update_image_section():
    if exists(RESULT_DIR):
        image.empty()
        image.image(RESULT_DIR)
        with open(RESULT_DIR, "rb") as file:
            button.download_button(label="Download", data=file, file_name=RESULT_DIR)


def init_image_section():
    global image, button
    image = st.empty()
    button = st.empty()


def main():
    st.title("Generator demotivators with Streamlit by [grance](https://github.com/Grigory-Aseev)")
    st.sidebar.header("Demotivation")
    st.sidebar.markdown("Generate your own demotivator based on your text!")
    text = st.sidebar.text_input("Enter your sentence", "One-eyed ghoul kaneki-ken")
    steps = st.sidebar.slider("Numbers of inference steps", value=50, min_value=4, max_value=100,
                              help="More steps usually lead to a higher quality of image and slower inference.")
    height = st.sidebar.slider("Height", value=512, min_value=128, max_value=512,
                               help="After receiving the image, the result of the neural network model will be resized to the specified height.")
    width = st.sidebar.slider("Width", value=512, min_value=128, max_value=512,
                              help="After receiving the image, the result of the neural network model will be resized to the specified width.")
    get_button = st.sidebar.button("Generate image")
    init_image_section()
    update_image_section()
    try:
        if get_button:
            generate_demotivator(text, steps, height, width, RESULT_DIR)
            update_image_section()
            st.sidebar.success("Success!")
    except Exception as e:
        st.sidebar.error("Failed with error: {}".format(e))


if __name__ == "__main__":
    main()
