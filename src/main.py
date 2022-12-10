import streamlit as st
from demotivation import generate_demotivator
from PIL import Image
from io import BytesIO

IMAGE_EXAMPLE_DIR = "..\\images\\result.jpg"


def update_image_section(img):
    st.image(img)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    st.download_button(label="Download", data=byte_im, file_name=IMAGE_EXAMPLE_DIR)


def init_image_section():
    img = Image.open(IMAGE_EXAMPLE_DIR)
    st.session_state.image = img


def main():
    st.title("Generator demotivators with Streamlit by [grance](https://github.com/Grigory-Aseev)")
    st.sidebar.header("Demotivation")
    st.sidebar.markdown("Generate your own demotivator based on your text!")
    text = st.sidebar.text_input("Enter your sentence", "One-eyed ghoul kaneki-ken", max_chars=50)
    steps = st.sidebar.slider("Numbers of inference steps", value=50, min_value=4, max_value=100,
                              help="More steps usually lead to a higher quality of image and slower inference.")
    height = st.sidebar.slider("Height", value=512, min_value=128, max_value=512,
                               help="After receiving the image, the result of the neural network model will be resized to the specified height.")
    width = st.sidebar.slider("Width", value=512, min_value=128, max_value=512,
                              help="After receiving the image, the result of the neural network model will be resized to the specified width.")

    if "image" not in st.session_state:
        init_image_section()

    get_button = st.sidebar.button("Generate image")
    try:
        if get_button:
            st.session_state.image = generate_demotivator(text, steps, height, width)
            st.sidebar.success("Success!")
    except Exception as e:
        st.sidebar.error("Failed with error: {}".format(e))

    update_image_section(st.session_state.image)


if __name__ == "__main__":
    main()
