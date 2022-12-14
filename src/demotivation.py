from diffusers import StableDiffusionPipeline
from transformers import FSMTForConditionalGeneration, FSMTTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor, \
    AutoTokenizer, pipeline
import torch
from demotivator import Demotivator
import streamlit as st

ENG_TO_RU_MODEL = "facebook/wmt19-en-ru"
RU_TO_ENG_MODEL = "facebook/wmt19-ru-en"
IMAGE_MODEL = "DGSpitzer/Cyberpunk-Anime-Diffusion"
IMAGE_CAPTION_MODEL = "nlpconnect/vit-gpt2-image-captioning"
DETECTION_LANGUAGE_MODEL = "papluca/xlm-roberta-base-language-detection"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEN_KWARGS = {"max_length": 80, "num_beams": 5}


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_pipe_image():
    if DEVICE == "cuda":
        return StableDiffusionPipeline.from_pretrained(IMAGE_MODEL, torch_dtype=torch.float16, revision="fp16",
                                                       requires_safety_checker=False).to(DEVICE)
    else:
        return StableDiffusionPipeline.from_pretrained(IMAGE_MODEL, requires_safety_checker=False).to(DEVICE)


def generate_image(text, steps, height, width):
    pipe_image = get_pipe_image()
    pipe_image.safety_checker = lambda images, clip_input: (images, False)
    image = pipe_image(text, num_inference_steps=steps, height=height, width=width).images[0]
    return image


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_pipe_translate(lang):
    model = RU_TO_ENG_MODEL if lang == "ru" else ENG_TO_RU_MODEL
    return FSMTTokenizer.from_pretrained(model), FSMTForConditionalGeneration.from_pretrained(model)


def translate(text, lang):
    tokenizer, model = get_pipe_translate(lang)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids, **GEN_KWARGS)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


def demotivate(image, watermark="grance", top_text="", bottom_text=""):
    dem = Demotivator(top_text, bottom_text)
    return dem.create(image, watermark=watermark)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_pipe_img_caption():
    return AutoTokenizer.from_pretrained(IMAGE_CAPTION_MODEL), ViTFeatureExtractor.from_pretrained(
        IMAGE_CAPTION_MODEL), VisionEncoderDecoderModel.from_pretrained(IMAGE_CAPTION_MODEL).to(DEVICE)


def get_image_caption(img):
    tokenizer, feature_extractor, model = get_pipe_img_caption()
    images = [img]
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(DEVICE)
    output_ids = model.generate(pixel_values, **GEN_KWARGS)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_pipe_language_detection():
    return pipeline("text-classification", model=DETECTION_LANGUAGE_MODEL)


def detect_language(text):
    pipe = get_pipe_language_detection()
    preds = pipe(text, return_all_scores=True, truncation=True, max_length=128)
    if preds:
        pred = preds[0]
        return {p["label"]: float(p["score"]) for p in pred}
    else:
        return None


def generate_demotivator(text, steps, height, width):
    top_text = text
    lang_prob = detect_language(text)
    if lang_prob is not None:
        language = max(lang_prob, key=lang_prob.get)
    else:
        language = "undefined"

    """it is better to translate the text into English
    because StableDiffusion produces more suitable results from the English text"""

    if language == "ru":
        text = translate(text, language)
    img = generate_image(text, steps, height, width)
    text = get_image_caption(img)
    if language == "ru":
        text = translate(text, "en")
    return demotivate(image=img, top_text=top_text, bottom_text=text)
