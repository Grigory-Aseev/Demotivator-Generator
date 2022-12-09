from diffusers import StableDiffusionPipeline
from transformers import FSMTForConditionalGeneration, FSMTTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor, \
    AutoTokenizer, pipeline
import torch
from simpledemotivators import Demotivator
import streamlit as st
from PIL import Image

ENG_TO_RU_MODEL = "facebook/wmt19-en-ru"
IMAGE_MODEL = "DGSpitzer/Cyberpunk-Anime-Diffusion"
IMAGE_CAPTION_MODEL = "nlpconnect/vit-gpt2-image-captioning"
DETECTION_LANGUAGE_MODEL = "papluca/xlm-roberta-base-language-detection"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_pipe_image():
    if DEVICE == "cuda":
        return StableDiffusionPipeline.from_pretrained(IMAGE_MODEL, torch_dtype=torch.float16, revision="fp16",
                                                       requires_safety_checker=False).to(DEVICE)
    else:
        return StableDiffusionPipeline.from_pretrained(IMAGE_MODEL, requires_safety_checker=False).to(DEVICE)


def generate_image(text, path, steps, height, width):
    pipe_image = get_pipe_image()
    pipe_image.safety_checker = lambda images, clip_input: (images, False)
    image = pipe_image(text, num_inference_steps=steps, height=height, width=width).images[0]
    image.save(path)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_pipe_translate():
    return FSMTTokenizer.from_pretrained(ENG_TO_RU_MODEL), FSMTForConditionalGeneration.from_pretrained(ENG_TO_RU_MODEL)


def translate(text):
    tokenizer, model = get_pipe_translate()
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


def demotivate(image_dir, watermark="grance", top_text="", bottom_text=""):
    dem = Demotivator(top_text, bottom_text)
    dem.create(image_dir, result_filename=image_dir, watermark=watermark)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_pipe_img_caption():
    return AutoTokenizer.from_pretrained(IMAGE_CAPTION_MODEL), ViTFeatureExtractor.from_pretrained(
        IMAGE_CAPTION_MODEL), VisionEncoderDecoderModel.from_pretrained(IMAGE_CAPTION_MODEL).to(DEVICE)


def get_image_caption(image_paths):
    tokenizer, feature_extractor, model = get_pipe_img_caption()
    gen_kwargs = {"max_length": 80, "num_beams": 8}
    images = [Image.open(image_paths)]
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(DEVICE)
    output_ids = model.generate(pixel_values, **gen_kwargs)
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


def generate_demotivator(text, steps, height, width, path):
    top_text = text[:30 if len(text) >= 30 else len(text)]
    lang_prob = detect_language(text)
    if lang_prob is not None:
        language = max(lang_prob, key=lang_prob.get)
    else:
        language = "undefined"
    generate_image(text, path, steps, height, width)
    text = get_image_caption(path)
    if language == "ru":
        text = translate(text)
    demotivate(image_dir=path, top_text=top_text, bottom_text=text)
