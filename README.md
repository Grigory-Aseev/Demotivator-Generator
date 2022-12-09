# Demotivator-Generator

## About
This project has implemented an application written in streamlit, which generates demotivators based on your sentence. It is recommended to use CUDA to run, because the process is slower and less accurate on the CPU.

## Usage
To test the performance of the project, you need to 
1. Clone the repository
```
git clone https://github.com/Grigory-Aseev/Demotivator-Generator.git
```
2. Install all the necessary dependencies:
```
pip install -r requirements.txt
```
3. Install desired version [torch](https://pytorch.org/)

4. Log in hugging face and run the command: 
```
huggingface-cli login
```

Almost everything is done. It remains to run the application using the command:
```
streamlit run src/main.py
```

## References to models:
[Model translating text into image in cyberpunk anime style](https://huggingface.co/DGSpitzer/Cyberpunk-Anime-Diffusion)

[Model translating text from English to Russian](https://huggingface.co/facebook/wmt19-en-ru)

[Model describing the image](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)

[Model that recognizes the language in which the text is written](https://huggingface.co/papluca/xlm-roberta-base-language-detection)