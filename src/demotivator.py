from PIL import Image, ImageDraw, ImageFont, ImageOps

FONT_COLOR = "white"
FILL_COLOR = "black"
FONT_NAME = "times.ttf"


def get_font(text, size, width):
    while True:
        font = ImageFont.truetype(font=FONT_NAME, size=size, encoding="UTF-8")
        text_width = font.getsize(text)[0]
        if text_width < width + 230:
            return font
        size -= 1


def draw_text(drawer, text, width, size, padding):
    font = get_font(text, size, width)
    size = drawer.textsize(text, font=font)
    drawer.text(((1280 - size[0]) / 2, padding), text, fill=FONT_COLOR, font=font)


class Demotivator:
    def __init__(self, top_text="", bottom_text=""):
        self._top_text = top_text
        self._bottom_text = bottom_text

    def create(self, file: Image.Image, watermark=None, top_size=80, bottom_size=60) -> Image.Image:
        """
        Create a template for a demotivator
        Inserting a photo into a frame
        """

        img = Image.new("RGB", (1280, 1024), color=FILL_COLOR)
        img_border = Image.new("RGB", (1060, 720), color="#000000")
        border = ImageOps.expand(img_border, border=2, fill="#ffffff")
        user_img = file.convert("RGBA").resize((1050, 710))
        (width, height) = user_img.size
        img.paste(border, (111, 96))
        img.paste(user_img, (118, 103))
        drawer = ImageDraw.Draw(img)

        """
        Choosing the optimal font size
        Adding text to the demotivator template
        """

        draw_text(drawer, self._top_text, width, top_size, 840)
        draw_text(drawer, self._bottom_text, width, bottom_size, 930)

        if watermark is not None:
            (width, height) = img.size
            idraw = ImageDraw.Draw(img)
            idraw.line((1000 - len(watermark) * 5, 817, 1008 + len(watermark) * 5, 817), fill=0, width=4)
            font = ImageFont.truetype(font=FONT_NAME, size=20, encoding="UTF-8")
            size = idraw.textsize(watermark.lower(), font=font)
            idraw.text((((width + 729) - size[0]) / 2, ((height - 192) - size[1])),
                       watermark.lower(), font=font)

        return img
