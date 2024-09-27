import numpy as np
from PIL import Image, ImageDraw, ImageFont


def remap_img(img: np.ndarray, a: int, b: int, c: int, d: int) -> np.ndarray:
    if a == b:
        return img
    img = img.astype(float)
    remapped_arr = c + (img - a) * (d - c) / (b - a)
    return remapped_arr.astype(np.uint8)


def generate_empty_img(width: int, height: int) -> np.ndarray:
    # create canvas
    canvas = Image.new('RGB', (width, height), (14,17,23))
    draw = ImageDraw.Draw(canvas)
    fnt = ImageFont.truetype("fonts/Deng.ttf", 40)
    content = 'NO DATA'
    _, _, w, h = draw.textbbox((0, 0), content, font=fnt)
    draw.text((width // 2 - w // 2, height // 2 - h // 2), content, font=fnt, fill='gray', align='center')
    return np.array(canvas)
