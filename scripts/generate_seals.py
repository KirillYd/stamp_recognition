import os
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json


def generate_text_with_even_spaces(total_length, min_letters, max_letters, min_word_length=4, max_word_length=10):
    num_letters = random.randint(min_letters, max_letters)
    words = []
    current_letters = 0

    while current_letters < num_letters:
        remaining_letters = num_letters - current_letters
        word_length = random.randint(min_word_length, max_word_length)
        if remaining_letters < min_word_length:
            word_length = remaining_letters
        word_length = min(word_length, remaining_letters)
        word = ''.join(random.choices([chr(code) for code in range(ord('А'), ord('Я') + 1)], k=word_length))
        words.append(word)
        current_letters += word_length

    num_spaces = total_length - num_letters
    num_gaps = len(words) - 1

    base_spaces = num_spaces // num_gaps if num_gaps > 0 else 0
    extra_spaces = num_spaces % num_gaps if num_gaps > 0 else 0

    text_parts = []
    for i in range(len(words) - 1):
        text_parts.append(words[i])
        spaces = base_spaces + (1 if i < extra_spaces else 0)
        text_parts.append(' ' * spaces)

    text_parts.append(words[-1])
    return ''.join(text_parts)


def generate_seal(image_size=(256, 256)):
    background_color = 'white'
    image = Image.new('RGB', image_size, background_color)
    draw = ImageDraw.Draw(image)

    center = (image_size[0] // 2, image_size[1] // 2)
    max_radius = min(center) - 10
    radius = random.randint(max_radius - 35, max_radius - 20)
    circle_width = random.randint(2, 5)

    metadata = {'center': None, 'second_circle': None, 'third_circle': None}
    # 1 круг
    draw.ellipse(
        [
            (center[0] - radius, center[1] - radius),
            (center[0] + radius, center[1] + radius)
        ],
        outline='blue',
        width=circle_width
    )

    # Текст или герб в центре
    if random.choice(['text', 'emblem']) == 'text':
        company_name = '«' + ''.join(random.choices([chr(code) for code in range(ord('А'), ord('Я') + 1)], k=7)) + '»'
        inn_digits = ''.join(random.choices('0123456789', k=10))
        metadata['center'] = {'type': 'text', 'name': company_name, 'inn': inn_digits}
    else:
        metadata['center'] = {'type': 'emblem', 'name': 'russia_emblem'}

    # 2 круг
    outer_radius = radius + 15
    if outer_radius > max_radius:
        outer_radius = max_radius

    draw.ellipse(
        [
            (center[0] - outer_radius, center[1] - outer_radius),
            (center[0] + outer_radius, center[1] + outer_radius)
        ],
        outline='blue',
        width=circle_width
    )

    # 3 круг
    third_radius = outer_radius + 15
    draw.ellipse(
        [
            (center[0] - third_radius, center[1] - third_radius),
            (center[0] + third_radius, center[1] + third_radius)
        ],
        fill='blue'
    )
    draw.ellipse(
        [
            (center[0] - outer_radius, center[1] - outer_radius),
            (center[0] + outer_radius, center[1] + outer_radius)
        ],
        fill='white'
    )

    draw.ellipse(
        [
            (center[0] - radius, center[1] - radius),
            (center[0] + radius, center[1] + radius)
        ],
        outline='blue',
        width=circle_width
    )

    # Текст для 3 круга
    third_text = generate_text_with_even_spaces(55, 35, 50)
    metadata['third_circle'] = third_text
    fonts_dir = os.path.join(os.path.dirname(__file__), '..', 'fonts')
    if not os.path.exists(fonts_dir):
        raise FileNotFoundError(f"Папка со шрифтами не найдена: {fonts_dir}")
    font_files = [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir) if f.endswith(('.ttf', '.otf'))]
    if not font_files:
        raise FileNotFoundError("Не найдено ни одного файла шрифта в папке fonts/")

    font_path = random.choice(font_files)
    font_size = 14
    font = ImageFont.truetype(font_path, font_size)
    draw_text_on_circle(image, center, third_radius - 5, third_text, font, text_color='white', start_angle_deg=270, clockwise=True, flip=True)

    # Текст для 2 круга
    second_text = generate_text_with_even_spaces(45, 30, 40)
    metadata['second_circle'] = second_text
    draw_text_on_circle(image, center, outer_radius - 5, second_text, font, text_color='blue', start_angle_deg=270, clockwise=True, flip=True)

    if random.choice(['text', 'emblem']) == 'text':
        company_name = '«' + ''.join(random.choices([chr(code) for code in range(ord('А'), ord('Я') + 1)], k=7)) + '»'
        font_size = random.randint(20, 25)
        font = ImageFont.truetype(font_path, font_size)
        draw.text(center, company_name, font=font, fill='blue', anchor='mm')

        inn_digits = ''.join(random.choices('0123456789', k=10))
        inn_radius = radius - 10
        inn_font_size = 14
        inn_font = ImageFont.truetype(font_path, inn_font_size)
        draw_text_on_circle(image, center, inn_radius, inn_digits, inn_font, start_angle_deg=90, flip=False)
    else:
        assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
        emblem_path = os.path.join(assets_dir, 'russia_emblem.png')
        if not os.path.exists(emblem_path):
            raise FileNotFoundError(f"Изображение герба не найдено: {emblem_path}")

        emblem = Image.open(emblem_path)
        emblem = remove_white_background(emblem)
        emblem_size = radius * 4
        emblem = emblem.resize((emblem_size, emblem_size), Image.Resampling.LANCZOS)
        emblem_x = center[0] - emblem.width // 2
        emblem_y = center[1] - emblem.height // 2
        image.paste(emblem, (emblem_x, emblem_y), emblem)

    return image,metadata


def draw_text_on_circle(image, center, radius, text, font, start_angle_deg=0, text_color='blue', clockwise=True, flip=False):
    draw = ImageDraw.Draw(image)
    start_angle = math.radians(start_angle_deg)
    circumference = 2 * math.pi * radius
    char_angles = []
    for char in text:
        bbox = font.getbbox(char)
        w = bbox[2] - bbox[0]
        char_angle = (w / circumference) * 2 * math.pi
        char_angles.append(char_angle)

    total_text_angle = sum(char_angles)
    if clockwise:
        current_angle = start_angle - total_text_angle / 2
    else:
        current_angle = start_angle + total_text_angle / 2

    for i, char in enumerate(text):
        char_angle = char_angles[i]
        w = font.getbbox(char)[2] - font.getbbox(char)[0]
        h = font.getbbox(char)[3] - font.getbbox(char)[1]
        if clockwise:
            angle = current_angle + char_angle / 2
        else:
            angle = current_angle - char_angle / 2
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        char_image = Image.new('RGBA', (int(w * 2), int(h * 2)), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_image)
        char_draw.text((w / 2, h / 2), char, font=font, fill=text_color)
        rotation = math.degrees(angle) + 90 if flip else math.degrees(angle) - 90
        rotated_char_image = char_image.rotate(-rotation, resample=Image.BICUBIC, expand=True)
        image.paste(rotated_char_image, (int(x - rotated_char_image.width // 2), int(y - rotated_char_image.height // 2)),
                    rotated_char_image)
        if clockwise:
            current_angle += char_angle
        else:
            current_angle -= char_angle


def remove_white_background(image):
    image = image.convert("RGBA")
    data = np.array(image)
    r, g, b, a = data.T
    white_areas = (r > 240) & (g > 240) & (b > 240)
    data[..., :-1][white_areas.T] = (0, 0, 0)
    data[..., -1][white_areas.T] = 0
    return Image.fromarray(data)


def generate_dataset(num_images, output_dir):
    # Создание папки для изображений
    images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Список для хранения метаданных
    labels = []

    for i in range(1, num_images + 1):
        # Генерация изображения и метаданных
        image, metadata = generate_seal()

        # Имя файла для сохранения изображения
        image_filename = f'seal_{i}.png'
        image_path = os.path.join(images_dir, image_filename)

        # Сохранение изображения
        image.save(image_path)

        # Добавление имени файла к метаданным
        metadata['filename'] = image_filename
        labels.append(metadata)

        print(f'Изображение {i}/{num_images} сгенерировано.')

    # Сохранение метаданных в JSON
    labels_filepath = os.path.join(output_dir, 'labels.json')
    with open(labels_filepath, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

    print(f"Метаданные сохранены в {labels_filepath}")



if __name__ == '__main__':
    num_images = 3
    output_directory = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    generate_dataset(num_images, output_directory)
