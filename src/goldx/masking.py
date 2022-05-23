import random

from PIL import Image, ImageDraw


def generate_mask(width, height, min_area=0.1, max_area=0.3):
    image_area = width * height
    min_size = int(image_area * min_area)
    max_size = int(image_area * max_area)

    pi = 3.14159265358979323846

    min_radius = (min_size / pi) ** 0.5
    max_radius = (max_size / pi) ** 0.5

    radius = random.randint(int(min_radius), int(max_radius))

    center_x = random.randint(radius, width - radius)
    center_y = random.randint(radius, height - radius)

    mask = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill=1,
    )

    return mask
