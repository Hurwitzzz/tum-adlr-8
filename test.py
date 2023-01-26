import numpy as np
from PIL import Image

img_path = "1a9b552befd6306cc8f2d5fe7449af61-6.png"

img = np.array(Image.open(img_path))

print(img.shape)


def _ray_cast(x_dir, y_dir, x_pos, y_pos):
    """Using given starting position on the frame, and also the vector direction given
    casts the ray and finds the first intersecting position within the expected image

    Args:
        x_dir (int): x start
        y_dir (int): y start
        x_pos (float): x vector dir
        y_pos (float): y vector dir

    Returns:
        (np.ndarray): Position of the first intersection point
    """
    # starting point should be on the frame
    assert 0 in [x_pos, y_pos] or 99 in [x_pos, y_pos]

    norm = np.linalg.norm(np.array([x_dir, y_dir]), ord=2)
    x_dir /= norm
    y_dir /= norm

    while (x_pos <= 99 and y_pos <= 99) and (x_pos >= 0 and y_pos >= 0):
        x_pos += x_dir
        y_pos += y_dir

        x, y = np.clip(int(x_pos), a_min=0, a_max=99, dtype=int), np.clip(int(y_pos), a_min=0, a_max=99, dtype=int)
        if img[x, y]:
            return x, y

    return -1, -1


new_img = np.zeros_like(img)
for i in range(0, 1000):

    radian = np.random.uniform(-1, 1) * np.pi
    x_dir = np.cos(radian)
    y_dir = np.sin(radian)

    pos = np.random.uniform(-1, 1)

    pos = int((pos + 1) * 199.5)

    if pos < 100:
        x_pos = pos
        y_pos = 0
    elif 100 <= pos < 200:
        x_pos = 99
        y_pos = pos - 100
    elif 200 <= pos < 300:
        x_pos = pos - 200
        y_pos = 99
    elif 300 <= pos < 400:
        x_pos = 0
        y_pos = pos - 300

    x, y = _ray_cast(x_dir, y_dir, x_pos, y_pos)
    if x != -1 and y != -1:
        new_img[x, y] = 1

Image.fromarray(new_img).save("test_raycast.png")
