import numpy as np
import cv2

def _rgb_to_color_name(rgb):
    """Map an RGB triplet to a coarse color name."""
    rgb_u8 = np.uint8([[rgb]])
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    print(h, s, v)

    # Low value first – too dark to distinguish hue.
    if v < 45:
        return "black"

    # Low saturation – achromatic.
    if s < 30:
        if v > 210:
            return "white"
        return "gray"

    # Brown is dark orange/yellow.
    if 8 <= h <= 25 and v < 160:
        return "brown"
    if h < 10 or h >= 170:
        return "red"
    if h < 20:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 105:
        return "cyan"
    if h < 135:
        return "blue"
    if h < 160:
        return "purple"
    return "pink"

rgb = [2, 2, 3]
print(_rgb_to_color_name(rgb))