import cv2
from utils import img_util
import pytesseract


def extract_text(src, scale=2, iterations = 1, config = r'--oem 3 --psm 6', blur=True):
    (x1, y1, w1, h1) = img_util.dilate_inner_get_contour(src,iterations=iterations)
    textarea = img_util.sharpen(src[y1:y1+h1, x1:x1+w1])
    enlarge = cv2.resize(textarea,(textarea.shape[1]*scale,textarea.shape[0]*scale))
    if blur:
        enlarge = cv2.bilateralFilter(enlarge, 9, 60, 60)
    return pytesseract.image_to_string(enlarge, config = config)

def denoise(src, base_kernel=23, scale=1, base_sigma=8):
    kernel_size = (base_kernel * scale) + (1 if scale % 2 == 0 else 0)
    sigma = base_sigma * scale
    thresh = cv2.adaptiveThreshold(src,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,kernel_size, sigma)
    return cv2.fastNlMeansDenoising(thresh, None, 51, 7, 31)