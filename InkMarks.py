import cv2
import glob
from PIL import Image, ImageFilter
import numpy as np


def remove_blue_marks(ddi_blue_marks, ddi_blue, ddi_clean_output):
    for filePath in glob.glob(ddi_blue_marks):
        src = cv2.imread(filePath)
        filename = filePath.partition("\\")[2]

        # remove blue ink marks

        blueImage = cv2.imread(ddi_blue + "/" + filename)
        hsv = cv2.cvtColor(blueImage, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([150, 255, 255])

        kernel = np.ones((3, 3), np.uint8)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_blue2 = cv2.dilate(mask_blue, kernel, iterations=5)
        dst = cv2.inpaint(blueImage, mask_blue2, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(ddi_clean_output + "DDI_blue_marks_removed/" + filename, dst,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # remove purple ink marks

        purple = cv2.imread(ddi_clean_output + "DDI_blue_marks_removed/" + filename)

        hsv = cv2.cvtColor(purple, cv2.COLOR_BGR2HSV)
        lower_purple = np.array([129, 50, 70])
        upper_purple = np.array([158, 255, 255])
        kernel = np.ones((5, 5), np.uint8)
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
        mask_purple2 = cv2.dilate(mask_purple, kernel, iterations=6)
        dst = cv2.inpaint(purple, mask_purple2, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(ddi_clean_output + "DDI_purple_marks_removed/" + filename, dst,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # remove black ink marks

        # black color removal
        black = cv2.imread(ddi_clean_output + "DDI_purple_marks_removed/" + filename)

        hsv = cv2.cvtColor(black, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])
        kernel = np.ones((5, 5), np.uint8)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask_black2 = cv2.dilate(mask_black, kernel, iterations=6)
        dst = cv2.inpaint(black, mask_black2, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(ddi_clean_output + "DDI_black_marks_removed/" + filename, dst,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def remove_black_marks(ddi_black_marks, ddi_black, ddi_clean_output):
    for filePath in glob.glob(ddi_black_marks):
        src = cv2.imread(filePath)
        filename = filePath.partition("\\")[2]

        black = cv2.imread(ddi_black + "/" + filename)

        hsv = cv2.cvtColor(black, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 40])
        kernel = np.ones((5, 5), np.uint8)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask_black2 = cv2.dilate(mask_black, kernel, iterations=6)
        dst = cv2.inpaint(black, mask_black2, 10, cv2.INPAINT_TELEA)
        cv2.imwrite(ddi_clean_output + "DDI_black_marks_removed/" + filename, dst,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def main():
    ddi_blue_marks = "D:/SkinLesionClassification/DDI_blue_marks/*.png"
    ddi_blue = "D:/SkinLesionClassification/DDI_blue_marks/"
    ddi_black_marks = "D:/SkinLesionClassification/DDI_black_marks/*.png"
    ddi_black = "D:/SkinLesionClassification/DDI_black_marks/"
    ddi_clean_output = "D:/SkinLesionClassification/DDI_Clean_Output/"
    # remove_blue_marks(ddi_blue_marks, ddi_blue, ddi_clean_output)
    remove_black_marks(ddi_black_marks, ddi_black, ddi_clean_output)


if __name__ == '__main__':
    main()
