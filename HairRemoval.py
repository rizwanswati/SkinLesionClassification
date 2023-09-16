import cv2
import glob
from PIL import Image, ImageFilter
import numpy as np


def remove_hair(ddi_black_hair, ddi_clean_output):
    for filePath in glob.glob(ddi_black_hair):
        src = cv2.imread(filePath)
        filename = filePath.partition("\\")[2]

        # Convert the original image to grayscale
        grayScaleImage = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        # don't need to write output of every step but for proofreading lets do it anyway.
        cv2.imwrite(ddi_clean_output + "DDI_grayscale/" + filename, grayScaleImage,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # Kernel for the morphological filtering
        kernel = cv2.getStructuringElement(1, (17, 17))

        # # Perform the blackHat filtering on the grayscale image to find the
        # # hair countours
        blackhatImage = cv2.morphologyEx(grayScaleImage, cv2.MORPH_BLACKHAT, kernel)
        cv2.imwrite(ddi_clean_output + "DDI_blackhat/" + filename, blackhatImage,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # intensify the hair countours in preparation for the inpainting
        # algorithm
        ret, thresh2Image = cv2.threshold(blackhatImage, 10, 255, cv2.THRESH_BINARY)
        cv2.imwrite(ddi_clean_output + "DDI_threshold/" + filename, thresh2Image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # # inpaint the original image depending on the mask
        inpaintedImage = cv2.inpaint(src, thresh2Image, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(ddi_clean_output + "DDI_inpainted/" + filename, inpaintedImage,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])


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


def main():
    ddi_black_hair = "D:/SkinLesionClassification/DDI_hair/*.png"
    # ddi_hair = "D:/SkinLesionClassification/DDI_hair/"
    ddi_clean_output = "D:/SkinLesionClassification/DDI_Clean_Output/"
    ddi_blue_marks = "D:/SkinLesionClassification/DDI_Clean_Output/DDI_inpainted/*.png"
    ddi_blue = "D:/SkinLesionClassification/DDI_Clean_Output/DDI_inpainted/"

    remove_hair(ddi_black_hair, ddi_clean_output)
    remove_blue_marks(ddi_blue_marks, ddi_blue, ddi_clean_output)


if __name__ == '__main__':
    main()
