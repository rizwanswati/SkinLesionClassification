import glob
import shutil
import csv


def save_original_images_to_folder(image_name, csv_file, output_benign, output_malignant, folder):
    csv_file = csv.reader(open(csv_file, "r"), delimiter=",")
    for row in csv_file:
        if row[2] == image_name and row[4] == 'TRUE':
            source_file_path = folder + "/" + image_name
            shutil.move(source_file_path, output_malignant)
        if row[2] == image_name and row[4] == 'FALSE':
            source_file_path = folder + "/" + image_name
            shutil.move(source_file_path, output_benign)


def split_original_images(input_image_folder, output_benign, output_malignant, class_file_path, output_folder):
    for filePath in glob.glob(input_image_folder):
        filename = filePath.partition("\\")[2]
        save_original_images_to_folder(filename, class_file_path, output_benign, output_malignant, output_folder)


def main():
    original_folder_path = "D:/SkinLesionClassification/DDI_original/*.png"
    original_folder = "D:/SkinLesionClassification/DDI_original/"
    original_benign = "D:/SkinLesionClassification/DDI_original/benign"
    original_malignant = "D:/SkinLesionClassification/DDI_original/malignant"
    class_file_path = "D:/SkinLesionClassification/ddi_metadata.csv"

    split_original_images(original_folder_path, original_benign, original_malignant, class_file_path, original_folder)


if __name__ == "__main__":
    main()
