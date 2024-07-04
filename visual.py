import os
import cv2
import matplotlib.pyplot as plt

def load_yolo_annotations(annotation_file):
    annotations = []
    with open(annotation_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            label = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            annotations.append((label, x_center, y_center, width, height))
    return annotations

def draw_annotations(image_path, annotations, class_names):
    img = cv2.imread(image_path)
    img_height, img_width, _ = img.shape

    for annotation in annotations:
        label, x_center, y_center, width, height = annotation
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, class_names[label], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img

def visualize_annotations(image_dir, annotation_dir, output_dir, class_names):
    os.makedirs(output_dir, exist_ok=True)

    for annotation_file in os.listdir(annotation_dir):
        if annotation_file.endswith('.txt'):
            annotation_file_path = os.path.join(annotation_dir, annotation_file)

            # 根据图像文件名构建图像路径
            img_id = annotation_file.rsplit('.', 1)[0].replace("_", ",")
            image_file_path = os.path.join(image_dir, f"{img_id}.jpg")

            # 确保图像文件存在
            if not os.path.exists(image_file_path):
                print(f"Image not found: {image_file_path}")
                continue

            annotations = load_yolo_annotations(annotation_file_path)
            annotated_img = draw_annotations(image_file_path, annotations, class_names)

            output_file_path = os.path.join(output_dir, f"{img_id}.jpg")
            cv2.imwrite(output_file_path, annotated_img)
            print(f"Annotated image saved to: {output_file_path}")

# 设置输入和输出目录
image_directory = r'F:\CrowdHuman\data\Images'
annotation_directory = r'F:\CrowdHuman\data\Annotations_YOLO'
output_directory = r'F:\CrowdHuman\data\Annotated_Images'

# 定义类名
class_names = {
    0: 'letters', 1: 'digits', 2: 'chinese_characters', 3: 'letters_digits', 4: 'letters_chinese_characters',
    5: 'digits_chinese_characters', 6: 'mixed_content', 7: 'face'
}

visualize_annotations(image_directory, annotation_directory, output_directory, class_names)

print("Visualization complete.")
