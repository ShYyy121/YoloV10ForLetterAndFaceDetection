import os
import cv2

def convert_to_yolo_format(annotation_file, output_file, img_width, img_height):
    with open(annotation_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            tag = parts[0]
            if tag != 'person':
                continue

            try:
                hbox_index = parts.index('hbox:')
                hbox = list(map(int, parts[hbox_index + 1:hbox_index + 5]))
            except (ValueError, IndexError):
                print(f"Skipping line due to format error: {line.strip()}")
                continue

            x1, y1, w, h = hbox
            if w <= 0 or h <= 0:
                continue

            # 计算YOLO格式的归一化中心坐标和宽高
            x_center = (x1 + w / 2) / img_width
            y_center = (y1 + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            # 类别标签统一为 7
            label = 7
            yolo_line = f"{label} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
            outfile.write(yolo_line)

def process_annotations(image_dir, annotation_dir, output_dir):
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

            # 读取图像的宽度和高度
            img = cv2.imread(image_file_path)
            if img is None:
                print(f"Failed to load image: {image_file_path}")
                continue
            img_height, img_width, _ = img.shape

            # 设置输出文件路径
            output_file_path = os.path.join(output_dir, annotation_file)

            # 转换为YOLO格式
            convert_to_yolo_format(annotation_file_path, output_file_path, img_width, img_height)

# 设置输入和输出目录
image_directory = r'F:\CrowdHuman\data\Images'
annotation_directory = r'F:\CrowdHuman\data\Annotations'
output_directory = r'F:\CrowdHuman\data\Annotations_YOLO'

process_annotations(image_directory, annotation_directory, output_directory)

print("Conversion complete.")
