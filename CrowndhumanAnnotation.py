import os

def parse_annotations(annotation_file):
    annotations = {}
    with open(annotation_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            img_id = parts[0]
            tag = parts[1]
            fbox_index = parts.index('fbox:')
            fbox = parts[fbox_index + 1:fbox_index + 5]
            hbox_index = parts.index('hbox:')
            hbox = parts[hbox_index + 1:hbox_index + 5]
            vbox_index = parts.index('vbox:')
            vbox = parts[vbox_index + 1:vbox_index + 5]
            if img_id not in annotations:
                annotations[img_id] = []
            annotations[img_id].append({
                'tag': tag,
                'fbox': fbox,
                'hbox': hbox,
                'vbox': vbox
            })
    return annotations

def save_annotation_txt(annotation, img_name, output_dir):
    img_id = img_name.rsplit('.', 1)[0]
    output_file = os.path.join(output_dir, f"{img_id}.txt")
    with open(output_file, 'w') as file:
        for box in annotation:
            tag = box['tag']
            fbox = ' '.join(box['fbox'])
            hbox = ' '.join(box['hbox'])
            vbox = ' '.join(box['vbox'])
            file.write(f"{tag} fbox: {fbox} hbox: {hbox} vbox: {vbox}\n")

def process_images(image_dir, annotations, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all image files in the directory
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_id = img_file.rsplit('.', 1)[0]
            if img_id in annotations:
                save_annotation_txt(annotations[img_id], img_file, output_dir)
            else:
                print(f"No annotation found for image: {img_file}")

# 读取 `annotations.txt` 文件
annotation_file_path = r'F:\CrowdHuman\data\annotation_train.txt'
annotations = parse_annotations(annotation_file_path)

# 处理图像并生成对应的标注文件
image_directory = r'F:\CrowdHuman\data\Images'
output_directory = r'F:\CrowdHuman\data\Annotations'
process_images(image_directory, annotations, output_directory)

print("Processing complete.")
