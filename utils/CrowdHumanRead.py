import json

def load_odgt(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_as_txt(data, output_file):
    with open(output_file, 'w') as f:
        for annotation in data:
            img_id = annotation['ID']
            for box in annotation['gtboxes']:
                tag = box['tag']
                fbox = ' '.join(map(str, box['fbox']))  # Full body bounding box
                hbox = ' '.join(map(str, box['hbox']))  # Head bounding box
                vbox = ' '.join(map(str, box['vbox']))  # Visible body bounding box
                line = f"{img_id} {tag} fbox: {fbox} hbox: {hbox} vbox: {vbox}\n"
                f.write(line)

# 读取 `annotations.odgt` 文件
file_path = r'D:\迅雷下载\CrowdHuman\data\annotation_train.odgt'
annotations = load_odgt(file_path)

# 保存为 `annotations.txt` 文件
output_file = r'D:\迅雷下载\CrowdHuman\data\annotation_train.txt'
save_as_txt(annotations, output_file)

print("Conversion complete.")
