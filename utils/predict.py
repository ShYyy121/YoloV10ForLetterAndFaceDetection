import os
import glob
import time
import numpy as np
import cv2
from ultralytics import YOLOv10

classes = {
    0: 'letters', 1: 'digits', 2: 'chinese_characters', 3: 'letters_digits', 4: 'letters_chinese_characters',
    5: 'digits_chinese_characters', 6: 'mixed_content',7:'face'
}

class Colors:
    """Ultralytics color palette https://ultralytics.com/."""

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to rgb values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

imgpath = r'F:\东北大学张伟老师-敏感信息检测加密\model\EasyOCR-master\totaltext\Images\Test'
modelpath = r'F:\东北大学张伟老师-敏感信息检测加密\yolov10-main\runs\detect\train9\weights\best.pt'
save_dir = imgpath + '_Rst'
os.makedirs(save_dir, exist_ok=True)
model = YOLOv10(modelpath)

imgs = glob.glob(os.path.join(imgpath, '*.jpg'))
for img in imgs:
    imgname = os.path.basename(img)
    frame = cv2.imread(img)

    # 记录预测开始时间
    start_time = time.time()

    results = model.predict(img)[0]

    # 记录预测结束时间
    end_time = time.time()

    # 计算预测消耗的时间
    prediction_time = end_time - start_time

    for box in results.boxes:
        xyxy = box.xyxy.squeeze().tolist()
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        c, conf = int(box.cls), float(box.conf)
        name = classes[c]
        color = colors(c, True)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"{name}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 保存带有标注的图像
    cv2.imwrite(os.path.join(save_dir, imgname), frame)

    print(f"Processed {imgname} in {prediction_time:.4f} seconds")
