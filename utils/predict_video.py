import cv2
import time
from ultralytics import YOLOv10

# 定义类别标签
classes = {
    0: 'letters', 1: 'digits', 2: 'chinese_characters', 3: 'letters_digits', 4: 'letters_chinese_characters',
    5: 'digits_chinese_characters', 6: 'mixed_content',7:'face'
}

# 定义颜色类
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
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

# 创建颜色实例
colors = Colors()

# 设置模型路径
modelpath = r'F:\东北大学张伟老师-敏感信息检测加密\yolov10-main\runs\detect\train9\weights\best.pt'

# 加载模型
model = YOLOv10(modelpath)

# 打开摄像头
cap = cv2.VideoCapture(1)  # 0 表示默认摄像头

if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# 获取视频帧的宽度和高度
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 定义视频编码器和输出文件
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('../output.avi', fourcc, 20.0, (frame_width, frame_height))

while True:
    # 读取帧
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # 记录预测开始时间
    start_time = time.time()

    # 使用模型进行推理
    results = model.predict(frame)[0]

    # 记录预测结束时间
    end_time = time.time()

    # 计算预测消耗的时间
    prediction_time = end_time - start_time

    # 处理预测结果
    for box in results.boxes:
        xyxy = box.xyxy.squeeze().tolist()
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        c, conf = int(box.cls), float(box.conf)
        name = classes[c]
        color = colors(c, True)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"{name}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 将帧写入视频文件
    out.write(frame)

    # 显示带有标注的帧
    cv2.imshow('YOLOv10 Detection', frame)
    print(f"Processed frame in {prediction_time:.4f} seconds")

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
