from ultralytics import YOLO
import torch
import multiprocessing
def main():
    # 加载模型并将其移动到 CUDA 设备上
    model = YOLO("yolov8n.yaml")
    model.load(r"C:\\Users\\86198\\Desktop\\base\\yolov8n.pt")
    # 获取所有可用的GPU设备
    device_ids = list(range(torch.cuda.device_count()))
    if len(device_ids) > 1:
        # 使用DataParallel来利用多GPU
        model.model = torch.nn.DataParallel(model.model, device_ids=device_ids)
        model.to('cuda')
    else:
        # 如果只有一个GPU或没有GPU，使用单个GPU或CPU
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
    # 开始训练，并指定使用 CUDA 设备
    model.train(data=r'C:\Users\86198\Desktop\base\dataset.yaml', epochs=100, imgsz=320, device='cpu')
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
