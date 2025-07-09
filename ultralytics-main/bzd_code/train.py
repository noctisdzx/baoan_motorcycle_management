from ultralytics import YOLO
import yaml
from pathlib import Path


def update_yaml_path(new_path):
    """更新yaml文件中的path参数"""
    yaml_path = "ultralytics-main/yolo-motorcycle.yaml"
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    data['path'] = str(new_path)
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    
    print(f"✅ 已更新yaml文件中的path为: {new_path}")

def main():
    # 获取用户输入的数据集路径
    default_path = "datasets/aaa"
    user_path = input(f"请输入训练集的绝对路径 (默认: {default_path}): ").strip()
    dataset_path = user_path if user_path else default_path
    
    model_path = "pretrain_model/yolo11n.pt"

    # 更新yaml文件
    update_yaml_path(dataset_path)
    
    # 加载模型并训练
    model = YOLO(Path(input(f"请输入model路径 (默认: {model_path}): ").strip() or model_path))    
    model.train(
        data="ultralytics-main/yolo-motorcycle.yaml",
        epochs=200,
        workers=0,
        imgsz=1280,
        freeze=11,  # 只训练最后的 Detect Head
        lr0=1e-2,
        batch=1
    )


if __name__ == '__main__':
    main()