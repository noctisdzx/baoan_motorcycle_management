from ultralytics import YOLO

model = YOLO('pretrain_model\yolo11n.pt')
#冻结部分层数的权重，冻结最后几个head层
#加图片，指定监控视角
#先过拟合

#通过对未能识别的情况，在冻结基础模型下，微调后几层
#model.train(data='yolo-motorcycle.yaml',workers=0,epochs=20,batch=1)
model.train(
    data="yolo-motorcycle.yaml",
    epochs=200,
    workers = 0,
    imgsz=1280,
    freeze=11,  # 只训练最后的 Detect Head
    lr0=1e-2,
    batch=1
)