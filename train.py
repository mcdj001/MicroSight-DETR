import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')
    model.train(data='/hy-tmp/pycharm_project_1/datasets/data.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=8,
                workers=4,
                project='runs/train',
                name='baseline',
                verbose=True,
                )
