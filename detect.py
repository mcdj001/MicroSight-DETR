import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('/hy-tmp/pycharm_project_1/weights/rtdetr-r18.pt') # select your model.pt path
    model.predict(source='dataset/images/vis-val',
                  # conf=0.25,
                  project='runs/detect',
                  name='r18',
                  save=True,
                  # visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )
