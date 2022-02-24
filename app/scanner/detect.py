import json
from io import BytesIO
import argparse
import sys
import time
from pathlib import Path
import base64
import cv2
import numpy as np
from numpy.core.fromnumeric import var
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
from utils.augmentations import letterbox
from utils.locate_asset import locate_asset, ocrScan
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

from PIL import Image
@torch.no_grad()
def run(
        img= np.ndarray,
        im0s = np.ndarray,
        ):
    weights='labelsmodel/best.pt'  # model.pt path(s)
    source='data/images'  # file/dir/URL/glob, 0 for webcam
    imgsz=[640, 640]  # inference size (pixels)
    conf_thres=0.4  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False  # show results
    save_txt=False  # save results to *.txt
    save_conf=False  # save confidences in --save-txt labels
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    update=False  # update all models
    project='runs'  # save results to project/name
    name='exp'  # save results to project/name
    exist_ok=False  # existing project/name ok, do not increment
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False

    dataset = [[img, im0s]]
    bs = 1

    save_img = not nosave and not source.endswith('.txt')  # save inference images

    

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix = False, Path(w).suffix.lower()
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for img, im0s in dataset:
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:
            visualize = False
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()


        # Process predictions
        for i, det in enumerate(pred):  # detections per image
           
            s, im0, frame = '', im0s.copy(), getattr(dataset, 'frame', 0)
            
            
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0 # for save_crop
            imageArr = [] 
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img or save_crop or view_img:  # Add bbox to image
                        #print(f'RESULTS. ({xyxy[1]}s)')
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}')
                        #im0 = plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_width=line_thickness)
                        numFix = (float(xyxy[0].data.cpu().numpy()), float(xyxy[1].data.cpu().numpy()), float(xyxy[2].data.cpu().numpy()), float(xyxy[3].data.cpu().numpy()))
                        
                        imageArr.append((numFix , plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_width=line_thickness), label))
                        

            # Stream results
            if view_img:
                # cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            data=[]
            if save_img:
                data.append(ocrScan(imageArr))            

    
    return data



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='labelsmodel/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def get_np_image(image_bytes):
    image = Image.open(BytesIO(base64.b64decode(image_bytes))).convert(mode='L')
    dataNp = np.asarray(image)  
    img0 = cv2.cvtColor(dataNp, cv2.COLOR_BGR2RGB)

    return img0

def get_fixed_image(image_bytes):
    image = Image.open(BytesIO(base64.b64decode(image_bytes))).convert(mode='L')
    
    data = np.array(image)  
    img0 = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    # Padded resize
    img = letterbox(img0, 640, stride=32, auto=True)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    return img

def lambda_handler(event, context):
    image_bytes = event['body'].encode('utf-8')
    imgArg1 = get_fixed_image(image_bytes)
    imgArg2 = get_np_image(image_bytes)
    dataSet = run(imgArg1, imgArg2)

    return {
        'statusCode': 200,
        'body': json.dumps(dataSet)
    }


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    imgSize = [640]
    imgSize *= 2
    run()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
