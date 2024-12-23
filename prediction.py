
import torch
import torch.nn as nn 
import time 
import torchvision
import cv2 as cv
from utils import *
from yoloModel.model import *
from typing import Optional
from torch.backends import mps
from vis import *
class Prediction(DataHandle):

    
    def __init__(self,data:str,model:nn.Module = None,pretrained:Optional[str] = None, names : Optional[dict]=None,vis = True,save=False):

        super().__init__(save,data)

        self.data = data
        self.vis = vis
        if pretrained is not None:
            self.model = load_model_weights(model,pretrained)

            
        else: 
            self.model = model

            self.model = self.model.fuse().eval()
            

        self.device =  "cuda:0" if torch.cuda.is_available() else "cpu"

        
        if names is None:
            self.names ={0: 'person',
                        1: 'bicycle',
                        2: 'car',
                        3: 'motorcycle', 
                        4: 'airplane', 
                        5: 'bus', 
                        6: 'train', 
                        7: 'truck', 
                        8: 'boat', 
                        9: 'traffic light', 
                        10: 'fire hydrant', 
                        11: 'stop sign', 
                        12: 'parking meter', 
                        13: 'bench', 
                        14: 'bird', 
                        15: 'cat', 
                        16: 'dog', 
                        17: 'horse', 
                        18: 'sheep', 
                        19: 'cow', 
                        20: 'elephant', 
                        21: 'bear', 
                        22: 'zebra', 
                        23: 'giraffe', 
                        24: 'backpack', 
                        25: 'umbrella', 
                        26: 'handbag', 
                        27: 'tie', 
                        28: 'suitcase', 
                        29: 'frisbee', 
                        30: 'skis', 
                        31: 'snowboard', 
                        32: 'sports ball', 
                        33: 'kite', 
                        34: 'baseball bat', 
                        35: 'baseball glove', 
                        36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

        else: self.names = names

        self.nc = len(self.names)
    
    @torch.no_grad()
    def predict(self):

        if self.check() != "video":


            start = time.time()

            im = self.process()
            im = im.to(self.device)
            end = time.time()

            print(f"preprocessing took {(end-start)*1000} ms ")

            with torch.no_grad():
                self.model.to(self.device)
                start = time.time()
                preds = self.model(im)
                end = time.time()
            print(f"inference time took {(end-start)*1000} ms ")

               
            start = time.time()
            post = non_max_suppression_ultra(preds,conf_thres=0.2,iou_thres=0.5,max_wh=100,max_nms=30000,nc = self.nc)
            end = time.time()

            print(f"postprocces took {(end-start)*1000} ms")
            bboxes = post[-1][:,:4].detach().cpu().numpy()
            confidances = post[-1][:,4:5].detach().cpu().numpy()
            names = [self.names[int(name.detach().cpu())] for name in post[-1][:,-1]]
            pred_dict = {"image": im, "names" : names, "confidances" : confidances, "bboxes" : bboxes,"labels" :post[-1][:,-1].detach().cpu().numpy()}

            if self.vis:
                self.visualize_preds(pred_dict)


                
            return pred_dict

    def predict_video(self):

        saved_video = []
        
        cap = cv.VideoCapture(self.data)

        w,h,fps = cap.get(cv.CAP_PROP_FRAME_WIDTH),cap.get(cv.CAP_PROP_FRAME_HEIGHT),cap.get(cv.CAP_PROP_FPS)

        while True:

            ret,frame = cap.read()

            if ret:

                frame = self.process(frame)

                with torch.no_grad():

                    preds = self.model(frame)

                post = non_max_suppression_ultra(preds,conf_thres=0.15,iou_thres=0.5,max_wh=100,max_nms=100000)
                bboxes = post[-1][:,:4].detach().cpu().numpy()
                confidances = post[-1][:,4:5].detach().cpu().numpy()
                names = [self.names[int(name.detach().cpu())] for name in post[-1][:,-1]]
                pred_dict = {"image": frame, "names" : names, "confidances" : confidances, "bboxes" : bboxes,"labels" :post[-1][:,-1].detach().cpu().numpy()}
                saved_video.append(self.visualize_preds(pred_dict))

                if cv.waitKey(25) & 0xFF == ord("q"):
                    break
        cap.release()
        cv.destroyAllWindows()

        if self.save:
            four_cc = cv.VideoWriter_fourcc(*'XVID')
            vv = cv.VideoWriter("saved_det.mp4",four_cc,fps,(640,448))
            for im in saved_video:
                vv.write(im)
    
    def __call__(self):

        return self.predict_video() if self.check() == "video" else self.predict()
        

            