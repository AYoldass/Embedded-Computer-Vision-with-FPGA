

import onnx
import onnxruntime

import torch
import os
from yoloModel.model_blocks import Detect,C2f

from utils import *

import warnings

warnings.filterwarnings("ignore",category = torch.jit.TracerWarning)
class Deploy():

    def __init__(self, pt_file="./pretrained_model/yolov9c_coppied.pt",dynamic = False):

        self.model = torch.load(pt_file)["model"]

        
        for par in self.model.parameters():
            par.requires_grad = False
        self.model.float()
        self.model.fuse()
        self.model.eval()

       

        self.im = torch.zeros((1,3,640,640))

        for _ in range(2):

            y = self.model(self.im)
        

        for m in self.model.modules():
            
            if isinstance(m,Detect):

                m.format = "onnx"

                m.export = True
                m.dynamic = dynamic
                m.max_det = 300

            elif isinstance(m,C2f):

                m.forward = m.forward_split

       

        self.dynamic = dynamic

    

            
    def export_to_onnx(self):

        dynamic = self.dynamic

        if dynamic:

            dynamic = {"images" : {0 : "batch"  , 2: "height", 3: "width"}}
            dynamic["output"] = {0 : "batch" , 2 : "anchors"}

        opset = max(int(k[14:]) for k in vars(torch.onnx) if "symbolic_opset" in k) - 1



        if os.path.exists("./ONNX_MODEL") is False:

            os.mkdir("./ONNX_MODEL")


        torch.onnx.export(
            self.model.cpu(), 
            self.im.cpu(), 
            os.path.join("./ONNX_MODEL","yolov9.onnx"), 
            opset_version=opset,
            export_params= True,
            do_constant_folding=True,
            input_names=["images"], 
            output_names=["output"],
            dynamic_axes= dynamic if self.dynamic else None
            )

    def __call__(self):
        return self.export_to_onnx()



    

        











        
       


        
