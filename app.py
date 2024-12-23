from utils import *
from yoloModel.model import *

from prediction import *

from to_oonx import *

import torch

if __name__ == "__main__":

    model = DetectionModel("cfg/yolov9c.yaml",nc = 1)

    # count_parameters(model)

    # model = torch.load("/Users/okanegemen/Desktop/YoloModelImplementation/pretrained_model/yolov9c_coppied.pt")["model"]

    # m = torch.jit.script(model)

    # torch.jit.save(m,"saved_jit_mode.pt")

    inference = Prediction("/Users/okanegemen/Desktop/YoloModelImplementation/100_jpg.rf.de01d21ffbe941ccbe6da7e4d96c73d8.jpg",
                            pretrained = "/Users/okanegemen/Desktop/YoloModelImplementation/pretrained_model/best-2.pt",
                            model = model,
                            save=False,names= {0:"follicle"})

    inference.pd = True

    inference()

    # deploy = Deploy(dynamic=True)


    # deploy()

    
   
