import cv2 as cv
import PIL.Image as Image
import numpy as np
import torch
from utils import preprocess,pre_transform
import matplotlib.pyplot as plt
class DataHandle():

    def __init__(self,save:bool=False,data:str = "",put_def = True):

        self.save = save 

        self.data = data  
        self.pd = put_def

        self.color_map = self.generate_colormap()


    def generate_colormap(self,num_classes=80):
        """
        Generate a colormap for the given number of classes.
        
        Args:
            num_classes (int): Number of classes to generate colors for.
        
        Returns:
            List[tuple]: A list of RGB color tuples.
        """
        colormap = plt.get_cmap('tab20', num_classes)  # 'tab20' is a good colormap with distinct colors
        colors = [colormap(i)[:3] for i in range(num_classes)]  # Get RGB colors

        colors[0] = (0,255,0)
        # Convert colors from [0, 1] to [0, 255] range
        colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]
    
        return colors 
    def plot_one_box(self,xyxy,image,color,defination):
        c1,c2 = ((int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])))
        colors = (255,0,0) if len(color) == 0 else color

        

        cv.rectangle(image,c1,c2,color=colors,thickness=2)

        if self.pd:

            cv.putText(image,defination,(c1[0],c1[-1]-3),cv.FONT_HERSHEY_SIMPLEX,1.,color,2)

    def visualize_preds(self,preds:dict):
        """
        To visualize the prediction for the single image

        Args:
            preds (dict): Model prediction dictionary.
        """

        if len(preds) != 1:
            if isinstance(preds["image"],torch.Tensor):
                image = (preds["image"].detach().squeeze(0).permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()
            else: image = preds["image"]
            print(preds["names"])
            for i,det in enumerate(preds["bboxes"]):

                cls = preds["names"][i]
                conf = float(preds["confidances"][i])

                definition = f"{cls}: {conf:.2f}"
                self.plot_one_box(det, image, color = self.color_map[int(preds["labels"][i])],defination = definition)

        else: image = self.read_img()

        if self.save:
            cv.imwrite("result.png",image)

        if self.check() == "video":
            cv.imshow("Detections",image)
            return image

        else:
        
            cv.imshow('Detections', image)
            cv.waitKey(0)
            cv.destroyAllWindows()

                   
    def check(self):
        """
        To check given data is video or image. It also dynamic if given data is regular image it return directly.

        Returns:
            _type_: _description_
        """
        
        if isinstance(self.data,str):

            check = self.data.endswith(".mp4") or self.data.endswith(".avi")
            return "video" if check else "image"
        
        else: return self.data
        

    def process(self,img = None):
        """
        Preprocessing for the given image to make prediction

        Args:
            img (Path or np.ndarray): To handle path and regular image

        Returns:
            torch.Tensor: The method return tensor to make prediction.
        """
        if img is  None:
            img = self.read_img()
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        img = img.reshape((1,*img.shape))
        img = preprocess(img)
        return img

    
    def read_img(self):
        """
        This method read the path of image if you give data astype np.ndarray it return data directly.

        Returns:
            np.ndarray : _description_
        """

        if  (self.check() == "image") or not (isinstance(self.data,np.ndarray)):
            return cv.imread(self.data)
        if isinstance(self.data,np.ndarray):
            return self.data

    def to_numpy(self,data:torch.Tensor):
        """
        This method is used for turn the tensor type image to numpy type image for visualizing   

        Args:
            data (torch.Tensor): Tensor type Output

        Returns:
            np.ndarray : It returns numpy ndarray
        """
    
        data = (data.detach().squeeze(0).permute(1,2,0).contiguous()*255).to(torch.uint8).cpu().numpy()
        return data



    

    
    
    
        

            





    


                    
                    

                    











