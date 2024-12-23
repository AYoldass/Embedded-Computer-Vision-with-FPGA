
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class ConvBlock(nn.Module):
    def __init__(self, in_channel,
                        out_channel,
                        kernel_size=1,
                        stride=1,
                        padding=None,
                        group = 1,
                        dilation=1,
                        act = True
                        ):
        super().__init__()

        self.default_act = nn.SiLU(inplace=True) if act==True else act if isinstance(act,nn.Module) else nn.Identity()

        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size,stride = stride,padding=autopad(kernel_size,padding,dilation),groups=group,dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channel)

        

    def forward(self,x):

        return self.default_act(self.bn(self.conv(x)))

    

class DFL(nn.Module):
    

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class RepNConv(nn.Module):

    def __init__(self,
                in_channels,
                out_channels,
                kernel_size = 3,
                stride=1,
                padding = 1,
                dilation=1,
                groups = 1, 
                act = True, 
                bn = False, 
                deploy = False):
        super().__init__()

        assert kernel_size == 3 and padding == 1 , "In the block defination kernel_size should be 3 and padding should be 1"

        self.in_ch = in_channels
        self.out_ch = out_channels
        self.g = groups

      
        default_act = nn.SiLU(inplace = True) 

        self.act = default_act if act is True else  act if isinstance(act,nn.Module) else nn.Identity()
      

        self.bn = None

        self.cv1  = ConvBlock(in_channels,out_channels,kernel_size,stride,padding,groups,dilation,False)
        self.cv2 = ConvBlock(in_channels,out_channels,1,1,padding=(padding-kernel_size//2),group=groups,dilation=dilation)
        
    def forward(self,x):

        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.cv1(x) + self.cv2(x) + id_out)
    
    def forward_fuse(self,x):

        return self.act(self.conv(x))

    
    def pad_1x1_to_3x3_tensor(self,kernel1x1):

        return 0 if kernel1x1 is None else F.pad(kernel1x1,[1,1,1,1]) 

    
    def fuse_bn_tensor(self,branch):

        if branch is None:

            return 0

        if isinstance(branch,ConvBlock):

            kernel = branch.conv.weight
            running_var = branch.bn.running_var
            running_mean = branch.bn.running_mean
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps

        elif isinstance(branch,nn.BatchNorm2d):
            if not hasattr(self,"id_tensor"):

                input_dim = self.in_ch//self.g
                kernel_values = torch.zeros((self.in_ch,input_dim,3,3),dtype=torch.float32)

                for i in range(self.in_ch):

                    kernel_values[i, i % input_dim , 1 , 1]= 1

                self.id_tensor = kernel_values.to(branch.weight.device)

            kernel = self.id_tensor

            running_mean = branch.running_mean 
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

            std = torch.sqrt(running_var +eps)
            t = (gamma/std).reshape([-1,1,1,1])

            return kernel * t, beta - running_mean*gamma /std

    def get_equivalent_kernel_bias(self):

        kernel3x3,bias3x3 = self.fuse_bn_tensor(self.cv1)
        kernel1x1,bias1x1 = self.fuse_bn_tensor(self.cv2)

        kernel_id,bias_id = self.fuse_bn_tensor(self.bn)

        return kernel3x3 + self.pad_1x1_to_3x3_tensor(kernel1x1) + kernel_id, bias3x3 + bias1x1 + bias_id

    def fuse_convs(self):
        
        if hasattr(self,"conv"):
            return
        
        kernel,bias = self.get_equivalent_kernel_bias()

        self.conv = nn.Conv2d(

            in_channels = self.cv1.conv.in_channels,
            out_channels= self.cv1.conv.out_channels,
            kernel_size = self.cv1.conv.kernel_size,
            stride = self.cv1.conv.stride,
            padding= self.cv1.conv.padding,
            groups = self.cv1.conv.groups,
            bias = self.cv1.conv.bias,
            dilation= self.cv1.conv.dilation
        ).requires_grad_(False)

        self.conv.weight.data = kernel
        self.conv.bias.data = bias

        for p in self.parameters():

            p.detach()
        
        self.__delattr__("cv1")

        self.__delattr__("cv2")

        if hasattr(self,"nm"):
        
            self.__delattr__("nm")

        if hasattr(self,"bn"):

            self.__delattr__("bn")

        if hasattr(self,"id_tensor"):

            self.__delattr__("id_tensor")



class RepNBottleNeck(nn.Module):

    def __init__(self,in_channel,out_channel,kernel_size = (3,3), groups = 1, expand = 0.5 , shortcut = True):
        super().__init__()

        hidden = int(out_channel * expand)

        self.cv1 = RepNConv(in_channel,hidden,kernel_size = kernel_size[0])
        self.cv2 = ConvBlock(hidden,out_channel,int(kernel_size[1]),padding = 1,group=groups)

        self.add = shortcut and in_channel == out_channel

    def forward(self,x):

        out = self.cv1(x)
        out = self.cv2(out)

        return x + out if self.add is True else out

class RepNCSP(nn.Module):

    def __init__(self,in_ch,out_ch,n = 1,shortcut = True, group = 1 , expand = 0.5 ) -> None:
        super().__init__()

        hidden = int(out_ch*expand)

        self.cv1 = ConvBlock(in_ch,hidden,kernel_size=1,stride =1)
        self.cv2 = ConvBlock(in_ch, hidden,kernel_size=1,stride = 1)

        self.conv3 = ConvBlock(2*hidden,out_ch,kernel_size=1,stride=1)

        self.nBootleNeck = nn.Sequential(*(RepNBottleNeck(hidden,hidden,shortcut=shortcut,groups=group,expand = 1.0) for _ in range(n)))

    def forward(self,x):
        
        out1 = self.cv1(x)
        out2 = self.cv2(x)

        outNeck = self.nBootleNeck(out1)

        out = torch.cat([outNeck,out2],dim=1)

        return self.conv3(out) 



class RepNCSPELAN4(nn.Module):

    def __init__(self, c1,c2,c3,c4,number = 1) -> None:
        super().__init__()

        
        self.cv1 = ConvBlock(c1,c3,1,1)
        self.cv2 = nn.Sequential(RepNCSP(c3//2,c4,number),ConvBlock(c4,c4,3,1))
        self.cv3 = nn.Sequential(RepNCSP(c4,c4,number),ConvBlock(c4,c4,3,1))

        self.cv4 = ConvBlock(c3+ (2*c4),c2,1,1)

    
    def forward(self,x):

        y = list(torch.chunk(self.cv1(x),2,1))

        y.extend(block(y[-1]) for block in [self.cv2,self.cv3])

        return self.cv4(torch.cat(y,dim = 1))


class Concat(nn.Module):

    def __init__(self,dim = 1):
        super().__init__()

        self.dim = dim
    
    def forward(self,x):
        return torch.cat(x,dim = self.dim)



        


class ADown(nn.Module):

    def __init__(self, in_ch,out_ch):
        super().__init__()
        self.c = in_ch//2
        self.cv1 = ConvBlock(in_ch//2,self.c,3,2,1)
        self.cv2 = ConvBlock(in_ch//2,self.c,1,1,0)

        self.avg = nn.AvgPool2d(2,1,0,False,True)
        self.maxpooling = nn.MaxPool2d(3,2,1,1)

    
    def forward(self,x):

        out1,out2 = torch.chunk(self.avg(x),2,1)

        out1 = self.cv1(out1)
        out2 = self.maxpooling(out2)
        out2 = self.cv2(out2)

        return torch.cat([out1,out2],dim = 1)



class SP(nn.Module):
    def __init__(self,kernel_size = 3, stride = 1):
        super().__init__()

        self.max = nn.MaxPool2d(kernel_size=kernel_size,stride=stride,padding = kernel_size//2)

    def forward(self,x):

        return self.max(x)


class SPPELAN(nn.Module):

    def __init__(self,c1,c2,c3):
        super().__init__()


        self.cv1 = ConvBlock(c1,c3,1,1)

        self.sp1 = SP(5)
        self.sp2 = SP(5)
        self.sp3 = SP(5)

        self.cv2 = ConvBlock(4*c3,c2,1,1)

    def forward(self,x):

        y = [self.cv1(x)]

        y.extend(block(y[-1]) for block in [self.sp1,self.sp2,self.sp3])


        out = torch.cat(y,dim = 1)
        print(out.size())

        return self.cv2(out)

class SPPF(nn.Module):

    def __init__(self,c1,c2,c3):
        super().__init__()

        self.cv1 = ConvBlock(c1,c3,3,1,0)

        self.sp1 = SP(5)
        self.sp2 = SP(5)
        self.sp3 = SP(5) 

        self.cv2 = ConvBlock(c3*4,c2,3,1,0)

    def forward(self,x):

        y = [self.cv1(x)]

        y.extend(block(y[-1]) for block in [self.sp1,self.sp2,self.sp3])

        out = torch.concat(y,dim = 1)

        return self.cv2(out)


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class BackBoneUltraly(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = ConvBlock(3,64,3,2)
        self.conv2 = ConvBlock(64,128,3,2)

        self.elan1 = RepNCSPELAN4(128,256,128,64,1)
        self.adown1 = ADown(256,256)


        self.elan2 = RepNCSPELAN4(256,512,256,128,1)
        self.adown2 = ADown(512,512)

        self.elan3 = RepNCSPELAN4(512,512,512,256,1)

        self.spp = SPPELAN(512,512,256)

    
    def forward(self,x):

        out = self.conv1(x)
        out = self.conv2(out)

        



    


class BackBone(nn.Module):

    def __init__(self):
        super().__init__()

        self.cv1 = ConvBlock(3,64,3,2)
        self.cv2 = ConvBlock(64,128,3,2)

        self.elan1 = RepNCSPELAN4(128,256,128,64,1)

        self.down = ADown(256,256)

        self.elan2 = RepNCSPELAN4(256,512,256,128,1)

        self.down2 = ADown(512,512)

        self.elan3 = RepNCSPELAN4(512,512,512,256,1)

        self.down3 = ADown(512,512)

        self.elan4 = RepNCSPELAN4(512,512,512,256,1)

    def forward(self,x):

        
        cv_out = self.cv1(x)
        cv_out = self.cv2(cv_out)

        elan1 = self.elan1(cv_out)

        down = self.down(elan1)

        elan2 = self.elan2(down)

        down2 = self.down2(elan2)

        elan3 = self.elan3(down2)

        down3 = self.down3(elan3)

        elan4 = self.elan4(down3)

        return x,elan2,elan3,elan4



class Neck(nn.Module):
    def __init__(self):
        super().__init__()

        self.sppelan = SPPELAN(512,512,256)
        self.Upsample1 = nn.Upsample(scale_factor = 2.)
        self.elan1 = RepNCSPELAN4(1024,512,512,256,1)
        self.Upsample2 = nn.Upsample(scale_factor = 2.)
        self.elan2 = RepNCSPELAN4(1024,256,256,128,1)

        self.down1 = ADown(256,256)
        self.elan3 = RepNCSPELAN4(768,512,512,256,1)
        self.down2 = ADown(512,512)
        self.elan4 = RepNCSPELAN4(1024,512,512,256)

    def forward(self,elan2,elan3,elan4):

        spp_out = self.sppelan(elan4)

        up_out = self.Upsample1(spp_out)

        cat = torch.cat([elan3,up_out],dim=1)

        elan_out = self.elan1(cat)

        
        up_out2 = self.Upsample2(elan_out)

        cat2 = torch.cat([elan2,up_out2],dim=1)

        elan_out2 = self.elan2(cat2) # will return

        down_out = self.down1(elan_out2)

        cat_down = torch.cat([elan_out,down_out],dim = 1)

        elan_out3 = self.elan3(cat_down) # will return

        down_out2 = self.down2(elan_out3) 

        cat_down2 = torch.cat([spp_out,down_out2],dim=1)


        elan_out4 = self.elan4(cat_down2)
        
        return elan_out2,elan_out3,elan_out4


class Auxiliary(nn.Module):
    def __init__(self):
        super(Auxiliary,self).__init__()

        self.conv1 = ConvBlock(3,64,3,2)
        self.conv2 = ConvBlock(64,128,3,2)

        self.elan1 = RepNCSPELAN4(128,256,128,64)
        self.down1 = ADown(256,256)
        self.cblinear1 = CBLinear(512,[256])
        self.cblinear2 = CBLinear(512,[256,512])
        self.cblinear3 = CBLinear(512,[256,512,512])

        self.fuse1 = CBFuse([0,0,0])
        self.fuse2 = CBFuse([1,1])
        self.fuse3 = CBFuse([2])

        self.elan2 = RepNCSPELAN4(256,512,256,128)
        self.down2 = ADown(512,512)

        self.elan3 = RepNCSPELAN4(512,512,512,256)

        self.down3 = ADown(512,512)

        self.elan4 = RepNCSPELAN4(512,512,512,256)


    def forward(self,silence,bbelan2,bbelan3,bbelan4):

        conv_out = self.conv1(silence)
        conv_out = self.conv2(conv_out)
        conv_out = self.elan1(conv_out)
        conv_out = self.down1(conv_out)
        cb_out1 = self.cblinear1(bbelan2)
        cb_out2 = self.cblinear2(bbelan3)
        cb_out3 = self.cblinear3(bbelan4)

        fuseout1 = self.fuse1([cb_out1,cb_out2,cb_out3,conv_out])

        elan_out = self.elan2(fuseout1)

        elan_out_down = self.down2(elan_out)


        fuseout2 = self.fuse2([cb_out2,cb_out3,elan_out_down])

        elan_out2 = self.elan3(fuseout2)
        elan_out2_down = self.down3(elan_out2)

        fuseout3 = self.fuse3([cb_out3,elan_out2_down])

        elan_out3 = self.elan4(fuseout3)

        return [elan_out,elan_out2,elan_out3]



class DualDDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 2  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4), max((ch[self.nl], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(ConvBlock(x, c2, 3), ConvBlock(c2, c2, 3, group=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(ConvBlock(x, c3, 3), ConvBlock(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(ConvBlock(x, c4, 3), ConvBlock(c4, c4, 3, group=4), nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl:])
        self.cv5 = nn.ModuleList(
            nn.Sequential(ConvBlock(x, c5, 3), ConvBlock(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        d1 = []
        d2 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
        if self.training:
            return [d1, d2]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1)]
        return y if self.export else (y, [d1, d2])


class Detect(nn.Module):
    # YOLO Detect head for detection models
   

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super(Detect,self).__init__()
        self.training = False
        self.dynamic = False  # force grid reconstruction
        self.export = True  # export mode
        self.shape = None
        self.anchors = torch.empty(0)  # init
        self.strides = torch.empty(0)  # init
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((ch[0] // 4, self.reg_max * 4, 16)), max((ch[0], min((self.nc * 2, 100))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(ConvBlock(x, c2, 3), ConvBlock(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(ConvBlock(x, c3, 3), ConvBlock(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
           
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
   
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)

        
        
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


        

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x) 

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)

class Backboneultra(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(3,64,3,2)
        self.conv2 = ConvBlock(64,128,3,2)

        self.elan1 = RepNCSPELAN4(128,256,128,64)
        self.adown1 = ADown(256,256)

        self.elan2 = RepNCSPELAN4(256,512,256,128)
        self.adown2 = ADown(512,512)

        self.elan3 = RepNCSPELAN4(512,512,512,256)
        self.adown3 = ADown(512,512)

        self.elan4 = RepNCSPELAN4(512,512,512,256)

        self.spp = SPPELAN(512,512,256)


    def forward(self,x):

        out = self.conv1(x)
        out = self.conv2(out)

        out = self.elan1(out)
        out = self.adown1(out)

        elan_out1 = self.elan2(out)

        down = self.adown2(elan_out1)

        elan_out2 = self.elan3(down)


        down2 = self.adown3(elan_out2)


        out = self.elan4(down2)

        spp_out = self.spp(out)


        return elan_out1,elan_out2,spp_out



class Head(nn.Module):

    def __init__(self):
        super().__init__()


        self.up1 = nn.Upsample(scale_factor=2,mode="nearest")

        self.elan1 = RepNCSPELAN4(1024,512,512,256)

        self.up2 = nn.Upsample(scale_factor=2,mode="nearest")

        self.elan2 = RepNCSPELAN4(1024,256,256,128)

        self.adown1 = ADown(256,256)

        self.elan3 = RepNCSPELAN4(768,512,512,256)

        self.adown2 = ADown(512,512)

        self.elan4 = RepNCSPELAN4(1024,512,512,256)

    
    def forward(self,spp_out,elan_out2,elan_out1):

        up_out1 = self.up1(spp_out)

        cat_out = torch.cat([up_out1,elan_out2],dim=1)

        elan_out_head1 = self.elan1(cat_out)

        up_out2 = self.up2(elan_out_head1)

        cat_out2 = torch.cat([up_out2,elan_out1],dim=1)

        elan_out_head2 = self.elan2(cat_out2)

        adown_out1 = self.adown1(elan_out_head2)

        cat_out3 = torch.cat([adown_out1,elan_out_head1],dim=1)

        elan_out_head3 = self.elan3(cat_out3)

        adown_out2 = self.adown2(elan_out_head3)

        cat_out4 = torch.cat([adown_out2,spp_out],dim=1)

        elan_out_head4 = self.elan4(cat_out4)


        return (elan_out_head2,elan_out_head3,elan_out_head4)

class Model(nn.Module):

    def __init__(self,nc) -> None:
        super().__init__()

        self.bb = BackBone()
        self.aux = Auxiliary()
        self.neck = Neck()
        self.nc = nc
        self.detect = Detect(self.nc,[512,512,256,256,512,512])

       

        forward = self.detect.forward()
        
    def forward(self,x):

        out_bb = self.bb(x)
        out_aux = self.aux(*out_bb)
        out_neck = self.neck(*out_bb[1:])
        out_aux.extend(out_neck)
        
        return self.detect(out_aux)



class ModelUltra(nn.Module):

    fp16 = False

    def __init__(self,nc = 80,detect_ultra = False):
        super().__init__()

        self.bb = Backboneultra()
        self.head = Head()

        if detect_ultra:
            self.detect = Detect(nc=nc,ch = [256,512,512])

        else:
            self.detect = Detect(80,[256,512,512])

        self.detect.stride = torch.tensor([32,32,32])

        self.stride = self.detect.stride

        self.strides = self.detect.strides


    def forward(self,x):

        out_back = self.bb(x)

        out_head = self.head(*reversed(out_back))

        out_head = list(out_head)

        detects = self.detect(out_head)

    
        return detects







