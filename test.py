# 这个code block主要是用来检查包和库的安装是不是完全，和requirements.txt中的区别在于多了一些作者认为比较常见的库
# 我在运行这一步的时候，安装了cupy-cuda92等

import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import torch
torch.cuda.empty_cache() # 清空一下GPU缓存
# 加载预训练模型
faster_rcnn = FasterRCNNVGG16()
print("成功装载预训练模型VGG16！")
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load('./三种不同finetuing的模型/chainer_best_model_converted_to_pytorch_0.7053.pth')
opt.caffe_pretrain=True # this model was trained from caffe-pretrained model
# 这加载测试图像，传的参数就是待测试的图像相对路径
img = read_image('misc/chrismas.JPG')
img = t.from_numpy(img)[None]
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))