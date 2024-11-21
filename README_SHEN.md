# NUDT - 成像制导技术大作业
基于深度学习的航拍图像目标识别算法研究

```python
# 如果本地没有 Nvidia 的 GPU，在运行之前需要修改 demo/image_demo.py parse_args() 方法中的 '--device', default值改为'cpu'，即：
# parser.add_argument('--device', default='cpu', help='Device used for inference')
python demo/image_demo.py demo/demo.jpg oriented_rcnn_r50_fpn_1x_dota_le90.py oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --out-file result/result00.jpg
python demo/image_demo.py demo/dota_demo.jpg oriented_rcnn_r50_fpn_1x_dota_le90.py oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --out-file result/result01.jpg
```




```python
import torch, torchvision, torchaudio

print("是否可用：", torch.cuda.is_available())          # 查看GPU是否可用
print("GPU数量：", torch.cuda.device_count())           # 查看GPU数量
print("torch方法查看CUDA版本：", torch.version.cuda)    # torch 方法查看 CUDA 版本
print("查看 torch 版本：", torch.__version__)
print("查看 torchvision 版本：", torchvision.__version__)
print("查看 torchaudio 版本：", torchaudio.__version__)
```