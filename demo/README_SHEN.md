# NUDT - 成像制导技术大作业
基于深度学习的航拍图像目标识别算法研究 rotated_retinaNet


- [ ] 目标识别算法 = 算法名称 + 算法思想 + 算法实现流程
- [ ] 算法的认识、（优缺点）评价、算法实现的体会
- [x] MMRotate 只是框架不是算法，是一款基于PyTorch的旋转框检测的开源工具箱
- [ ] 所选算法，对15类目标，检测精度（AP） & 平均检测精度（mAP）
- [ ] 典型场景的检测可视化结果（选几张典型检测图片）
- [ ] （缺少计算资源，可不进行训练）使用MMRotate代码库中提供的官方权重文件复现论文结果


```
.py         配置文件
.pth        模型权重文件

hbb means the input of the assigner is the predicted box and the horizontal box that can surround the GT.
obb means the input of the assigner is the predicted box and the GT.
They can be switched by assign_by_circumhbbox in RotatedRetinaHead.

r50         使用 ResNet-50 作为主干网络来提取特征

一种用于多尺度目标检测的结构，可以有效地处理不同尺度的目标。
fpn（Feature Pyramid Network） 是一种用于多尺度特征处理的网络架构，通过自顶向下的上采样和自底向上的特征融合来提升目标检测精度。
refpn（Refined Feature Pyramid Network） 是 FPN 的一种改进版本，加入了对多尺度特征的精细化处理，进一步提高了模型在复杂场景中的表现。


Angle
oc          Counterclockwise，逆时针方向
le90        less than or equal to 90°，旋转框的角度是小于或等于 90 度的


lr = 学习率调度（Learning Rate Scheduling）
1x          表示 1倍的训练周期（通常指的是训练一个标准的训练周期（比如在 12、24 或 36 个 epoch 内训练）。这通常意味着每个训练阶段使用一个标准的训练时间长度）
6x          表示 6倍的训练周期

```


## 本地环境
```
CPU：Intel(R) Core(TM) Ultra 5 125H
- 基准速度:1.20GHz
- 插槽:1
- 内核:14
- 逻辑处理器:18
- 虚拟化:已启用
- L1 缓存:1.4 MB
- L2 缓存:14.0 MB
- L3 缓存:18.0 MB

内存：32GB
GPU：Intel(R) Arc(TM) Graphics 15.8GB
NPU：Intel(R) Al Boost
```




```python

mim install mmcv-full -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install "mmdet<3.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载 oriented_rcnn_r50_fpn_1x_dota_le90.py 和 oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth 这两个文件
mim download mmrotate --config oriented_rcnn_r50_fpn_1x_dota_le90 --dest .

# 如果本地没有 Nvidia 的 GPU，在运行之前需要修改 demo/image_demo.py parse_args() 方法中的 '--device', default值改为'cpu'，即：
# parser.add_argument('--device', default='cpu', help='Device used for inference')
python demo/image_demo.py demo/images/P1781.png demo/configs/oriented_rcnn_r50_fpn_1x_dota_le90.py demo/checkpoints/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --out-file demo/results/P1781.png
--device cpu



python tools/test.py demo_shenli/oriented_rcnn_r50_fpn_1x_dota_le90.py checkpoints/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --eval bbox
python tools/test.py configs/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_le90.py checkpoints/rotated_retinanet_hbb_r50_fpn_1x_dota_oc-e8a7c7df.pth --show --eval bbox

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


- [DOTA(Dataset for Object Detection in Aerial Images) Dataset Download](https://captain-whu.github.io/DOTA/dataset.html)



