# Improved-MobileNetV2
基于改进MobileNetV2与注意力机制的小样本植物病害识别

基于预训练MobileNetV2，冻结浅层微调高层；结合动态增强（裁剪、翻转、擦除、颜色抖动）与Mixup插值增强，平滑决策边界；嵌入CBAM双注意力聚焦病害区域，增强关键特征提取的鲁棒性；采用余弦退火调度与混合精度训练优化收敛；通过Grad-CAM实现可视化解释。

数据集来源：https://tianchi.aliyun.com/dataset/160100
手动下载数据集后将其命名为dataset放入项目中
