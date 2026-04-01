import os
import cv2
import requests
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import zipfile
# PyTorch相关
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import random

# 随机种子函数（仅在训练时调用）
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------- 0. 配置matplotlib中文字体 --------------------------
def setup_matplotlib_chinese():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'

setup_matplotlib_chinese()

# -------------------------- 1. 模型/训练配置 --------------------------
# 模型配置
MODEL_TYPE = "MobileNetV2"
FREEZE_BACKBONE = True
NUM_FREEZE_LAYERS = 10
# 训练配置
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
EPOCHS = 20
WEIGHT_DECAY = 1e-4
# 数据配置
DATASET_UNZIP_PATH = "dataset"  #数据集位置
IMAGE_SIZE = (224, 224)
SAMPLE_RATIO = 0.1
# 创新点开关
USE_MIXUP = True                    # 是否使用Mixup     True    False
MIXUP_ALPHA = 0.1
USE_CBAM = True                     # 是否使用CBAM      True    False
CBAM_REDUCTION = 16
# 学习率调度器选择：'step' 或 'cosine' 或 'cosine_restart'
LR_SCHEDULER = 'cosine'
# 版本哈希
CONFIG_STR = f"{MODEL_TYPE}_{FREEZE_BACKBONE}_{NUM_FREEZE_LAYERS}_{LEARNING_RATE}_{SAMPLE_RATIO}_{USE_MIXUP}_{USE_CBAM}_{LR_SCHEDULER}"
MODEL_VERSION = hashlib.md5(CONFIG_STR.encode()).hexdigest()[:8]
MODEL_SAVE_PATH = f"plant_disease_{MODEL_VERSION}.pth"

# -------------------------- 2. 环境检测 --------------------------
def check_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f" PyTorch GPU可用，设备：{torch.cuda.get_device_name(0)}")
        print(f" CUDA版本：{torch.version.cuda}")
    else:
        print(" PyTorch GPU不可用，将使用CPU训练（速度较慢）")
    return device

device = check_gpu()

# -------------------------- 3. 数据集下载 --------------------------
def auto_download_tianchi_dataset():
    if os.path.exists(DATASET_UNZIP_PATH):
        print(f" 数据集已存在，跳过下载")
        return

    TIANCHI_DOWNLOAD_URL = "https://tianchi.aliyun.com/dataset/160100"
    DATASET_SAVE_PATH = "plant_disease_small.zip"
    print(f" 开始下载数据集...")
    try:
        response = requests.get(TIANCHI_DOWNLOAD_URL, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        with open(DATASET_SAVE_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress = (downloaded_size / total_size) * 100
                    print(f"\r下载进度：{progress:.1f}%", end="")
        print(f"\n 数据集下载完成")
        with zipfile.ZipFile(DATASET_SAVE_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATASET_UNZIP_PATH)
        os.remove(DATASET_SAVE_PATH)
        print(f" 解压完成，已删除压缩包")
    except Exception as e:
        print(f"\n 下载/解压失败：{e}，请打开https://tianchi.aliyun.com/dataset/160100手动下载数据集")
        exit()

auto_download_tianchi_dataset()

# -------------------------- 4. 数据集加载（含Mixup） --------------------------
class PlantDiseaseDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None, use_mixup=False):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.use_mixup = use_mixup

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)

        if self.use_mixup:
            idx2 = np.random.randint(0, len(self.img_paths))
            img2_path = self.img_paths[idx2]
            label2 = self.labels[idx2]
            img2 = cv2.imread(img2_path)
            if img2 is None:
                img2 = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            if self.transform:
                img2 = self.transform(img2)

            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
            img_mix = lam * img + (1 - lam) * img2
            return img_mix, label, label2, lam
        else:
            return img, label, -1, 0.0

def load_dataset():
    img_paths = []
    labels = []
    class_names = []
    for root, dirs, files in os.walk(DATASET_UNZIP_PATH):
        if dirs and not class_names:
            class_names = sorted(dirs)
            print(f" 共{len(class_names)}个病害类别：{class_names}")
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                class_name = os.path.basename(root)
                if class_name not in class_names:
                    continue
                class_idx = class_names.index(class_name)
                img_paths.append(os.path.join(root, file))
                labels.append(class_idx)
    labels = np.array(labels, dtype=np.int64)
    x_train_paths, x_test_paths, y_train, y_test = train_test_split(
        img_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_sample_size = int(len(x_train_paths) * SAMPLE_RATIO)
    test_sample_size = int(len(x_test_paths) * SAMPLE_RATIO)
    x_train_paths = x_train_paths[:train_sample_size]
    y_train = y_train[:train_sample_size]
    x_test_paths = x_test_paths[:test_sample_size]
    y_test = y_test[:test_sample_size]

    print(f" 数据集加载完成（{SAMPLE_RATIO * 100}%数据）：")
    print(f"   - 训练集：{len(x_train_paths)}张 | 测试集：{len(x_test_paths)}张")
    return x_train_paths, x_test_paths, y_train, y_test, class_names, len(class_names)

x_train_paths, x_test_paths, y_train, y_test, class_names, num_classes = load_dataset()

# 数据增强
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE[0] + 32, IMAGE_SIZE[1] + 32)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = PlantDiseaseDataset(x_train_paths, y_train, transform=train_transform, use_mixup=USE_MIXUP)
test_dataset = PlantDiseaseDataset(x_test_paths, y_test, transform=test_transform, use_mixup=False)

def mixup_collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels1 = torch.tensor([item[1] for item in batch], dtype=torch.long)
    labels2 = torch.tensor([item[2] for item in batch], dtype=torch.long)
    lams = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    return imgs, labels1, labels2, lams

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=mixup_collate_fn
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=mixup_collate_fn
)

# -------------------------- 5. CBAM注意力模块 --------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# -------------------------- 6. 模型构建（仅MobileNetV2） --------------------------
class MobileNetV2WithCBAM(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True, num_freeze_layers=10, reduction=16):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=True)
        in_channels = self.backbone.classifier[1].in_features
        self.cbam = CBAM(1280, reduction=reduction)  # MobileNetV2 最后输出通道=1280
        self.classifier = nn.Linear(in_channels, num_classes)

        if freeze_backbone:
            for param in self.backbone.features[:num_freeze_layers].parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.cbam(x)  
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def build_model(num_classes):
    if USE_CBAM:
        model = MobileNetV2WithCBAM(num_classes,
                                    freeze_backbone=FREEZE_BACKBONE,
                                    num_freeze_layers=NUM_FREEZE_LAYERS,
                                    reduction=CBAM_REDUCTION)
    else:
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        if FREEZE_BACKBONE:
            for param in model.features[:NUM_FREEZE_LAYERS].parameters():
                param.requires_grad = False
    return model.to(device)

# -------------------------- 7. 模型加载或训练 --------------------------
if os.path.exists(MODEL_SAVE_PATH):
    print(f"\n 发现已训练的{MODEL_TYPE}模型（版本：{MODEL_VERSION}），直接加载...")
    model = build_model(num_classes)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    # 计算测试集准确率
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels1, _, _ in test_loader:
            images = images.to(device)
            labels = labels1.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    test_acc = 100 * correct_val / total_val
    print(f" 测试集准确率：{test_acc:.2f}%")
else:
    # 训练开始，固定随机种子
    set_seed(42)
    print(f"\n 构建{MODEL_TYPE}模型（版本：{MODEL_VERSION}）：")
    print(f"   - 类别数：{num_classes}")
    print(f"   - 冻结主干：{FREEZE_BACKBONE} | 冻结层数：{NUM_FREEZE_LAYERS}")
    print(f"   - Mixup：{USE_MIXUP} | CBAM：{USE_CBAM} | 学习率调度：{LR_SCHEDULER}")
    model = build_model(num_classes)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if LR_SCHEDULER == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    elif LR_SCHEDULER == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    elif LR_SCHEDULER == 'cosine_restart':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    else:
        raise ValueError(f"不支持的学习率调度器：{LR_SCHEDULER}")

    scaler = GradScaler() if torch.cuda.is_available() else None

    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    print(f"\n 开始训练（{device}）...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels1, labels2, lams in train_loader:
            images = images.to(device)
            labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            lams = lams.to(device)

            optimizer.zero_grad()

            if scaler:
                with autocast():
                    outputs = model(images)
                    loss = 0.0
                    for i in range(images.size(0)):
                        if lams[i] > 0:
                            loss += lams[i] * F.cross_entropy(outputs[i:i+1], labels1[i:i+1]) + \
                                    (1 - lams[i]) * F.cross_entropy(outputs[i:i+1], labels2[i:i+1])
                        else:
                            loss += F.cross_entropy(outputs[i:i+1], labels1[i:i+1])
                    loss = loss / images.size(0)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = 0.0
                for i in range(images.size(0)):
                    if lams[i] > 0:
                        loss += lams[i] * F.cross_entropy(outputs[i:i+1], labels1[i:i+1]) + \
                                (1 - lams[i]) * F.cross_entropy(outputs[i:i+1], labels2[i:i+1])
                    else:
                        loss += F.cross_entropy(outputs[i:i+1], labels1[i:i+1])
                loss = loss / images.size(0)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels1.size(0)
            correct_train += (predicted == labels1).sum().item()

        # 验证
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels1, _, _ in test_loader:
                images = images.to(device)
                labels = labels1.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_train_loss = train_loss / total_train
        train_acc = 100 * correct_train / total_train
        avg_val_loss = val_loss / total_val
        val_acc = 100 * correct_val / total_val

        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        scheduler.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}], 训练损失：{avg_train_loss:.4f}, 训练准确率：{train_acc:.2f}%, 验证准确率：{val_acc:.2f}%, 学习率：{optimizer.param_groups[0]['lr']:.6f}")

    final_acc = val_acc
    print("\n 最终评估结果：")
    print(f"测试集准确率（最后一个epoch）：{final_acc:.2f}%")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f" 模型已保存为：{MODEL_SAVE_PATH}")

    # 训练曲线可视化
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_accs, label="训练准确率", color="blue")
    plt.plot(val_accs, label="验证准确率", color="red")
    plt.title("准确率变化")
    plt.xlabel("训练轮次（Epoch）")
    plt.ylabel("准确率（%）")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(train_losses, label="训练损失", color="blue")
    plt.plot(val_losses, label="验证损失", color="red")
    plt.title("损失值变化")
    plt.xlabel("训练轮次（Epoch）")
    plt.ylabel("损失值")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"training_history_{MODEL_VERSION}.png", dpi=100, bbox_inches='tight')
    plt.show()
    print(f" 训练曲线已保存为：training_history_{MODEL_VERSION}.png")

# -------------------------- 8. 单张图片预测 --------------------------
def predict_single_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f" 无法读取图片：{img_path}")
        return None, 0.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = test_transform
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred_idx = torch.max(outputs, 1)
        pred_class = class_names[pred_idx.item()]
        confidence = torch.softmax(outputs, dim=1)[0][pred_idx].item()

    plt.imshow(img)
    plt.title(f"预测类别：{pred_class}\n置信度：{confidence:.4f}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return pred_class, confidence

# -------------------------- 9. Grad-CAM可视化 --------------------------
def grad_cam(model, img_tensor, target_layer, device):
    model.eval()
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    handle = target_layer.register_forward_hook(forward_hook)
    output = model(img_tensor)
    pred_idx = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred_idx].backward()

    grad = gradients[0].detach().cpu().numpy()[0]
    activation = activations[0].detach().cpu().numpy()[0]

    # 计算权重
    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(activation.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activation[i, :, :]

    #归一化
    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    handle.remove()
    return cam, pred_idx

def visualize_gradcam(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("无法读取图片")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()
    h, w = original_img.shape[:2]

    transform = test_transform
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 选择正确的层（MobileNetV2 最后一个有效特征层）
    if USE_CBAM:
        target_layer = model.backbone.features[-1]
    else:
        target_layer = model.features[-1]

    cam, pred_idx = grad_cam(model, img_tensor, target_layer, device)
    pred_class = class_names[pred_idx]

    #生成热力图
    cam = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 融合图片
    superimposed = heatmap * 0.4 + np.float32(original_img) * 0.6
    superimposed = np.uint8(superimposed)

    # 展示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    true_label = os.path.basename(os.path.dirname(img_path))
    plt.title(f"原始图片\n真实类别：{true_label}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed)
    plt.title(f"Grad-CAM\n预测类别：{pred_class}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 随机选取一张测试图片进行预测和Grad-CAM展示
test_img_paths = [os.path.join(root, f) for root, _, files in os.walk(DATASET_UNZIP_PATH) for f in files if f.lower().endswith(('.jpg','.jpeg','.png'))]
if test_img_paths:
    random_img = random.choice(test_img_paths)
    print(f"\n 随机测试图片：{random_img}")
    predict_single_image(random_img)
    visualize_gradcam(random_img)
