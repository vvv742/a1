import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from collections import defaultdict
import argparse
import timm
from asam import ASAM, SAM, NRASAM, NRSAM
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

# 注意：这里应该包含您的ASAM, SAM, NRASAM, NRSAM优化器类
# 由于您要求不包含优化器代码，此处省略
def add_laplacian_noise(data, scale=0.001):
    """使用PyTorch实现的拉普拉斯噪声"""
    # 创建拉普拉斯分布 (loc=均值, scale=尺度参数b)
    laplace_dist = torch.distributions.Laplace(
        loc=0.0,
        scale=scale
    )
    # 生成与data相同形状的噪声样本
    noise = laplace_dist.sample(data.shape)
    return data + noise

def add_gradient_noise(model, std=0.005):
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * std
                param.grad.add_(noise)

def add_uniform_noise(x, scale=0.01):
    """添加均匀分布的随机噪声，范围 [-scale, scale]"""
    noise = torch.empty_like(x).uniform_(-scale, scale)
    return torch.clamp(x + noise, 0.0, 1.0)


def load_data(dataset_name, batch_size=256, num_workers=2, args=None):
    """加载CIFAR10, CIFAR100或SVHN数据集"""
    # 数据集均值和标准差配置
    if dataset_name == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
    elif dataset_name == 'SVHN':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

#    def add_laplacian_noise_tensor(x, scale=0.1):
#        noise = torch.distributions.Laplace(0, scale).sample(x.shape).to(x.device)
#        return torch.clamp(x + noise, 0, 1)  # 保持像素值有效

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 转换为张量（值范围[0,1]）
        #transforms.Lambda(lambda x: add_uniform_noise(x, scale=args.uniform_noise_scale)),
        #transforms.Lambda(lambda x: torch.clamp(
        #    add_laplacian_noise(x, scale=args.laplace_noise_scale),
        #    0.0, 1.0
        #)),
        transforms.Normalize(mean, std),  # 归一化
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # 添加拉普拉斯噪声并裁剪到有效范围
        #transforms.Lambda(lambda x: torch.clamp(
        #    add_laplacian_noise(x, scale=args.laplace_noise_scale),
        #    0.0, 1.0
        #)),
        transforms.Normalize(mean, std)
    ])

    # 加载数据集
    if dataset_name in ['CIFAR10', 'CIFAR100']:
        dataset_class = CIFAR10 if dataset_name == 'CIFAR10' else CIFAR100
        train_set = dataset_class(root='./data', train=True, download=True, transform=train_transform)
        test_set = dataset_class(root='./data', train=False, download=True, transform=test_transform)
    else:  # SVHN
        train_set = SVHN(root='./data', split='train', download=True, transform=train_transform)
        test_set = SVHN(root='./data', split='test', download=True, transform=test_transform)

    # 创建DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, num_classes


# ===== 新增模型定义 =====
# ===== WideResNet-28-2 =====
class WideBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
             self.shortcut = nn.Sequential(
                 nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False)
             )

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        return out + shortcut


class WideResNet28_2(nn.Module):
    def __init__(self, num_classes=100, dropout=0.0):
        super().__init__()
        channels = [16, 32, 64]
        k = 2 # widen factor
        channels = [c * k for c in channels]

        self.conv1 = nn.Conv2d(3, channels[0], 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(channels[0], channels[0], 4, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(channels[0], channels[1], 4, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(channels[1], channels[2], 4, stride=2, dropout=dropout)
        self.bn = nn.BatchNorm2d(channels[2])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[2], num_classes)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride, dropout):
        layers = [WideBlock(in_ch, out_ch, stride, dropout)]
        for _ in range(num_blocks - 1):
            layers.append(WideBlock(out_ch, out_ch, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class ResNetCifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCifar, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet20(num_classes=10):
    return ResNetCifar(BasicBlock, [3, 3, 3], num_classes=num_classes)


def ResNet32(num_classes=10):
    return ResNetCifar(BasicBlock, [5, 5, 5], num_classes=num_classes)

class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, n_classes=10):
        super().__init__()
        self.embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.act = nn.GELU()
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for _ in range(depth)]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.act(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = int(round(in_ch * expand_ratio))
        self.use_res = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, 1, 0, bias=False),
                      nn.BatchNorm2d(hidden),
                      nn.ReLU6(inplace=True)]
        layers += [
                   # depthwise
                   nn.Conv2d(hidden, hidden, 3, stride, 1,
                   groups=hidden, bias=False),
                   nn.BatchNorm2d(hidden),
                   nn.ReLU6(inplace=True),
                   # pointwise-linear
                   nn.Conv2d(hidden, out_ch, 1, 1, 0, bias=False),
                   nn.BatchNorm2d(out_ch)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2_1_0(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        cfg = [ # (expand_ratio, out_channels, stride, num_blocks)
              (1, 16, 1, 1),
              (6, 24, 1, 2), # stride=1 保持 32×32
              (6, 32, 2, 3), # 16×16
              (6, 64, 2, 4), # 8×8
              (6, 96, 1, 3),
              (6, 160, 2, 3), # 4×4
              (6, 320, 1, 1),
        ]
        in_ch = 32
        layers = [nn.Conv2d(3, in_ch, 3, 1, 1, bias=False),
                 nn.BatchNorm2d(in_ch),
                 nn.ReLU6(inplace=True)]

        for t, c, s, n in cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(in_ch, c, stride, t))
                in_ch = c
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_ch, num_classes)

        self.conv_stem = self.features[0]  # 指向第一层 3×3 conv
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)
# ===== 新增模型定义结束 =====

def train(args):
    # 加载数据集
    train_loader, test_loader, num_classes = load_data(
        args.dataset, args.batch_size, args.num_workers, args
    )

    # 创建模型
    # ===== 修改部分开始 =====
    # 处理ResNet20/32模型
    if args.model == 'resnet20':
        model = ResNet20(num_classes=num_classes)
    elif args.model == 'resnet32':
        model = ResNet32(num_classes=num_classes)
    elif args.model == 'wrn28_2':
        model = WideResNet28_2(num_classes=num_classes)
    elif args.model == 'mobilenetv2_1_0':
        model = MobileNetV2_1_0(num_classes=num_classes)
    # 处理ViT-Tiny (patch=4)
    elif args.model == 'vit_tiny_patch4_32':
        model = timm.create_model(
            'vit_tiny_patch16_224',  # 使用基础模型
            pretrained=False,
            num_classes=num_classes,
            in_chans=3,
            img_size=32,  # 调整输入尺寸
            patch_size=4  # 调整patch大小
        )
    # 处理ConvMixer-256/8
    elif args.model == 'convmixer_256_8':
        model = ConvMixer(dim=256, depth=8, kernel_size=9, patch_size=4, n_classes=num_classes)
    else:
        # 其他模型使用timm创建
        model = timm.create_model(
            args.model,
            num_classes=num_classes,
            pretrained=False,
            in_chans=3
        )

    # 根据模型类型调整第一层
    if 'mobilenet' in args.model:
        first_conv = model.conv_stem
        model.conv_stem = nn.Conv2d(
            in_channels=3,
            out_channels=first_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        nn.init.kaiming_normal_(model.conv_stem.weight, mode='fan_out', nonlinearity='relu')
    elif 'resnet' in args.model and args.model not in ['resnet20', 'resnet32', 'resnet20-stir']:
        first_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=first_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
        model.maxpool = nn.Identity()
    elif 'densenet' in args.model:
        first_conv = model.features.conv0
        model.features.conv0 = nn.Conv2d(
            in_channels=3,
            out_channels=first_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        nn.init.kaiming_normal_(model.features.conv0.weight, mode='fan_out', nonlinearity='relu')
        model.features.pool0 = nn.Identity()
    # ===== 修改部分结束 =====

    model = model.cuda()

    # 创建优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # 创建Minimizer（ASAM/SAM/NRASAM/NRSAM）
    if args.minimizer in ['NRASAM', 'NRSAM']:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0,  # NRASAM要求momentum=0
            weight_decay=args.weight_decay
        )
        minimizer = eval(args.minimizer)(optimizer, model, rho=args.rho, eta=args.eta)
    elif args.minimizer in ['ASAM', 'SAM']:
        minimizer = eval(args.minimizer)(optimizer, model, rho=args.rho, eta=args.eta)
    else:  # SGDM
        minimizer = None

    # 学习率调度器
    milestones = [int(args.epochs * 0.2),
                  int(args.epochs * 0.5),
                  int(args.epochs * 0.8)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    # 损失函数（带标签平滑）
    if args.smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        cnt = 0
        if hasattr(minimizer, 'current_epoch'):
            minimizer.current_epoch = epoch
        if minimizer is None:
            # SGDM训练流程
            for inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()

                # 前向传播
                predictions = model(inputs)
                batch_loss = criterion(predictions, targets)

                # 反向传播
                optimizer.zero_grad()
                batch_loss.mean().backward()

                # 梯度噪声
                if args.noise_std > 0:
                    add_gradient_noise(model, std=args.noise_std)

                # 参数更新
                optimizer.step()

                # 统计指标
                with torch.no_grad():
                    train_loss += batch_loss.sum().item()
                    train_acc += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)

            train_loss /= cnt
            train_acc = 100.0 * train_acc / cnt
            print(f"Epoch: {epoch}, Train accuracy: {train_acc:6.2f}%, Train loss: {train_loss:8.5f}")

        else:
            # ASAM/SAM/NRASAM训练流程
            for inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()

                # 第一次前向-反向传播（计算原始梯度）
                predictions = model(inputs)
                batch_loss = criterion(predictions, targets)
                batch_loss.mean().backward()

                # 对抗步（计算扰动）
                minimizer.ascent_step()

                # 第二次前向-反向传播（计算扰动点梯度）
                criterion(model(inputs), targets).mean().backward()

                # 梯度噪声
                if args.noise_std > 0:
                    add_gradient_noise(model, std=args.noise_std)

                # 下降步（更新参数）
                minimizer.descent_step()

                # 统计指标
                with torch.no_grad():
                    train_loss += batch_loss.sum().item()
                    train_acc += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)

            train_loss /= cnt
            train_acc = 100.0 * train_acc / cnt
            print(f"Epoch: {epoch}, Train accuracy: {train_acc:6.2f}%, Train loss: {train_loss:8.5f}")

        # 更新学习率
        scheduler.step()

        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        cnt = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                predictions = model(inputs)
                loss = criterion(predictions, targets)

                test_loss += loss.sum().item()
                test_acc += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)

            test_loss /= cnt
            test_acc = 100.0 * test_acc / cnt
        if best_accuracy < test_acc:
            best_accuracy = test_acc
        print(
            f"Epoch: {epoch}, Test accuracy: {test_acc:6.2f}%, Test loss: {test_loss:8.5f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    print(f"Best test accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noise-Resistant Training")

    # 数据集参数
    parser.add_argument("--dataset", default="CIFAR10", type=str,
                        choices=['CIFAR10', 'CIFAR100', 'SVHN'],
                        help="Dataset to use: CIFAR10, CIFAR100 or SVHN")

    # 模型参数 - 添加新模型支持
    parser.add_argument("--model", default="vit_tiny_patch4_32", type=str,
                        help="Model architecture (e.g., wrn28_2,mobilenetv2_1_0,resnet18, densenet121, vit_tiny_patch4_32, convmixer_256_8, resnet20, resnet32)")

    # 优化器参数
    parser.add_argument("--minimizer", default="NRASAM", type=str,
                        choices=['ASAM', 'SAM', 'NRASAM', 'NRSAM', 'SGDM'],
                        help="Optimizer type: ASAM, SAM, NRSAM, NRASAM or SGDM")
    parser.add_argument("--lr", default=0.1, type=float, help="Initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")
    parser.add_argument("--rho", default=0.1, type=float, help="SAM radius parameter")
    parser.add_argument("--eta", default=0.01, type=float, help="ASAM scaling parameter")

    # 训练参数
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("--smoothing", default=0.1, type=float, help="Label smoothing factor")
    parser.add_argument("--noise_std", default=0.005, type=float, help="Gradient noise std dev")
    parser.add_argument("--num_workers", default=16, type=int, help="Number of data loader workers")
    parser.add_argument("--laplace_noise_scale", default=0, type=float,
                        help="Scale parameter for Laplacian noise added to input data")
    parser.add_argument("--uniform_noise_scale", default=0, type=float,
                        help="Scale of uniformly distributed noise added to input data")

    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("Training Configuration:")
    print(f"Dataset:       {args.dataset}")
    print(f"Model:         {args.model}")
    print(f"Optimizer:     {args.minimizer}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Epochs:        {args.epochs}")
    print(f"Rho:           {args.rho}")
    print(f"Eta:           {args.eta}")
    print(f"Noise Std:     {args.noise_std}")
    print(f"Laplace Noise Scale: {args.laplace_noise_scale}")  # 新增打印
    print(f"uniform noise to input data with scale={args.uniform_noise_scale}")
    print("=" * 50 + "\n")

    train(args)