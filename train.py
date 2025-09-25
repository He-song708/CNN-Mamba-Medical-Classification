# train.py
import torch
import yaml
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from dataset import build_loader
from model import CNN_Mamba
import torchvision.models as models
import os
from torch.optim.lr_scheduler import CosineAnnealingLR


def load_model(model_name, num_classes, pretrained=True):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "cnn_mamba":
        model = CNN_Mamba(n_class=num_classes, d_model=512, n_mamba=2, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据加载器
    train_ld, val_ld, classes = build_loader(cfg["data_dir"], cfg["batch_size"], cfg["num_workers"])
    num_classes = len(classes)

    # 支持的模型列表
    model_names = cfg.get("model_names", ["resnet18", "vgg16", "densenet121", "mobilenet_v2", "efficientnet_b0", "cnn_mamba"])

    for model_name in model_names:
        print(f"Training {model_name}...")
        # 模型
        model = load_model(model_name, num_classes, pretrained=cfg.get("pretrained", True)).to(device)
        # 损失函数和优化器
        crit = nn.CrossEntropyLoss()
        opt = optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
        scaler = GradScaler()
        scheduler = CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=1e-6)  # 添加学习率调度器

        best_acc = 0.0
        for epoch in range(1, cfg["epochs"] + 1):
            # 训练
            model.train()
            tot_loss, tot = 0, 0
            for imgs, lbls in train_ld:
                imgs, lbls = imgs.to(device), lbls.to(device)
                opt.zero_grad()
                with autocast():
                    logits = model(imgs)
                    loss = crit(logits, lbls)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot_loss += loss.item() * imgs.size(0)
                tot += imgs.size(0)
            print(f"[{epoch}] {model_name} train loss: {tot_loss / tot:.4f}")

            # 验证
            model.eval()
            correct, tot = 0, 0
            with torch.no_grad():
                for imgs, lbls in val_ld:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    logits = model(imgs)
                    pred = logits.argmax(1)
                    correct += (pred == lbls).sum().item()
                    tot += lbls.size(0)
            acc = correct / tot
            print(f"[{epoch}] {model_name} val acc: {acc * 100:.2f}%")

            # 更新学习率
            scheduler.step()

            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                save_path = f"best_{model_name}.pt"
                torch.save(model.state_dict(), save_path)
                print(f"{model_name} model saved to {save_path}!")


if __name__ == "__main__":
    try:
        cfg = yaml.safe_load(open("configs/default.yaml", "r", encoding="utf-8"))
    except FileNotFoundError:
        print("⚠️  未找到 configs/default.yaml，使用脚本内置默认参数。")
        cfg = {
            "data_dir": "/home/hesong/fetal_data",
            "batch_size": 32,
            "num_workers": 4,
            "epochs": 40,
            "lr": 3e-4,
            "pretrained": True,
            "model_names": ["resnet18", "vgg16", "densenet121", "mobilenet_v2", "efficientnet_b0", "cnn_mamba"]
        }
    main(cfg)