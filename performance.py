# my_performance.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_curve, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import numpy as np
from model import CNN_Mamba  # 确保你的模型定义在 model.py 中
import torchvision.models as models
from tabulate import tabulate

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 设置中文字体为文泉驿正黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 获取当前工作目录
current_dir = os.getcwd()

# 设置测试数据集路径
test_data_dir = os.path.join(current_dir, 'test')

# 获取文件夹名称作为类别名称
class_names = [d for d in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, d))]
class_names.sort()  # 可选：对类别名称进行排序

# 创建测试数据集
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)

# 创建数据加载器
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型并将其移动到设备上
def load_model(model_def, weights_path=None, n_class=len(class_names)):
    model = model_def(n_class)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)  # 确保模型在正确的设备上
    model.eval()
    return model

# 获取模型预测结果
def get_predictions(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_preds_prob = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds_prob = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds_prob.extend(preds_prob)
    return all_preds, all_labels, all_preds_prob

# 绘制 ROC 曲线和计算 AUC（多分类问题）
def plot_multiclass_roc_curve(labels, preds_prob, n_classes, class_names, model_name):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels == i, preds_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve (Multiclass)')
    plt.legend(loc='lower right')
    plt.savefig(f'{model_name}_roc.png')  # 保存图表
    plt.show()

    return roc_auc  # 返回每个类别的 AUC 值

# 绘制 PR 曲线和计算 PR 值（多分类问题）
def plot_multiclass_pr_curve(labels, preds_prob, n_classes, class_names, model_name):
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels == i, preds_prob[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
    
    plt.figure()
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label=f'{class_names[i]} (PR AUC = {pr_auc[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} PR Curve (Multiclass)')
    plt.legend(loc='best')
    plt.savefig(f'{model_name}_pr.png')  # 保存图表
    plt.show()

    return pr_auc  # 返回每个类别的 PR 值

# 定义对比算法的模型定义函数
def model_resnet18(n_class):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_class)
    return model

def model_vgg16(n_class):
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_class)
    return model

def model_densenet121(n_class):
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, n_class)
    return model

def model_mobilenet_v2(n_class):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_class)
    return model

def model_efficientnet_b0(n_class):
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_class)
    return model

def model_cnn_mamba(n_class):
    return CNN_Mamba(n_class=n_class, d_model=512, n_mamba=2)

# 比较多个模型
def compare_models(models_def, models_weights, model_names):
    results = []
    labels = None

    for model_def, weights_path, model_name in zip(models_def, models_weights, model_names):
        model = load_model(model_def, weights_path)
        preds, labels, preds_prob = get_predictions(model, test_loader, device)

        # 计算评估指标
        report = classification_report(labels, preds, target_names=class_names, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f_measure = report['weighted avg']['f1-score']
        accuracy = report['accuracy']

        # 绘制 ROC 和 PR 曲线并计算 AUC 和 PR 值
        roc_auc = plot_multiclass_roc_curve(np.array(labels), np.array(preds_prob), len(class_names), class_names, model_name)
        pr_auc = plot_multiclass_pr_curve(np.array(labels), np.array(preds_prob), len(class_names), class_names, model_name)

        # 存储结果
        results.append({
            'Model': model_name,
            'Precision': precision,
            'Recall': recall,
            'F-Measure': f_measure,
            'Accuracy': accuracy,
            'ROC AUC (avg)': np.mean(list(roc_auc.values())),
            'PR AUC (avg)': np.mean(list(pr_auc.values()))
        })

    # 打印结果表格
    print(tabulate(results, headers='keys', tablefmt='grid', floatfmt='.4f'))

    return results

# 主函数
def main():
    # 定义模型定义函数、权重路径和名称
    models_def = [
        model_resnet18,
        model_vgg16,
        model_densenet121,
        model_mobilenet_v2,
        model_efficientnet_b0,
        model_cnn_mamba
    ]
    models_weights = [
        'best_resnet18.pt',  # 确保这些文件存在
        'best_vgg16.pt',
        'best_densenet121.pt',
        'best_mobilenet_v2.pt',
        'best_efficientnet_b0.pt',
        'best_cnn_mamba.pt'
    ]
    model_names = [
        'ResNet-18',
        'VGG-16',
        'DenseNet-121',
        'MobileNetV2',
        'EfficientNet-B0',
        'CNN-Mamba'
    ]

    # 比较模型
    results = compare_models(models_def, models_weights, model_names)

if __name__ == "__main__":
    main()