# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/3/26 21:40 
# ide： PyCharm

import torch
import numpy as np

from prettytable import PrettyTable
import matplotlib.pyplot as plt


def accuracy(output, target, topk=(1,)):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            correct_k /= batch_size
            res.append(correct_k)
        return res


class BestMetric():
    def __init__(self, metric_name, init_res=None, better='large') -> None:
        assert better in ['large', 'small']
        self.better = better
        if init_res is not None:
            self.init_res = init_res
        else:
            if better == 'small':
                self.init_res = float('inf')
            else:
                self.init_res = -float('inf')
        self.best_ep = -1
        self.best_res = self.init_res
        self.metric_name = metric_name

    def isbetter(self, new_res, old_res):
        if self.better == 'large':
            return new_res > old_res
        if self.better == 'small':
            return new_res < old_res

    def update(self, new_res, ep):
        if self.isbetter(new_res, self.best_res):
            self.best_res = new_res
            self.best_ep = ep
            return True
        return False

    def resume(self, best_res, best_ep):
        self.best_res = best_res
        self.best_ep = best_ep

    def __str__(self) -> str:
        return "best/{}: {:.5f}\t best/epoch: {}".format(self.metric_name, self.best_res, self.best_ep)

    def __repr__(self) -> str:
        return self.__str__()

    def summary(self) -> dict:
        return {
            'best/{}'.format(self.metric_name): self.best_res,
            'best/epoch': self.best_ep,
        }


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, class_nume: list = None, print_func=None):
        self.preds_list = []
        self.labels_list = []
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为5
        if class_nume is None:
            self.class_nume = list(range(num_classes))
        else:
            self.class_nume = class_nume  # 类别标签
        self.print_func = print_func

    @property
    def preds(self):
        return self.preds_list

    @property
    def labels(self):
        return self.labels_list

    def update(self, outputs, labels):
        ret, preds = torch.max(outputs.data, 1)
        self.preds_list.extend(preds.cpu().tolist())
        self.labels_list.extend(labels.cpu().tolist())
        for p, t in zip(preds.cpu().numpy(), labels.cpu().numpy()):  # pred为预测结果，labels为真实标签
            self.matrix[t, p] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):  # 计算指标函数
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n  # 总体准确率
        # print("the model accuracy is ", round(acc, 3))

        # kappa
        po = acc
        sum_pe = 0
        for i in range(self.num_classes):
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        pe = sum_pe / (n * n)
        kappa = (po - pe) / (1 - pe)
        # print("the model kappa is ", round(kappa, 3))

        # precision, recall, specificity
        table = PrettyTable()  # 创建一个表格
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):  # 精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[:, i]) - TP
            FN = np.sum(self.matrix[i, :]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.  # 每一类准确度
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            table.add_row([self.class_nume[i], Precision, Recall, Specificity])
        if self.print_func is not None:
            self.print_func(table)
        return {
            'Accuracy': round(acc, 3),
            'Precision': Precision,
            'Recall': Recall,
            'Specificity': Specificity,
            'Kappa': round(kappa, 3)
        }

    def plot(self, show=False, return_fig=True):  # 绘制混淆矩阵
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = str(round(sum_TP / n, 3))  # 总体准确率

        matrix = self.matrix.copy()
        for r in range(self.num_classes):
            row = matrix[r, :]
            num = np.sum(matrix[r, :])
            matrix[r, :] = row / num
            for c in range(self.num_classes):
                matrix[r, c] = round(matrix[r, c], 3)
        # if self.print_func is not None:
        #     self.print_func(matrix)
        fig = plt.figure()
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.class_nume, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.class_nume)
        # 显示colorbar
        plt.colorbar()
        plt.ylabel('True Labels')
        plt.xlabel('Predicted Labels')
        plt.title('Confusion matrix (acc=' + acc + ')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = matrix[y, x]
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        if show:
            plt.show()
        if return_fig:
            return fig
        else:
            plt.close(fig)

    def __str__(self) -> str:
        table = PrettyTable()
        table.field_names = self.class_nume
        for i in range(self.num_classes):
            row = self.matrix[i, :].copy()
            for c in range(self.num_classes):
                row[c] = round(row[c], 3)
            table.add_row(row)
        return str(table)

    def __repr__(self) -> str:
        return self.__str__()

# if __name__ == '__main__':
#     cm = ConfusionMatrix(7)
#     print(cm)
