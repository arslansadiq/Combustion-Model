import torch
from sklearn.metrics import r2_score


def my_metric(output, target):
    with torch.no_grad():
        #pred = torch.argmax(output, dim=1)
        #assert pred.shape[0] == len(target)
        #correct = 0
        #correct += torch.sum(output == target).item()
        R2 = r2_score(output, target)
    return R2#correct / len(target)
'''
def my_metric2(output, target, k=3):
    with torch.no_grad():
        #pred = torch.topk(output, k, dim=1)[1]
        #assert pred.shape[0] == len(target)
        correct = 0
        #for i in range(k):
        correct += torch.sum(output == target).item()
    return correct / len(target)
'''