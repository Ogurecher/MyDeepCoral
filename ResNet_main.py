import ResNet as models
import settings
import data_loader

import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def train(epoch, model):
    LEARNING_RATE = settings.lr / math.pow((1 + 10 * (epoch - 1) / settings.epochs), 0.75)
    print('learning rate{: .4f}'.format(LEARNING_RATE) )
    optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=LEARNING_RATE, momentum=settings.momentum, weight_decay=settings.l2_decay)

    model.train()

    iter_source = iter(data_loader.source_loader)
    num_iter = data_loader.len_source_loader
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_source, label_source = Variable(data_source), Variable(label_source)

        optimizer.zero_grad()
        label_source_pred = model(data_source)
        loss = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)
        loss.backward()
        optimizer.step()
        if i % settings.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data_source), data_loader.len_source_dataset,
                100. * i / data_loader.len_source_loader, loss.data[0]))


def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in data_loader.target_test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        s_output = model(data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, size_average=False).data[0] # sum up batch loss
        pred = s_output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= data_loader.len_target_dataset
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, data_loader.len_target_dataset,
        100. * correct / data_loader.len_target_dataset))
    return correct


if __name__ == '__main__':
    models.resNet_main = True
    model = models.ResNet(models.Bottleneck, [3, 4, 6, 3], num_classes=2)
    correct = 0
    print(model)

    for epoch in range(1, settings.epochs + 1):
        train(epoch, model)
        t_correct = test(model)
        if t_correct > correct:
            correct = t_correct
        print('source: {} max correct: {} max accuracy{: .2f}%\n'.format(
              settings.source_name, correct, 100. * correct / data_loader.len_target_dataset ))