from __future__ import print_function
import settings
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import ResNet as models
from torch.utils import model_zoo
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import utils

training_statistic = []
testing_statistic = []
fig = plt.figure()
subplot = fig.add_subplot(1,1,1)
x_train, y_train, x_test, y_test = [], [], [], []

def load_pretrain(model):
    url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    pretrained_dict = model_zoo.load_url(url)
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if not "cls_fc" in k:
            model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
    model.load_state_dict(model_dict)
    return model

def train(epoch, model):
    result = []
    LEARNING_RATE = settings.lr / math.pow((1 + 10 * (epoch - 1) / settings.epochs), 0.75)
    print('learning rate{: .4f}'.format(LEARNING_RATE) )
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=settings.momentum, weight_decay=settings.l2_decay)

    model.train()

    iter_source = iter(data_loader.source_loader)
    iter_target = iter(data_loader.target_train_loader)
    num_iter = data_loader.len_source_loader
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        if i % data_loader.len_target_loader == 0:
            iter_target = iter(data_loader.target_train_loader)
        data_source, label_source = Variable(data_source), Variable(label_source)
        data_target = Variable(data_target)

        optimizer.zero_grad()
        label_source_pred, loss_coral = model(data_source, data_target)
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)
        gamma = 2 / (1 + math.exp(-10 * (epoch) / settings.epochs)) - 1
        loss = loss_cls + gamma * loss_coral
        loss.backward()
        optimizer.step()
        if i % settings.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttotal_Loss: {:.6f}\tcls_Loss: {:.6f}\tcoral_Loss: {:.6f}'.format(
                epoch, i * len(data_source), data_loader.len_source_dataset,
                100. * i / data_loader.len_source_loader, loss.data[0], loss_cls.data[0], loss_coral.data[0]))

            result.append({
                'epoch': epoch,
                'step': i + 1,
                'total_steps': num_iter,
                'loss': loss.data[0],  # classification_loss.data[0]
            })

        return result

def test(model, dataset_loader, epoch, mode = "training"):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in data_loader.target_test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        s_output, t_output = model(data, data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, size_average=False).data[0] # sum up batch loss
        pred = s_output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= data_loader.len_target_dataset
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        settings.target_name, test_loss, correct, data_loader.len_target_dataset,
        100. * correct / data_loader.len_target_dataset))

    testing_statistic.append({
        'data': mode,
        'epoch': epoch,
        'average_loss': test_loss,
        'correct': correct,
        'total': len(dataset_loader.dataset),
        'accuracy': 100. * len(dataset_loader.dataset)
    })

    if mode == "training":
        x_train.append(epoch)
        y_train.append(test_loss)
    elif mode == "testing":
        x_test.append(epoch)
        y_test.append(test_loss)

    return correct


if __name__ == '__main__':
    model = models.DeepCoral(num_classes=2)
    correct = 0
    print(model)
    #model = load_pretrain(model)
    for epoch in range(1, settings.epochs + 1):
        train(epoch, model)

        test(model, data_loader.source_loader, epoch=epoch, mode="training")
        t_correct = test(model, data_loader.target_test_loader, epoch=epoch, mode="testing")

        if t_correct > correct:
            correct = t_correct
        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
              settings.source_name, settings.target_name, correct, 100. * correct / data_loader.len_target_dataset ))

    subplot.plot(x_train, y_train, 'g')
    subplot.plot(x_test, y_test, 'r')
    plt.show()

    utils.save(training_statistic, 'training_statistic.pkl')
    utils.save(testing_statistic, 'testing_statistic.pkl')
    utils.save_net(model, 'checkpoint.tar')