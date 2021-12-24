import torch.optim as optim
import torch.nn as nn
import torch
import time

import argparse
from models import Net
# import matplotlib.pyplot as plt
import numpy as np
from data import get_train_val_datasets


def train(model, train_loader, test_loader, scheduler, optimizer, loss_func, epochs=20):
    model.train()
    loss_list = []
    for epoch in range(epochs):
        total_loss = []
        torch.autograd.set_detect_anomaly(True)
        for _, (img, target) in enumerate(train_loader):
            target = target.double()
            # target = target.float()
            optimizer.zero_grad()
            # Forward pass
            output = model(img)
            # Calculating loss
            output = torch.sigmoid(output)
            loss = loss_func(output, target)
            # Backward pass
            loss.backward()
            # Optimize the weights
            optimizer.step()

            total_loss.append(loss.item())
        loss_list.append(sum(total_loss)/len(total_loss))
        scheduler.step(loss_list[-1])
        val(model, test_loader)
        print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
            100. * (epoch + 1) / epochs, loss_list[-1]))

    return loss_list

def val(model, val_loader):
    model.eval()
    accuracy = []
    with torch.no_grad():
        for _, (data, target) in enumerate(val_loader):
            output = model(data)
            pred = torch.sigmoid(output)
            pred = pred > 0.5
            accuracy.append(target == pred)
    print(np.mean(accuracy))

def visualize(loss_list):
    plt.plot(loss_list)
    plt.title('Hybrid NN Training Convergence')
    plt.xlabel('Training Iterations')
    plt.ylabel('Neg Log Likelihood Loss')
    plt.savefig('loss_list.jpg')

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default='',
                        help='path to config file')
    args = parser.parse_args()

    start = time.time()
    model = Net()
    train_loader, val_loader = get_train_val_datasets(args.root)
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    loss_func = nn.BCELoss()

    epochs = 50
    losses = train(model, train_loader, val_loader, scheduler, optimizer, loss_func, epochs)
    val(model, val_loader)
    end = time.time()
    print(losses)
    print("time: ", start - end)

if __name__ == "__main__":
    main()