import torch
import numpy as np
import time

from torchvision import transforms

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(
            args.log_dir, 'train' + '/{}'.format(time.strftime('%H-%M-%S'))), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(
            args.log_dir, 'valid' + '/{}'.format(time.strftime('%H-%M-%S'))), flush_secs=1)

    lr = args.learning_rate
    train_dir = args.train
    valid_dir = args.valid
    epochs = args.epochs
    batch_size = args.batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    print(device)

    model = model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    loss = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(2.0)).to(device)

    augmentation = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()]
    )

    valid_transform = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()]
    )

    train_data = load_detection_data(
        train_dir, num_workers=0, batch_size=batch_size, transform=augmentation)
    valid_data = load_detection_data(
        valid_dir, num_workers=0, batch_size=batch_size, transform=augmentation)

    global_step = 0
    for epoch in range(args.epochs):
        print(epoch)
        model.train()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img).view(-1, 1, 128, 128)
            loss_val = loss(logit, label)

            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, label, logit, global_step)

            if train_logger is not None:
                train_logger.add_scalar('train/loss_heat', loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        model.eval()
        running_loss = 0
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            logit = model(img).view(-1, 1, 128, 128)
            running_loss += loss(logit, label).item()
            
        if valid_logger is not None:
            valid_logger.add_scalar('valid/loss', running_loss/len(valid_data), global_step)

        if valid_logger is not None:
            log(valid_logger, img, label, logit, global_step)

        save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-log', '--log_dir', type=str, default='runs')
    # Put custom arguments here
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-t', '--train', type=str, default='data/train')
    parser.add_argument('-v', '--valid', type=str, default='data/valid')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-mo', '--momentum', type=float, default=0.9)
    parser.add_argument('-d', '--decay', type=float, default=0.01)
    parser.add_argument('-b', '--batch', type=int, default=50)
    parser.add_argument('-c', '--continue_training', action='store_true')
    args = parser.parse_args()
    train(args)
