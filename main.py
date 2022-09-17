import argparse
import torch
import os
import yaml
import wandb
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
from models.base_model import BaseModel
from torch.utils.data import DataLoader
from PIL import Image
from ntpath import basename
from datasets.euvp import EUVPDataset
from utils.loss_utils import VGG19PerceptionLoss
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
from metrics import SSIMs_PSNRs, UIQMs


def train(model, wandb, train_loader):

    # gpu setting for debug
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = nn.DataParallel(model, device_ids=wandb.config.training_device_ids)

    if wandb.config.pre_trained:
        assert os.path.exists(wandb.config.pre_weight)
        pre_weight = torch.load(wandb.config.pre_weight)
        model.load_state_dict(pre_weight)

    # criterion and optimizer
    criterion1 = VGG19PerceptionLoss().to(device)
    criterion2 = torch.nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=wandb.config.lr,
        betas=(wandb.config.lr_b1, wandb.config.lr_b2)
    )
    scheduler = StepLR(optimizer=optimizer, step_size=wandb.config.step_size, gamma=wandb.config.gamma)
    wandb.watch(model, log='all')
    min_loss = 1000
    for epoch in range(wandb.config.num_epoch):
        # train model
        model.train()
        train_loss_list = []
        for i, batch in enumerate(train_loader):
            data = batch['A'].to(device)
            label = batch['B'].to(device)
            optimizer.zero_grad()
            prediction = model(data)
            loss1 = criterion1(prediction, label)
            loss2 = criterion2(prediction, label)
            loss = wandb.config.lambda_1 * loss1 + wandb.config.lambda_2 * loss2
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss = loss.item()
            print(f'========> Epoch: {epoch}\t Index: {i}\t Training Loss: {train_loss}')
            wandb.log({'Train Loss In Each Batch': train_loss})
            train_loss_list.append(train_loss)
        print(f'========> Epoch: {epoch} has finished! Mean training loss: {np.mean(train_loss_list)}')
        wandb.log({'Train Loss Of Each Batch': np.mean(train_loss_list)})

        # save model
        if np.mean(train_loss_list) < min_loss:
            min_loss = np.mean(train_loss_list)
            print(f'========> Minimal training loss: {min_loss} at Epoch: {epoch}')
            wandb.log({'Min Loss': min_loss})
            os.makedirs(os.path.join(wandb.config.save_path, wandb.config.run_name), exist_ok=True)
            torch.save(
                model.module.state_dict(),
                os.path.join(wandb.config.save_path, wandb.config.run_name, 'model.h5')
            )
            print(f'========> Successfully save this checkpoint!!!')


def valid(model, wandb, valid_loader):

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(
        torch.load(os.path.join(wandb.config.save_path, wandb.config.run_name, 'model.h5')),
        strict=False,
    )
    print(f'========> Loaded checkpoint file from {os.path.join(wandb.config.save_path, wandb.config.run_name)}')
    model.eval()
    for i, (batch, img_path) in enumerate(valid_loader):
        data = batch['val'].to(device)
        with torch.no_grad():
            prediction = model(data)
            img_sample = torch.cat((data, prediction), -1)
            save_dir = os.path.join(wandb.config.save_path, wandb.config.run_name, 'valid')
            os.makedirs(save_dir, exist_ok=True)
            save_image(img_sample, os.path.join(save_dir, basename(img_path[0])), normalize=True)
            print(f'========> Processing on {i} of {len(valid_loader)} image ===> {basename(img_path[0])} !!!')


def test(model, wandb, test_loader):

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(
        torch.load(os.path.join(wandb.config.save_path, wandb.config.run_name, 'model.h5')),
        strict=False,
    )

    print(f'========> Loaded checkpoint file from {os.path.join(wandb.config.save_path, wandb.config.run_name)}')
    model.eval()
    for i, (batch, img_path) in enumerate(test_loader):
        data = batch['data'].to(device)
        with torch.no_grad():
            prediction = model(data)
            save_dir = os.path.join(wandb.config.save_path, wandb.config.run_name, 'test')
            os.makedirs(save_dir, exist_ok=True)

            save_image(prediction, os.path.join(save_dir, basename(img_path[0])))
            print(f'========> Processing on {i} of {len(test_loader)} image ===> {basename(img_path[0])} !!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="configs/baseline.yaml")
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--valid", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--metrics", action='store_true')
    args = parser.parse_args()

    # config file
    with open(args.cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    wandb.init(name=cfg['run_name'], project=cfg['project_name'])
    wandb.config.update(cfg)

    transforms_ = [
        transforms.Resize((wandb.config.height, wandb.config.width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    model = BaseModel()

    if args.train:
        train_loader = DataLoader(
            EUVPDataset(
                root=wandb.config.dataset_root,
                split='train',
                paired=wandb.config.paired,
                transforms_=transforms_
            ),
            batch_size=wandb.config.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        print(f'========> The training process begins !!!')
        train(model, wandb, train_loader)
        print(f'========> The training process ends !!!')

    if args.valid:
        valid_loader = DataLoader(
            EUVPDataset(
                root=wandb.config.dataset_root,
                split='val',
                paired=wandb.config.paired,
                transforms_=transforms_
            ),
            batch_size=wandb.config.test_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        print(f'========> The validation process begins !!!')
        valid(model, wandb, valid_loader)
        print(f'========> The validation process ends !!!')

    if args.test:
        test_loader = DataLoader(
            EUVPDataset(
                root=wandb.config.test_dataset_root,
                split='test',
                paired=wandb.config.paired,
                transforms_=transforms_,
            ),
            batch_size=wandb.config.test_batch_size,
            num_workers=8,
            pin_memory=True,
        )

        print(f'========> The test process begins !!!')
        test(model, wandb, test_loader)
        print(f'========> The test process ends !!!')

    if args.metrics:
        print(f'========> Calculating SSIM, PSNR, UQIM !!!')
        ssim_list, psnr_list = SSIMs_PSNRs(
            os.path.join(wandb.config.test_dataset_root, 'GTr'),
            os.path.join(wandb.config.save_path, wandb.config.run_name, 'test')
        )
        print(f'========> Mean of SSIM: {np.mean(ssim_list)}, std of SSIM: {np.std(ssim_list)}')
        print(f'========> Mean of PSNR: {np.mean(psnr_list)}, std of PSNR: {np.std(psnr_list)}')

        data_uqim = UIQMs(os.path.join(wandb.config.test_dataset_root, 'Inp'))
        print(f'========> UIQM of Raw data >> Mean: {np.mean(data_uqim)}, std: {np.std(data_uqim)}')

        label_uqim = UIQMs(os.path.join(wandb.config.test_dataset_root, 'GTr'))
        print(f'========> UIQM of Label >> Mean: {np.mean(label_uqim)}, std: {np.std(label_uqim)}')

        prediction_uqim = UIQMs(os.path.join(wandb.config.save_path, wandb.config.run_name, 'test'))
        print(f'========> UIQM of Predictions >> Mean: {np.mean(prediction_uqim)}, std: {np.std(prediction_uqim)}')












