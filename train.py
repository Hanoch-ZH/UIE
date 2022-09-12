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
from datasets.euvp import EUVPDataset
from utils.loss_utils import VGG19PerceptionLoss
from torch.optim.lr_scheduler import StepLR
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="configs/baseline.yaml")
    # parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    # config file
    with open(args.cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    wandb.init(name=cfg['run_name'], project=cfg['project_name'])
    wandb.config.update(cfg)

    # dataloader
    transforms_ = [
        transforms.Resize((wandb.config.height, wandb.config.width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
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
    # valid_loader = DataLoader(
    #     EUVPDataset(
    #         root=wandb.config.dataset_root,
    #         split='valid',
    #         paired=wandb.config.paired,
    #         transforms_=transforms_
    #     ),
    #     batch_size=wandb.config.batch_size,
    #     num_workers=8,
    #     pin_memory=True,
    # )

    # model setting
    model = BaseModel()

    # gpu setting for debug
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # gpu setting for running
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.to(device)
        # torch.distributed.init_process_group(backend="nccl")
        # model.to(device)
        # model = nn.parallel.DistributedDataParallel(model)

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
            wandb.log({'Train Loss In Each Batch': train_loss})
            train_loss_list.append(train_loss)
        wandb.log({'Train Loss Of Each Batch': np.mean(train_loss_list)})

        # save model
        if np.mean(train_loss_list) < min_loss:
            min_loss = np.mean(train_loss_list)
            wandb.log({'Min Loss': min_loss})
            os.makedirs(os.path.join(wandb.config.save_path, wandb.config.run_name), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(wandb.config.save_path, wandb.config.run_name, 'model.h5'))










