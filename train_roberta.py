#!/usr/bin/env python
# coding: utf-8

# OMP_NUM_THREADS=4 for i in {0..4}; do python -m torch.distributed.launch --nproc_per_node=2 train_roberta.py --kfold_rank=$i;done
import argparse
import os
import pickle
import re
import shutil
import unicodedata as ud
from collections import OrderedDict, defaultdict
from random import shuffle

import apex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from apex import amp
from apex.optimizers import FusedAdam
from apex.parallel import DistributedDataParallel as DDP
from fastprogress import master_bar, progress_bar
from scipy import sparse
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import (DataLoader, Dataset, DistributedSampler, Sampler,
                              SequentialSampler, Subset)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import *

from dataset import BucketDataset, pad_collate
from models import *


def main():

    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Necessary for multi-GPU training')
    parser.add_argument('--kfold_rank', default=-1, type=int,
                        help='Necessary for multi-GPU training')
    args = parser.parse_args()
    if args.local_rank == -1:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.distributed = True

    torch.backends.cudnn.benchmark = True

    np.random.seed(42)
    MAX_LEN = 512 - 1
    BATCH_SIZE = 1
    gradient_accumulation_steps = 16
    bins = [64, 128, 192, 256, 320, 384, 448]
    bs = np.array([BATCH_SIZE] * len(bins) + [BATCH_SIZE])

    LEARNING_RATE = 3e-5
    CYCLE_MOMENTUM = True
    ANNEAL_STRATEGY = 'linear'
    PCT_START = 0.1
    EPOCHS = 6
    EMB_DROPOUT = 0.1
    SAVE_CKPT = False

    model_name = 'init8_roberta_cls_6ep_bs32'
    COATTENTION = False

    fp16 = True
    max_grad_norm = 10.
    device = torch.device('cuda')
    criterion = nn.MSELoss()

    MODEL_CLASS = 'roberta'
    PRETRAIN_MODEL = 'roberta-large'
    PRETRAIN_MODEL = os.path.join('../pretrained_models', PRETRAIN_MODEL)

    sample = pd.read_csv('sample_submission.csv', index_col=0)
    TARGET_COLUMNS = sample.columns

    df = pd.read_csv('train.csv', index_col=0)
    for x in TARGET_COLUMNS:
        df[x] = StandardScaler().fit_transform(
            rankdata(df[x]).reshape(-1, 1)).reshape(-1)

    tokenizer = MODEL_CLASSES[MODEL_CLASS][2].from_pretrained(PRETRAIN_MODEL)

    # category_map = {x: len(tokenizer)+i for i,
    #                 x in enumerate(df.category.unique())}
    # df['length'] = [max(len(x.split(' ')), len(y.split(' ')))
    #             for x,y in zip(df["question_title"].fillna(" ") + ' ' + df["question_body"].fillna(""), df["answer"].fillna(""))]

    if args.local_rank == 0:
        if not os.path.exists('models/'+model_name):
            os.makedirs('models/'+model_name)
        writer = SummaryWriter('runs/{}_{}/'.format(model_name,args.kfold_rank))
        writer_final = SummaryWriter('runs/{}/'.format(model_name))

    criterion = criterion.to(device)

    def train_model(model, optimizer, criterion, dloaders, compute_loss_fn, scheduler=None, num_epochs=25, save_ckpt=False,):
        if os.path.exists('models/{}/model.pkl'.format(model_name)):
            ckpt = torch.load(
                'models/{}/model.pkl'.format(model_name), map_location='cpu')
            model.module.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            amp.load_state_dict(ckpt['amp'])
            scheduler = ckpt['scheduler']
            mb = master_bar(range(ckpt['epoch'], num_epochs))

            del ckpt
            torch.cuda.empty_cache()
            import gc; gc.collect()
        else:
            mb = range(0, num_epochs)
            if args.local_rank == 0:
                mb = master_bar(mb)
        alpha = 0.99
        loss_sm = 0.
        acc_sm = 0.5
        epoch_loss = 0.
        for epoch in mb:
            running_loss = 0.
            running_acc = 0.

            model.train()
            torch.set_grad_enabled(True)

            dloader_iter = iter(dloaders['train'])
            pb = range(len(dloaders['train']) // gradient_accumulation_steps)
            if args.local_rank == 0:
                pb = progress_bar(pb, parent=mb)
            for it in pb:
                for _ in range(gradient_accumulation_steps):
                    questions, answers, labels = (
                        x.to(device) for x in next(dloader_iter))
                    outputs, loss = compute_loss_fn(
                        questions, answers, labels, criterion)
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps
                    if fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    with torch.no_grad():
                        running_loss += loss.item() * gradient_accumulation_steps
                        loss_sm = loss_sm * alpha + loss.item() * gradient_accumulation_steps * (1 - alpha)
                        if args.local_rank == 0:
                            mb.child.comment = 'Loss: ' + \
                                str(loss_sm)[:8]  # + ' ' + str(acc_sm)[:8]

                if max_grad_norm:
                    if fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm)
                optimizer.step()
                for p in model.parameters():
                    p.grad = None
                if scheduler:
                    scheduler.step()  # Update learning rate schedule

            val_outputs = []
            val_labels = []
            val_acc = 0.0
            val_loss = 0.0
            model.eval()
            torch.set_grad_enabled(False)
            for j, inputs in enumerate(dloaders['val']):
                questions, answers, labels = (x.to(device) for x in inputs)
                outputs, loss = compute_loss(
                    questions, answers, labels, criterion)
                val_loss += loss.item()
                val_outputs.append(outputs.detach())
                val_labels.append(labels.detach())
            val_outputs = torch.cat(val_outputs)
            val_labels = torch.cat(val_labels)

            if args.distributed:
                val_loss = torch.tensor(val_loss).cuda()
                running_loss = torch.tensor(running_loss).cuda()
                torch.distributed.all_reduce(
                    val_loss, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(
                    running_loss, op=torch.distributed.ReduceOp.SUM)
                val_loss = val_loss.item()
                running_loss = running_loss.item()

                gather_outputs = [torch.zeros_like(val_outputs)
                                  for _ in range(args.world_size)]
                gather_labels = [torch.zeros_like(val_labels)
                                 for _ in range(args.world_size)]
                torch.distributed.all_gather(gather_outputs, val_outputs)
                torch.distributed.all_gather(gather_labels, val_labels)
                val_outputs = torch.cat(gather_outputs)
                val_labels = torch.cat(gather_labels)

            if args.local_rank == 0:
                val_outputs = val_outputs.cpu().numpy()
                val_labels = val_labels.cpu().numpy()
                score = 0
                for x, y in zip(val_outputs.T, val_labels.T):
                    score += np.nan_to_num(spearmanr(x, y).correlation) / 30
                print(score, val_loss /
                      (len(dloaders['val']) * args.world_size))

                writer.add_scalar('epoch/{}_loss'.format('val'), val_loss / (len(
                    dloaders['val']) * args.world_size), global_step=(epoch + 1) * (len(dloaders['train']) // gradient_accumulation_steps))

                writer.add_scalar('epoch/{}_loss'.format('train'), running_loss / (len(
                    dloaders['train']) * args.world_size), global_step=(epoch + 1) * (len(dloaders['train']) // gradient_accumulation_steps))

                writer.add_scalar('epoch/final_metric', score,
                                  global_step=(epoch + 1) * (len(dloaders['train']) // gradient_accumulation_steps))
                if scheduler:
                    writer.add_scalar(
                        'epoch/lr', scheduler.get_lr()[0], global_step=(epoch + 1) * (len(dloaders['train']) // gradient_accumulation_steps))
                writer.flush()
                if save_ckpt:
                    torch.save(model.module.state_dict(
                    ), 'models/{}/model_{}.pkl'.format(model_name, epoch+1))
        if args.local_rank == 0: return score

    def compute_loss(questions, answers, labels, criterion):
        inputs = torch.cat([questions, answers], 0)
        # segments = torch.cat(
        #     [torch.zeros(questions.size(0), questions.size(1), dtype=torch.long),
        #         torch.ones(answers.size(0), answers.size(1), dtype=torch.long)], 0).to(device)
        attention_mask = inputs != tokenizer.pad_token_id
        outputs = model(inputs, attention_mask=attention_mask, token_type_ids=None
                        )[0]
        loss = criterion(outputs, labels)
        return outputs, loss
    if COATTENTION:
        model = create_roberta_coatt_model(PRETRAIN_MODEL, 20)
    else:
        model = create_roberta_model(PRETRAIN_MODEL)
    model, optimizer = wrap_model(model)
    kfold = GroupKFold(n_splits=5)
    for kfold_it, (train_idx, valid_idx) in enumerate(kfold.split(df, groups=df.question_body)):
        if args.kfold_rank != kfold_it:
            continue
        if args.local_rank == 0:
            print('KFold iteration: {}'.format(kfold_it))
        train_df, valid_df = df.iloc[train_idx], df.iloc[valid_idx]
        train_indices = list(iter(DistributedSampler(
            train_df, args.world_size, args.local_rank)))
        train_dataset = BucketDataset(
            list(zip(train_df.question_title.values[train_indices],
                     train_df.question_body.values[train_indices])),
            train_df.answer.values[train_indices], train_df[TARGET_COLUMNS].iloc[train_indices].values,
            tokenizer, crop=True, max_len=MAX_LEN, emb_dropout=EMB_DROPOUT)

        valid_indices = list(iter(DistributedSampler(
            valid_df, args.world_size, args.local_rank)))
        valid_dataset = BucketDataset(
            list(zip(valid_df.question_title.values[valid_indices],
                     valid_df.question_body.values[valid_indices])),
            valid_df.answer.values[valid_indices], valid_df[TARGET_COLUMNS].iloc[valid_indices].values,
            tokenizer, crop=False, max_len=MAX_LEN, emb_dropout=0.)
        
        def pad_collate_fn(x): return pad_collate(x, tokenizer.pad_token_id, 0 if COATTENTION else 1)
        train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=pad_collate_fn, num_workers=4)
        valid_dl = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=pad_collate_fn, num_workers=4)
        dloaders = {'train': train_dl, 'val': valid_dl}

        scheduler = OneCycleLR(optimizer, LEARNING_RATE, None, EPOCHS, len(dloaders['train']) // gradient_accumulation_steps,
                               cycle_momentum=CYCLE_MOMENTUM, pct_start=PCT_START, anneal_strategy=ANNEAL_STRATEGY)
        score = train_model(model, optimizer, criterion,
                            dloaders, compute_loss, scheduler, EPOCHS, save_ckpt=SAVE_CKPT)

        if args.local_rank == 0:
            writer_final.add_scalar('kfold', score, kfold_it)
            writer.flush()
            torch.save(model.module.state_dict(),
                       'models/{}/model_{}.pkl'.format(model_name, kfold_it))


if __name__ == '__main__':
    main()
