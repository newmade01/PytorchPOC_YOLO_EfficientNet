import torch
import random
import os
import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

from transforms import get_transforms

from model import Classifier
from train import train_fn, test_fn
from dataset import CustomDataset


# eff - Training settings
parser = argparse.ArgumentParser(description='PyTorch Classification')
parser.add_argument('--train_path', type=str, default="train_covid_croped.csv", metavar='P',
                    help='train label path')
parser.add_argument('--test_path', type=str, default="test_covid_croped.csv", metavar='P',
                    help='test label path')
parser.add_argument('--dataset_dir', type=str, default="D:\\LYL\\final_colorectal_gastric_train\\dataset\\classification", metavar='P',
                    help='dataset_dir')
parser.add_argument('--output_dir', type=str, default="output_b4_3", metavar='P',
                    help='output dir')
parser.add_argument('--model_name', type=str, default="efficientnet_b4", metavar='S',
                    help='model name in timm package (default: efficientnet_b4)')
parser.add_argument('--pretrained', action='store_true', default=True,
                    help='load pretrained model')
parser.add_argument('--pretrained_path', type=str, default="", metavar='S',
                    help='pretrained model path (default: "")')
parser.add_argument('--train_batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='N',
                    help='input batch size for validation (default: 32)')
parser.add_argument('--target_col', type=str, default="label", metavar='S',
                    help='target column name (default: label)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--train_img_size', type=int, default=320, metavar='N',   # 192, 256, 320, 380
                    help='train image size (default: 320)')
parser.add_argument('--test_img_size', type=int, default=380, metavar='N',   # 380
                    help='test image size (default: 380)')

parser.add_argument('--train_img_sizes', type=int, nargs='+', metavar='N',
                    help='train image sizes')

parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                    help='how many training processes to use (default: 0)')
parser.add_argument('--val_per_epochs', type=int, default=10, metavar='N',
                    help='validation per epoch (default: 10)')
parser.add_argument('--num_classes', type=int, default=4, metavar='N',
                    help='number of classes')
parser.add_argument('--n_fold', type=int, default=2, metavar='N',
                    help='number of folds (default: 4)')

parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='RMSprop momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=1e-5, metavar='D',
                    help='RMSprop weight decay (default: 0.00001)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run(args, model, device, train_dataset, valid_dataset, LOGGER, writer, test_dataset=None, fold=0):

    criterion = torch.nn.CrossEntropyLoss()
    train_fn(args=args, model=model, device=device, train_dataset=train_dataset, valid_dataset=valid_dataset,
             criterion=criterion, fold=fold, writer=writer, LOGGER=LOGGER)

    # load the best validation loss model
    if fold != 0:
        model.load_state_dict(
            torch.load(os.path.join(args.output_dir, f'{args.model_name}_fold{fold}_best_loss.pt')))

    if test_dataset is not None:
        test_loss, test_accuracy = test_fn(args, model, device, test_dataset, criterion, LOGGER)
        if not os.path.exists(os.path.join(args.output_dir, "test_good")):
            os.makedirs(os.path.join(args.output_dir, "test_good"))

        torch.save(model.state_dict(),
                   os.path.join(args.output_dir,"test_good", f'{args.model_name}_{fold}_acc_{test_accuracy:.2f}_loss_{test_loss:.2f}.pt'))

        writer.add_scalars('Loss', {'test': test_loss}, (fold+1)*args.epochs)
        writer.add_scalars('Accuracy', {'test': test_accuracy}, (fold+1)*args.epochs)



def main():
    args = parser.parse_args()

    # make output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # train setting
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    seed_torch(seed=args.seed)

    # lr setting
    # args.lr = args.train_batch_size * torch.cuda.device_count() * 0.256 / 4096

    # logger
    def init_logger(log_file=args.output_dir + '/train.log'):
        from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
        logger = getLogger(__name__)
        logger.setLevel(INFO)
        handler1 = StreamHandler()
        handler1.setFormatter(Formatter("%(message)s"))
        handler2 = FileHandler(filename=log_file)
        handler2.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        return logger
    LOGGER = init_logger()


    # Tensorboard writer
    tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tensorboard_dir)


    # data read
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)


    # split folders
    Fold = StratifiedKFold(n_splits=args.n_fold, shuffle=True, random_state=args.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(train_df, train_df[args.target_col])):
        train_df.loc[val_index, 'fold'] = int(n)
    train_df['fold'] = train_df['fold'].astype(int)


    # model create
    model = Classifier(args, pretrained=True)

    try:
        model.load_state_dict(torch.load(args.pretrained_path))
        # model.module.load_state_dict(torch.load(args.pretrained_path))
    except:
        print("cannot load pretrained model")

    model.to(device)

    # train image sizes
    if args.train_img_sizes is not None:
        train_img_sizes = args.train_img_sizes
    else:
        train_img_sizes = [args.train_img_size]
    n_sizes = len(train_img_sizes)
    size_idx = 0

    # training
    for fold in range(args.n_fold*n_sizes):
        # print(f'========fold_{fold}=========')
        LOGGER.info(f"\n========== fold: {fold} training ==========")

        # train image size setting
        if fold != 0 and fold % args.n_fold == 0:
            size_idx += 1
        try:
            args.train_img_size = train_img_sizes[size_idx]
        except:
            pass


        trn_idx = train_df[train_df['fold'] != fold%args.n_fold].index
        val_idx = train_df[train_df['fold'] == fold%args.n_fold].index

        train_folds = train_df.loc[trn_idx].reset_index(drop=True)
        valid_folds = train_df.loc[val_idx].reset_index(drop=True)

        train_dataset = CustomDataset(args=args, df=train_folds,
                                    transforms=get_transforms(img_size=args.train_img_size, data='train'))
        valid_dataset = CustomDataset(args=args, df=valid_folds,
                                    transforms=get_transforms(img_size=args.test_img_size, data='test'))
        test_dataset = CustomDataset(args=args, df=test_df,
                                    transforms=get_transforms(img_size=args.test_img_size, data='test'))

        run(args, model, device, train_dataset, valid_dataset, LOGGER, writer, test_dataset, fold)



if __name__ == '__main__':
    main()