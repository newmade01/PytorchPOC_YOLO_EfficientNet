import argparse
import os
import pandas as pd
import torch
import albumentations as A
import shutil
from dataset import CustomDataset

from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from torch.utils.data import Dataset
from model import Classifier


parser = argparse.ArgumentParser(description='PyTorch Classification')
parser.add_argument('--model_name', type=str, default="efficientnet_b4", metavar='S',
                    help='model name')
parser.add_argument('--model_path', type=str, default="efficientnet_b4_result_1_4.pt", metavar='S',
                    help='model path')
parser.add_argument('--num_classes', type=int, default=2, metavar='N',
                    help='num classes')
parser.add_argument('--num_workers', type=int, default=6, metavar='N',
                    help='num workers')

parser.add_argument('--dataset_dir', type=str, default="E:\\work\\kesco\\raw_data\\20211008", metavar='S',
                    help='model path')
parser.add_argument('--df_path', type=str, default="segmented_test.csv", metavar='S',
                    help='model path')
parser.add_argument('--save_dir_name', type=str, default="test_result_1", metavar='S',
                    help='model path')




def main():
    args = parser.parse_args()

    # use gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model init
    model = Classifier(args)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    # test data load
    test_df = pd.read_csv(args.df_path).reset_index(drop=True)

    # transform
    transform = A.Compose([
        A.Resize(height=380, width=380),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # dataset
    test_dataset = CustomDataset(args, test_df, transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers)

    # correct variable for calculate accuracy
    num_correct = 0
    num_data = len(test_df)

    correct_imgs = []
    wrong_imgs = []

    # data load
    for batch_idx, (image, label, img_path) in tqdm(enumerate(test_loader)):
        # inference
        input = image.to(device)
        with torch.no_grad():
            pred = model(input)
        pred_label = pred.max(1)[1]# .to("cpu").numpy()

        pred_res = pred_label.eq(label.to(device)).cpu()


        for i, res in enumerate(pred_res):
            if res == 1:
                correct_imgs.append(img_path[i])
            elif res == 0:
                wrong_imgs.append(img_path[i])

        num_correct += pred_res.sum().item()

    print(f'\nAccuracy : {num_correct / num_data:.2f}\n')


    # image save path
    dataset_name = args.dataset_dir.split("\\")[-1]
    save_folder = args.dataset_dir.replace(dataset_name, args.save_dir_name)
    correct_folder = os.path.join(save_folder, "correct")
    wrong_folder = os.path.join(save_folder, "wrong")

    # make directories
    os.makedirs(correct_folder, exist_ok=True)
    os.makedirs(wrong_folder, exist_ok=True)

    print("copy correct images...")
    for correct_img in correct_imgs:
        file_name = correct_img.split("\\")[-1]
        save_img_path = os.path.join(correct_folder, file_name)
        shutil.copy(correct_img, save_img_path)

    print("\nwrong correct images...")
    for wrong_img in wrong_imgs:
        file_name = wrong_img.split("\\")[-1]
        save_img_path = os.path.join(wrong_folder, file_name)
        shutil.copy(wrong_img, save_img_path)






if __name__ == '__main__':
    main()