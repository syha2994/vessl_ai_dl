import os
import random
import argparse

import torch.onnx
import torch.optim as optim
from torch.utils.data import DataLoader

from cnn.vgg import vgg19_bn as vgg19
from cnn.resnet import wide_resnet50_2 as wrn50_2
from cnn.resnet import resnet18 as res18
from cnn.efficientnet import EfficientNet as effnet

import make_datasets.load_datasets as load_dataset
from make_datasets.load_datasets import LoadDataset

from utils.metric import *
from utils.visualizer import *

from utils.cfa import *
import warnings
import json

import vessl

config_path = '/torch/public/result/config.json'
with open(config_path, 'r') as f:
    config_list = json.load(f)


warnings.filterwarnings("ignore", category=UserWarning)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser('CFA configuration')
    parser.add_argument('--data_path', type=str, default=os.environ.get('data_path', '/cfa_dataset/'))
    parser.add_argument('--save_path', type=str, default=os.environ.get('save_path', '/torch/public/result'))
    parser.add_argument('--save_model_type', type=str, choices=['dict', 'total'], default=os.environ.get('save_model_type', 'total'))
    parser.add_argument('--load_model_type', type=str, choices=['dict', 'total'], default=os.environ.get('load_model_type', 'total'))
    parser.add_argument('--Rd', type=bool, default=os.environ.get('rd', 'False') == 'True')
    parser.add_argument('--cnn', type=str, choices=['res18', 'wrn50_2', 'effnet-b5', 'vgg19'], default=os.environ.get('cnn', 'wrn50_2'))
    parser.add_argument('--size', type=int, choices=[224, 256], default=int(os.environ.get('size', 256)))
    parser.add_argument('--gamma_c', type=int, default=int(os.environ.get('gamma_c', 1)))
    parser.add_argument('--gamma_d', type=int, default=int(os.environ.get('gamma_d', 1)))
    parser.add_argument('--train_batch_size', type=int, default=int(os.environ.get('train_batch_size', 4)))
    parser.add_argument('--test_batch_size', type=int, default=int(os.environ.get('test_batch_size', 1)))
    parser.add_argument('--lr', type=float, default=float(os.environ.get('lr', 1e-3)))
    parser.add_argument('--epochs', type=int, default=int(os.environ.get('epochs', 30)))
    parser.add_argument('--pretrained_path', type=str, default=os.environ.get('pretrained_path', 'None'))
    parser.add_argument('--class_name', type=str, default=os.environ.get('class_name', 'MVtec'))

    return parser.parse_args()


# Function to Convert to ONNX
def Convert_ONNX(model):

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, 1792, 64, 64, dtype=torch.float32)
    dynamic_axes = {'input_0': {0: 'batch_size'},
                    'output_1': {0: 'batch_size'}}
    dummy_input_names = ['input_0']
    output_names = ['output_0', 'output_1']

    torch.onnx.export(model, dummy_input, config_list['cfa_model_path'],
                      opset_version=12, do_constant_folding=True,
                      input_names=dummy_input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes)
    print(" ")
    print('Model has been converted to ONNX')


def save_checkpoint(model, args):
    print("Saved Model !")
    if args.save_model_type\
            == 'dict':
        state = {'model': model.state_dict()}
        torch.save(state, os.path.join(args.save_path, 'model.pth'))
    elif args.save_model_type == 'total':
        torch.save(model, os.path.join(args.save_path, 'model.pth'))
    Convert_ONNX(model)


def run():

    args = parse_args()

    seed = 1024
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    class_names = load_dataset.CLASS_NAMES if args.class_name == 'all' else [args.class_name]

    total_roc_auc = []

    # plt.subplots(행 개수, 열 개수, 생성할 그림의 크기)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    for class_name in class_names:
        best_acc = -1
        best_img_roc = -1
        best_pxl_roc = -1
        best_pxl_pro = -1
        print(' ')
        print('%s | newly initialized...' % class_name)

        train_dataset    = LoadDataset(dataset_path  = args.data_path,
                                        class_name    =     class_name,
                                        resize        =      args.size,
                                        cropsize      =      args.size,
                                        info          =         'train')

        test_dataset     = LoadDataset(dataset_path  = args.data_path,
                                        class_name    =     class_name,
                                        resize        =      args.size,
                                        cropsize      =      args.size,
                                        info          =         'test')


        train_loader   = DataLoader(dataset         = train_dataset,
                                    batch_size      = args.train_batch_size,
                                    pin_memory      =          True,
                                    shuffle         =          True,
                                    drop_last       =          True,)

        test_loader   =  DataLoader(dataset        =   test_dataset,
                                    batch_size     = args.test_batch_size,
                                    pin_memory     =           True,
                                    drop_last       =          True,)

        if args.cnn == 'wrn50_2':
            model = wrn50_2(pretrained=True, progress=True)
        elif args.cnn == 'res18':
            model = res18(pretrained=True,  progress=True)
        elif args.cnn == 'effnet-b5':
            model = effnet.from_pretrained('efficientnet-b5')
        elif args.cnn == 'vgg19':
            model = vgg19(pretrained=True, progress=True)


        model = model.to(device)
        model.eval()

        output_onnx = config_list['pretrained_model_path']
        input_names = ["input_0"]
        output_names = ["output_0"]
        inputs = torch.randn(1, 3, 256, 256).to(device)
        dynamic_axes = {'input_0': {0: 'batch_size'}}
        torch.onnx.export(
            model, inputs, output_onnx, export_params=True, verbose=False,
            input_names=input_names, output_names=output_names,
            opset_version=11, dynamic_axes=dynamic_axes
        )

        if args.pretrained_path == 'None':
            loss_fn = DSVDD(model, train_loader, args.cnn, args.gamma_c, args.gamma_d, device)
            loss_fn = loss_fn.to(device)
        else:
            if args.load_model_type == 'dict':
                pass
                print("[state_dict] Loaded model !")
            elif args.load_model_type == 'total':
                loss_fn = torch.load(os.path.join(args.pretrained_path), map_location=torch.device('cpu'))
                loss_fn = loss_fn.to(device)
                print("[whole] Loaded model !")

        params = [{'params': loss_fn.parameters()},]
        optimizer     = optim.AdamW(params        = params,
                                    lr            = args.lr,
                                    weight_decay  = 5e-4,
                                    amsgrad       = True )

        for epoch in tqdm(range(args.epochs), '%s -->'%(class_name)):
            r'TEST PHASE'

            test_imgs = list()
            gt_mask_list = list()
            gt_list = list()
            heatmaps = None

            loss_fn.train()
            for (x, _, _) in train_loader:
                optimizer.zero_grad()
                sample = model(x.to(device))
                train_loss, train_score = loss_fn(sample)
                train_loss.backward()
                optimizer.step()

            loss_fn.eval()

            for x, y, mask in test_loader:
                test_imgs.extend(x.cpu().detach().numpy())
                gt_list.extend(y.cpu().detach().numpy())
                gt_mask_list.extend(mask.cpu().detach().numpy())

                sample = model(x.to(device))

                test_loss, score = loss_fn(sample)

                heatmap = score.cpu().detach()
                heatmap = torch.mean(heatmap, dim=1)
                heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps != None else heatmap

            heatmaps = upsample(heatmaps, size=x.size(2), mode='bilinear')
            heatmaps = gaussian_smooth(heatmaps, sigma=4)
            
            gt_mask = np.asarray(gt_mask_list, dtype=object)
            scores = rescale(heatmaps)

            scores = scores
            # threshold = get_threshold(gt_mask, scores)

            r'Image-level AUROC'
            # fpr, tpr, img_roc_auc = cal_img_roc(scores, gt_list)
            fpr, tpr, thresholds, img_roc_auc = cal_img_roc(scores, gt_list)
            # best_img_roc = img_roc_auc if img_roc_auc > best_img_roc else best_img_roc
            if img_roc_auc > best_img_roc:
                best_img_roc = img_roc_auc
                best_heatmaps = heatmaps
                optimal_idx = np.argmax(tpr - fpr)
                best_optimal_threshold = thresholds[optimal_idx]
                save_checkpoint(loss_fn, args)

                tn, fp, fn, tp, acc, f1, fn_idx, fp_idx = cal_acc(scores, gt_list, best_optimal_threshold)
                if acc > best_acc:
                    print('best_acc!')
                    best_acc = acc
            elif img_roc_auc == best_img_roc:
                tn, fp, fn, tp, acc, f1, fn_idx, fp_idx = cal_acc(scores, gt_list, best_optimal_threshold)
                if acc > best_acc:
                    print('best_acc!')
                    best_acc = acc
                    best_img_roc = img_roc_auc
                    best_heatmaps = heatmaps
                    optimal_idx = np.argmax(tpr - fpr)
                    best_optimal_threshold = thresholds[optimal_idx]
                    save_checkpoint(loss_fn, args)

            vessl.log({"best_img_roc": best_img_roc, "loss": best_acc})

            print(f'tn: {tn}\n'
                  f'fp: {fp}\n'
                  f'fn: {fn}\n'
                  f'tp: {tp}\n'
                  f'f1: {f1}\n'
                  f'acc: {acc}\n'
                  f'fn_idx: {fn_idx}\n'
                  f'fp_idx: {fp_idx}\n')

            fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))

            print('[%d / %d]image ROCAUC: %.3f | best: %.3f'% (epoch, args.epochs, img_roc_auc, best_img_roc))
            print('image ROCAUC: %.3f'% (best_img_roc))
            print(f'min: {best_heatmaps.min()}')
            print(f'max: {best_heatmaps.max()}')
            print(f'threshold: {best_optimal_threshold}')
            config_list['min_s'], config_list['max_s'], config_list[
                'threshold'] = float(best_heatmaps.min()), float(best_heatmaps.max()), float(best_optimal_threshold)
            with open(config_path, 'w') as f:
                json.dump(config_list, f)

        total_roc_auc.append(best_img_roc)

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


if __name__ == '__main__':
    run()
