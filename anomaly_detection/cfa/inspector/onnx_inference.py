from utils.draw_heatmap import HeatMap
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import gaussian_filter
from pathlib import Path
import cv2
import random
import argparse
import warnings
import torch
import onnxruntime as rt
import numpy as np
import time
import json

import os
os.environ["ORT_LOGGING_LEVEL"] = "VERBOSE"

with open('config.json', 'r') as f:
    config_list = json.load(f)

warnings.filterwarnings("ignore", category=UserWarning)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(f'device: {device}')

class LoadDataset(Dataset):
    def __init__(self, patches, resize=256, cropsize=256, info='train', resize_check=True):
        self.patches = patches
        self.resize = resize
        self.cropsize = cropsize
        self.info = info
        self.resize_check = resize_check
        self.x = self.load_dataset()

    def transform_x(self, image):
        a = time.time()
        image = Image.fromarray(np.uint8(image))
        c = time.time()
        if self.resize_check==False:
            image = image.resize((self.resize, self.resize), Image.ANTIALIAS)
        d = time.time()
        image = np.array(image).astype(np.float32)
        image = image / 255.0
        image -= np.array([0.485, 0.456, 0.406])
        image /= np.array([0.229, 0.224, 0.225])
        image = np.transpose(image, (2, 0, 1))
        b = time.time()

        return image

    def transform(self, image):
        # convert 3channel-grayscale
        image = np.repeat(image[..., np.newaxis], repeats=3, axis=2)
        image = Image.fromarray(image)
        image = self.transform_x(image)

        return image

    def __getitem__(self, idx):
        x = self.x[idx]
        x = np.array(x)
        transed_x = self.transform(x)

        return transed_x

    def __len__(self):
        return len(self.x)

    def load_dataset(self):
        x = []
        img_fpath_list = self.patches
        x.extend(img_fpath_list)

        return list(x)

class Fabric_Split:
    def __init__(self, args):
        self.x = 0
        self.y = 0
        if args.visual:
            self.width = 1280
            self.height = 1280
        else:
            self.width = 256*args.patch
            self.height = self.width
        self.patch_size = self.width//args.patch


    def run(self, img_array):
        patches = []

        # Calculate the number of patches in each dimension
        num_patches_x = self.width // self.patch_size
        num_patches_y = self.height // self.patch_size

        for i in range(num_patches_y):
            for j in range(num_patches_x):
                # # Patch area coordinates
                # patch_x = self.x + j * self.patch_size
                # patch_y = self.y + i * self.patch_size
                # patch_width = min(self.patch_size, self.width - j * self.patch_size)
                # patch_height = min(self.patch_size, self.height - i * self.patch_size)
                #
                # # Crop the patch image
                # patch_array = img_array[patch_y:patch_y + patch_height, patch_x:patch_x + patch_width]
                # # patch_img = Image.fromarray(patch_array)
                # patches.append(patch_array)
                patches.append(img_array)
        return patches


class Evaluate:
    def __init__(self, args):
        self.isVisual = args.visual
        self.pre_model, self.model, self.pre_input_name, self.pre_output_names, self.loss_input_name, self.loss_output_names = self.load_model(args)
        self.images_dict, self.images_files = create_image_dict(args.images_path1, args.images_path2)
        # self.images_files = [os.path.join(args.images_path, filename)
        #                      for filename
        #                      in os.listdir(args.images_path)
        #                      if filename.endswith('.bmp')]
        self.split = Fabric_Split(args)

        if self.isVisual:
            self.hm = HeatMap()

        self.min_s = config_list['min_s']
        self.max_s = config_list['max_s']
        self.threshold = config_list['threshold']



    def run(self, args):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        precision = 0
        recall = 0
        f1 = 0
        for i, f in enumerate(self.images_files):
            t0 = time.time()
            heat_maps = []
            heatmaps = None
            print(f"\n\n\n[{i}]file: {f}")
            ## 추후, 파일명이 아닌 image 객체로 바꿀 예정.
            t0_1 = time.time()
            if self.isVisual:
                resize_check = False
            else:
                resize_check = True
            patches = self.preprocessing(f, resize_check)
            t0_2 = time.time()
            test_dataset = LoadDataset(patches=patches,
                                       resize=args.size,
                                       cropsize=args.size,
                                       info='real_test',
                                       resize_check=resize_check)



            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=args.test_batch_size,
                                     pin_memory=True,
                                     drop_last=True)
            t1 = time.time()
            for transed_x in test_loader:
                t2 = time.time()
                transed_x = transed_x.numpy()
                t3 = time.time()
                sample = self.pre_model.run(self.pre_output_names, {self.pre_input_name: transed_x})
                t4 = time.time()
                outputs = self.model.run(self.loss_output_names, {self.loss_input_name: sample[0]})
                t5 = time.time()
                score = outputs[1]
                t6 = time.time()
                score = score[:, -1, ...]
                t7 = time.time()
                if heatmaps is None:
                    heatmaps = score
                else:
                    concat1=time.time()
                    heatmaps = np.concatenate((heatmaps, score), axis=0)
                    concat2 = time.time()
                t8 = time.time()
            t9 = time.time()
            heatmaps = self.upsampling(heatmaps, transed_x.shape[2], 'bilinear')
            t10 = time.time()
            heatmaps = self.gaussian_smooth(heatmaps, sigma=4)
            t11 = time.time()
            scores = self.testset_rescale(heatmaps, self.min_s, self.max_s)
            t12 = time.time()
            for j, p in enumerate(patches):
                max_score = np.max(scores[:, :, j])
                ###############################################
                ####### threshold 보다 높으면 NG라고 판단할 것 ######
                ###############################################
                # {'경로':bad,'경로':good,}
                status = 'good'
                if max_score > self.threshold:
                    print(f'patch {j} bad')
                    status = 'bad'
                    if self.images_dict[f] == 'good':
                        fp += 1
                    else:
                        tp +=1
                else:
                    print(f'patch {j} good')
                    print(self.images_dict[f])
                    if self.images_dict[f] == 'good':
                        tn +=1
                    else:
                        fn += 1
                ###############################################


                sc = scores[:, :, j]
                if self.isVisual:
                    hm_img = self.hm.draw(p, sc, f, j)
                    hm_img = self.marking_ng_ok(hm_img, max_score)
                    heat_maps.append(hm_img)

            if self.isVisual:
                concated_hm = self.concat_heatmap(heat_maps)
                name = Path(f).name
                name = name.replace('.bmp','')
                rename = name + '_hm.png'
                if self.images_dict[f] == 'good':
                    cv2.imwrite(f'{args.images_save_path}/good/{rename}', np.array(concated_hm))
                elif self.images_dict[f] == 'bad':
                    cv2.imwrite(f'{args.images_save_path}/bad/{rename}', np.array(concated_hm))
            t13 = time.time()
            print(f'all: {t13-t0}')
            print(f'preprocessing: {t0_2 - t0_1}')
            print(f'preprocessing + dataloaser define: {t1 - t0}')
            print(f'dataloader_getitem: {t2 - t1}')
            print(f'backbone model: {t4 - t3}')
            print(f'cfa model: {t5 - t4}')
            print(f'score: {t6 - t5}')
            print(f'score mean: {t7 - t6}')
            print(f'heatmaps: {t8 - t7}')
            print(f'image preprocessing + model: {t9 - t1}')
            print(f'upsampling: {t10 - t9}')
            print(f'gaussian: {t11-t10}')
            print(f'testset_rescale: {t12 - t11}')
            print(f'inference: {t13-t12}')
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 / ((1/precision)+(1/recall))
        print(f'thr: {self.threshold}, tp: {tp}, fp:{fp}, tn: {tn}, fn:{fn}, precision: {precision}, recall: {recall}, f1: {f1}')




    def marking_ng_ok(self, image, max_score):
        if max_score > self.threshold:
            return cv2.putText(image, 'NG', (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            return cv2.putText(image, 'OK', (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (243, 206, 125), 2)

    def concat_heatmap(self, heatmaps):
        w_num = len(heatmaps)
        h_num = 1
        # w_size, h_size = heatmaps[0].shape[:-1]
        h_size, w_size = heatmaps[0].shape[:-1]
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',w_size)
        dst = Image.new('RGB', (w_size * w_num, h_size * h_num))
        for i in range(w_num):
            dst.paste(Image.fromarray(heatmaps[i]), (i*w_size, 0))

        return dst

    def preprocessing(self, f, resize):
        # aligned_image = self.alignment.align(f)
        image = Image.open(f).convert('RGB')
        if resize:
            print('resize')
            image = image.resize((256, 256))
        else:
            print('else')
        image = np.array(image)
        print(f'img_shape:{image.shape}')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        patches = self.split.run(image)

        return patches


    def testset_rescale(self, x, min_s, max_s):
        return (x - min_s) / (max_s - min_s)

    def upsampling(self, x, size, mode):
        x = np.transpose(x, (1, 2, 0))
        output = zoom(x, zoom=list(np.array(size) / np.array(x.shape[:-1])) + [1], order=1, mode='nearest')

        return output

    def interpolate(self, x, new_size):
        """
        :param x: input array
        :param new_size: new size tuple (h, w)
        :return: output array of size (h, w)
        """
        h, w = new_size
        y = np.zeros((h, w))
        old_h, old_w = x.shape[-2:]

        # Calculate scale factors
        sh, sw = h / old_h, w / old_w

        # Create output coordinates
        y_coords, x_coords = np.mgrid[:h, :w]

        # Convert output coordinates to input coordinates
        y_coords = (y_coords + 0.5) / sh - 0.5
        x_coords = (x_coords + 0.5) / sw - 0.5

        # Find nearest neighbor pixel coordinates
        top = np.floor(y_coords).astype(int)
        bottom = top + 1
        left = np.floor(x_coords).astype(int)
        right = left + 1

        # Clip coordinates to image boundaries
        top = np.clip(top, 0, old_h - 1)
        bottom = np.clip(bottom, 0, old_h - 1)
        left = np.clip(left, 0, old_w - 1)
        right = np.clip(right, 0, old_w - 1)

        # Interpolate
        dy = y_coords - top
        dx = x_coords - left
        y = x[..., top, left] * (1 - dx) * (1 - dy) \
            + x[..., top, right] * dx * (1 - dy) \
            + x[..., bottom, left] * (1 - dx) * dy \
            + x[..., bottom, right] * dx * dy

        return y

    def avg_pool2d(self, x, kernel_size, stride=None, padding=0):
        if stride is None:
            stride = kernel_size
        _, _, h, w = x.shape
        k = kernel_size
        s = stride
        p = padding

        oh = int(np.floor((h + 2 * p - k) / s)) + 1
        ow = int(np.floor((w + 2 * p - k) / s)) + 1

        x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        out = np.zeros((x.shape[0], x.shape[1], oh, ow))

        for i in range(oh):
            for j in range(ow):
                out[:, :, i, j] = np.mean(x_pad[:, :, i * s:i * s + k, j * s:j * s + k], axis=(2, 3))

        return out

    def gaussian_smooth(self, x, sigma=4):
        bs = x.shape[2]
        print('xxxxxxx:',x.shape)
        for i in range(0, bs):
            x[i] = gaussian_filter(x[i], sigma=sigma)
        return x



    def load_model(self, args):
        if use_cuda:
            # 프로바이더 설정
            providers = ['CUDAExecutionProvider']  # GPU 프로바이더로 설정
            onnx_pre_model = rt.InferenceSession(config_list['pretrained_model_path'], providers=providers)
            onnx_loss_fn = rt.InferenceSession(config_list['cfa_model_path'], providers=providers)

            pre_input_name = onnx_pre_model.get_inputs()[0].name
            pre_output_names = [output.name for output in onnx_pre_model.get_outputs()]

            loss_input_name = onnx_loss_fn.get_inputs()[0].name
            loss_output_names = [output.name for output in onnx_loss_fn.get_outputs()]
        else:
            print('gpu not available')

        return onnx_pre_model, onnx_loss_fn, pre_input_name, pre_output_names, loss_input_name, loss_output_names


def create_image_dict(folder1, folder2):
    folder1_label = 'good'
    folder2_label = 'bad'
    image_dict = {}
    image_list = []

    for file_name in os.listdir(folder1):
        if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".bmp"):  # 이미지 파일 확장자에 맞게 수정하세요
            file_path = os.path.join(folder1, file_name)
            image_dict[file_path] = folder1_label
            image_list.append(file_path)

    for file_name in os.listdir(folder2):
        if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".bmp"):  # 이미지 파일 확장자에 맞게 수정하세요
            file_path = os.path.join(folder2, file_name)
            image_dict[file_path] = folder2_label
            image_list.append(file_path)

    return image_dict, image_list


def parse_args():
    parser = argparse.ArgumentParser('CFA configuration')
    parser.add_argument('--images_save_path', type=str, default='/torch/output')
    parser.add_argument('--size', type=int, choices=[224, 256], default=256)
    parser.add_argument('--visual', type=bool, default=True)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--patch', type=int, default=1)
    parser.add_argument('--label', type=str, default='bad')
    parser.add_argument('--images_path1', type=str, default='/test/good')
    parser.add_argument('--images_path2', type=str, default='/test/bad')

    return parser.parse_args()



def main():
    seed = 1024
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    args = parse_args()
    ev = Evaluate(args)
    ev.run(args)



if __name__ == '__main__':
    main()
