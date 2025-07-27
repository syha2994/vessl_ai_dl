import os
from PIL import Image, ImageDraw, ImageFont
import utils
import numpy as np
import onnxruntime as rt
import json

with open('/centernet/public/config.json', 'r') as f:
    config_list = json.load(f)


def transform_x(image, input_size):
    # PIL 이미지인 경우 NumPy 배열로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)  # (H, W) 또는 (H, W, C)

    # Grayscale 이미지인 경우 RGB로 변환
    if len(image.shape) == 2:  # (H, W) -> (H, W, 3)
        image = np.stack([image] * 3, axis=-1)

    image = Image.fromarray(np.uint8(image))  # PIL 변환
    image = image.resize((input_size, input_size), Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32)
    image = image / 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    image = np.transpose(image, (2, 0, 1))
    return image


def load_model():
    od_model = rt.InferenceSession(config_list['onnx_model_path'])
    od_input_name = od_model.get_inputs()[0].name
    od_output_names = [output.name for output in od_model.get_outputs()]
    return od_model, od_input_name, od_output_names


def inference(options, image):
    model, input_name, output_names = load_model()
    real_w, real_h = image.size
    img = transform_x(image, options.od_input_size)
    img = np.expand_dims(img, 0)

    #out_hm, out_wh, out_reg = model.run(input_name, {output_names: img})
    out_hm, out_wh, out_reg = model.run(output_names, {input_name: img})
    bbox, cls, scores = utils.heatmap_bbox(out_hm, out_wh, out_reg, options.topk)

    w_ratio = real_w * options.down_ratio / options.od_input_size
    h_ratio = real_h * options.down_ratio / options.od_input_size

    cls = np.expand_dims(cls, axis=-1).astype(float)
    scores = np.expand_dims(scores, axis=-1)

    bbox_cls_score = np.concatenate((bbox, cls, scores), axis=-1)
    bbox_cls_score = np.squeeze(bbox_cls_score)

    draw = ImageDraw.Draw(image)

    for bcs in bbox_cls_score:
        box, cls, score = bcs[:4], int(bcs[4]), bcs[-1]
        if score > 0.5:
            box = box[0] * w_ratio, box[1] * h_ratio, box[2] * w_ratio, box[3] * h_ratio
            if options.visual:
                font_size = 16
                #font = ImageFont.truetype('/Users/syha/Desktop/마루 부리/MaruBuriOTF/MaruBuri-Regular.otf', font_size)
                font = ImageFont.load_default() 
                draw.rectangle(box, outline=f'red')
                draw.text((box[0], box[1] - 10), "(%d,%0.3f)" % (cls, score), fill=f'yellow', font=font)

    image.save(os.path.join(options.output_dir, img_name))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--od_input_size', type=int, default=512, help='image input size')
    parser.add_argument('--topk', type=int, default=40, help='topk in target')
    parser.add_argument('--down_ratio', type=int, default=4, help='downsample ratio')
    parser.add_argument('--test_dir', type=str, default='/centernet_dataset/Images/val')
    parser.add_argument('--output_dir', type=str, default='/centernet/inspector/output',
                        help='output directory')
    parser.add_argument('--visual', type=bool, default=True)

    opt = parser.parse_args()

    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)

    files = os.listdir(opt.test_dir)

    for test_img_path in files:
        if test_img_path.endswith(".png") or test_img_path.endswith(".bmp") or test_img_path.endswith(".jpg"):
            img_name = os.path.basename(test_img_path)
            test_img = Image.open("{}/{}".format(opt.test_dir, test_img_path))
            inference(opt, test_img)
