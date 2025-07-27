import os
import json
import tqdm
import torch
from models import cnet
from losses import FocalLoss, RegL1Loss
from dataset import CTDataset
from torchvision import transforms
from torch.utils.data import DataLoader

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

with open("/centernet/public/config.json", "r") as f:
    config_list = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(41)


def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file),
                                os.path.join(path, '..'))
            )


def convert_onnx(model, opt):
    model.eval()
    dummy_input = torch.randn(opt.batch_size, 3, opt.input_size, opt.input_size)
    if torch.cuda.is_available():
        model.cuda()
        dummy_input = dummy_input.cuda()
    else:
        model.cpu()

    dynamic_axes = {"input_0": {0: "batch_size"}}
    dummy_input_names = ["input_0"]
    output_names = ["output_0", "output_1", "output_2"]

    torch.onnx.export(
        model,
        dummy_input,
        config_list["onnx_model_path"],
        opset_version=12,
        do_constant_folding=True,
        input_names=dummy_input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def train_epoch(epoch, model, dl, optimizer, criterion_hm, criterion_wh, criterion_reg):
    model.train()
    loss_meter, it = 0, 0
    bar = tqdm.tqdm(dl)
    bar.set_description_str("%02d" % epoch)
    for item in bar:
        item = [x.to(device) for x in item]
        img, hm, wh, reg, ind, reg_mask = item
        optimizer.zero_grad()
        out_hm, out_wh, out_reg = model(img)
        hm_loss = criterion_hm(out_hm, hm)
        wh_loss = criterion_wh(out_wh, wh, reg_mask, ind)
        reg_loss = criterion_reg(out_reg, reg, reg_mask, ind)
        loss = hm_loss + 0.1 * wh_loss + reg_loss
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        bar.set_postfix(
            hm_loss=hm_loss.item(),
            wh_loss=wh_loss.item(),
            reg_loss=reg_loss.item(),
            loss=loss.item(),
        )
        it += 1
    return loss_meter / it


@torch.no_grad()
def val_epoch(model, dl, criterion_hm, criterion_wh, criterion_reg):
    model.eval()
    loss_meter, it = 0, 0
    for item in dl:
        item = [x.to(device) for x in item]
        img, hm, wh, reg, ind, reg_mask = item
        out_hm, out_wh, out_reg = model(img)
        hm_loss = criterion_hm(out_hm, hm)
        wh_loss = criterion_wh(out_wh, wh, reg_mask, ind)
        reg_loss = criterion_reg(out_reg, reg, reg_mask, ind)
        loss = hm_loss + 0.1 * wh_loss + reg_loss
        loss_meter += loss.item()
        it += 1
    return loss_meter / it


def train(opt):
    model = cnet(nb_res=opt.resnet_num, num_classes=opt.num_classes)
    #    model.load_state_dict(torch.load(opt.ckpt, map_location='cpu'))
    model = model.to(device)

    transform_train = transforms.Compose(
        [
            transforms.Resize((opt.input_size, opt.input_size)),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dic_data = torch.load("/centernet/public/data.pth")

    train_dataset = CTDataset(
        opt=opt, data=dic_data["train"], transform=transform_train
    )
    val_dataset = CTDataset(opt=opt, data=dic_data["val"], transform=transform_train)
    train_dl = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True,
    )
    val_dl = DataLoader(
        dataset=val_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers
    )

    criterion_hm = FocalLoss()
    criterion_wh = RegL1Loss()
    criterion_reg = RegL1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    min_loss, best_epoch = 1e7, 1
    for epoch in range(1, opt.max_epoch + 1):
        train_loss = train_epoch(
            epoch, model, train_dl, optimizer, criterion_hm, criterion_wh, criterion_reg
        )
        val_loss = val_epoch(model, val_dl, criterion_hm, criterion_wh, criterion_reg)
        print(
            "Epoch%02d train_loss:%0.3e val_loss:%0.3e min_loss:%0.3e(%02d)"
            % (epoch, train_loss, val_loss, min_loss, best_epoch)
        )
        if min_loss > val_loss:
            min_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), config_list["torch_model_path"])
            convert_onnx(model, opt)

            #with zipfile.ZipFile(config_list["model_archive_path"], 'w', zipfile.ZIP_DEFLATED) as zipf:
            #    zipdir("/centernet/public/dl_models", zipf)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", 4)))
    parser.add_argument("--num_workers", type=int, default=int(os.getenv("NUM_WORKERS", 0)))
    parser.add_argument("--max_epoch", type=int, default=int(os.getenv("MAX_EPOCH", 500)))
    parser.add_argument("--lr", type=float, default=float(os.getenv("LR", 1e-4)))
    parser.add_argument("--resnet_num", type=int, default=int(os.getenv("RESNET_NUM", 18)),
                        choices=[18, 34, 50, 101, 152])
    parser.add_argument("--num_classes", type=int, default=int(os.getenv("NUM_CLASSES", 10)))
    parser.add_argument("--input_size", type=int, default=int(os.getenv("INPUT_SIZE", 512)))
    parser.add_argument("--max_objs", type=int, default=int(os.getenv("MAX_OBJS", 100)))
    parser.add_argument("--topk", type=int, default=int(os.getenv("TOPK", 100)))
    parser.add_argument("--threshold", type=float, default=float(os.getenv("THRESHOLD", 0.5)))
    parser.add_argument("--down_ratio", type=int, default=int(os.getenv("DOWN_RATIO", 4)))
    opt = parser.parse_args()

    train(opt)
