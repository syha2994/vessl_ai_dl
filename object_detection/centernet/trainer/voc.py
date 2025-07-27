import os, torch
import xml.etree.ElementTree as ET

voc_root = "/centernet_dataset"
voc_Annotations = os.path.join(voc_root, "Annotations")
voc_JPEGImages = os.path.join(voc_root, "Images")


def annot_box_loc(ann_path):
    tree = ET.ElementTree(file=ann_path)
    root = tree.getroot()
    object_set = root.findall("object")
    obj_bnd_box_set = {}
    for Object in object_set:
        obj_name = Object.find("name").text
        bnd_box = Object.find("bndbox")
        x1 = int(round(float(bnd_box.find("xmin").text)))
        y1 = int(round(float(bnd_box.find("ymin").text)))
        x2 = int(round(float(bnd_box.find("xmax").text)))
        y2 = int(round(float(bnd_box.find("ymax").text)))
        bnd_box_loc = [(x1), y1, x2, y2]
        if obj_name in obj_bnd_box_set:
            obj_bnd_box_set[obj_name].append(bnd_box_loc)
        else:
            obj_bnd_box_set[obj_name] = [bnd_box_loc]
    return obj_bnd_box_set


def get_classes_name():
    classes_name = set()
    for name in os.listdir(voc_Annotations):
        if ".xml" in name:
            ann_path = os.path.join(voc_Annotations, name)
            tree = ET.ElementTree(file=ann_path)
            root = tree.getroot()
            object_set = root.findall("object")
            for Object in object_set:
                obj_name = Object.find("name").text
                classes_name.add(obj_name)
    return classes_name


def get_data(set_list, idx2names, id):
    data = []
    for img_name in set_list:
        if id == "train":
            img_path = os.path.join(voc_JPEGImages + "/train/{}".format(img_name))
        if id == "val":
            img_path = os.path.join(voc_JPEGImages + "/val/{}".format(img_name))
        img_name = img_name.replace(".png", "")
        img_name = img_name.replace(".bmp", "")
        img_name = img_name.replace(".jpg", "")
        print(img_name)

        ann_path = os.path.join(voc_Annotations, "%s.xml" % img_name)
        dic_bbox = annot_box_loc(ann_path)
        bbox_cls = []
        for cls, list_bbox in dic_bbox.items():
            ix = idx2names[cls]
            for bbox in list_bbox:
                bbox_cls.append((bbox, ix))
        data.append([img_path, bbox_cls])
    return data


def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    train_set_path = os.path.join(voc_JPEGImages, "train")
    train_list = [
        train
        for train in os.listdir(train_set_path)
        if train.endswith(".png")
        or train.endswith(".JPG")
        or train.endswith("jpg")
        or train.endswith("bmp")
    ]

    val_set_path = os.path.join(voc_JPEGImages, "val")
    val_list = [
        val
        for val in os.listdir(val_set_path)
        if val.endswith(".png")
        or val.endswith(".JPG")
        or val.endswith("jpg")
        or val.endswith("bmp")
    ]

    classes_name = get_classes_name()

    idx2names = {}
    for i, name in enumerate(classes_name):
        if is_int(name):
            idx2names[name] = int(name)
        else:
            idx2names[name] = i


    print(idx2names)
    train = get_data(train_list, idx2names, "train")
    val = get_data(val_list, idx2names, "val")
    val = val[-len(val) // 2 :]
    data_info = {"classes_name": classes_name, "train": train, "val": val}
    torch.save(data_info, "/centernet/public/data.pth")
    print("classes_name", len(classes_name), classes_name)
    print("train", len(train), "val", len(val))
    print(train)
    print(val[:10])
