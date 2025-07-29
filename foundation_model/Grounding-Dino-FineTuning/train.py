import os
import cv2
import csv
import vessl
import torch
import argparse
import torch.optim as optim
from collections import defaultdict
from groundingdino.util.train import load_model, load_image, train_image


def draw_box_with_label(image, output_path, coordinates, label, color=(0, 0, 255), thickness=2, font_scale=0.5):
    """
    Draw a box and a label on an image using OpenCV.

    Parameters:
    - image (str):  Input image.
    - output_path (str): Path to save the image with the box and label.
    - coordinates (tuple): A tuple (x1, y1, x2, y2) indicating the top-left and bottom-right corners of the box.
    - label (str): The label text to be drawn next to the box.
    - color (tuple, optional): Color of the box and label in BGR format. Default is red (0, 0, 255).
    - thickness (int, optional): Thickness of the box's border. Default is 2 pixels.
    - font_scale (float, optional): Font scale for the label. Default is 0.5.
    """

    # Draw the rectangle
    cv2.rectangle(image, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), color, thickness)

    # Define a position for the label (just above the top-left corner of the rectangle)
    label_position = (coordinates[0], coordinates[1]-10)

    # Draw the label
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    # Save the modified image
    cv2.imwrite(output_path, image)


def read_dataset(ann_file, img_dir):
    ann_Dict= defaultdict(lambda: defaultdict(list))
    with open(ann_file) as file_obj:
        ann_reader= csv.DictReader(file_obj)
        # Iterate over each row in the csv file
        # using reader object
        for row in ann_reader:
            #print(row)
            img_n=os.path.join(img_dir, row['image_name'])
            x1=int(row['bbox_x'])
            y1=int(row['bbox_y'])
            x2=x1+int(row['bbox_width'])
            y2=y1+int(row['bbox_height'])
            label=row['label_name']
            ann_Dict[img_n]['boxes'].append([x1,y1,x2,y2])
            ann_Dict[img_n]['captions'].append(label)
    return ann_Dict


def train(model, ann_file, epochs=1, save_path='weights/model_weights', save_epoch=50, image_dir='', vessl_model_repo='', vessl_model_number=None):
    # Read Dataset
    ann_Dict = read_dataset(ann_file, image_dir)

    # Add optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Ensure the model is in training mode
    model.train()

    for epoch in range(epochs):
        total_loss = 0  # Track the total loss for this epoch
        for idx, (IMAGE_PATH, vals) in enumerate(ann_Dict.items()):
            image_source, image = load_image(IMAGE_PATH)
            bxs = vals['boxes']
            captions = vals['captions']

            # Zero the gradients
            optimizer.zero_grad()

            # Call the training function for each image and its annotations
            loss = train_image(
                model=model,
                image_source=image_source,
                image=image,
                caption_objects=captions,
                box_target=bxs,
            )

            # Backpropagate and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # Accumulate the loss
            print(f"Processed image {idx+1}/{len(ann_Dict)}, Loss: {loss.item()}")

        # Print the average loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss / len(ann_Dict)}")
        if (epoch%save_epoch)==0:
            # Save the model's weights after each epoch
            torch.save(model.state_dict(), f"{save_path}{epoch}.pth")
            print(f"Model weights saved to {save_path}{epoch}.pth")
            if vessl_model_repo:
                print(f"Registering model to VESSL repository: {vessl_model_repo}")
                vessl.register_torch_model(
                    repository_name=vessl_model_repo,
                    model_number=vessl_model_number,
                    model_instance=model,
                    requirements=["torch"],
                )


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train Grounding DINO model using VESSL")

    parser.add_argument("--config_path", type=str, default=os.getenv("CONFIG_PATH", "groundingdino/config/GroundingDINO_SwinT_OGC.py"), help="Path to model config file")
    parser.add_argument("--weights_path", type=str, default=os.getenv("WEIGHTS_PATH", "weights/groundingdino_swint_ogc.pth"), help="Path to model weights")
    parser.add_argument("--images_dir", type=str, default=os.getenv("IMAGES_DIR", "multimodal-data/images"), help="Directory with training images")
    parser.add_argument("--ann_file", type=str, default=os.getenv("ANN_FILE", "multimodal-data/annotation/annotation.csv"), help="CSV annotation file")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", 100)), help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default=os.getenv("SAVE_PATH", "weights/model_weights"), help="Directory to save model weights")
    parser.add_argument("--save_epoch", type=int, default=int(os.getenv("SAVE_EPOCH", 50)), help="Epoch interval to save model weights")
    parser.add_argument("--vessl_model_repo", type=str, default=int(os.getenv("VESSL_MODEL_REPO", None)), help="Epoch interval to save model weights")
    parser.add_argument("--vessl_model_number", type=str, default=int(os.getenv("VESSL_MODEL_NUMBER", None)), help="Epoch interval to save model weights")

    args = parser.parse_args()

    # Update model and dataset paths
    model = load_model(args.config_path, args.weights_path)
    images_files = sorted(os.listdir(args.images_dir))
    ann_file = args.ann_file

    train(model=model, ann_file=ann_file, epochs=args.epochs, save_path=args.save_path, save_epoch=args.save_epoch, image_dir=args.images_dir, vessl_model_repo=args.vessl_model_repo, vessl_model_number=args.vessl_model_number)
