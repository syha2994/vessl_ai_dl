import os
import cv2
import time
import torch
import numpy as np
from torchvision.ops import box_convert

# Grounding DINO
from groundingdino.util.inference import load_model as load_dino, predict as dino_predict
from groundingdino.util.inference import Model  # Model.preprocess_image 사용


class DinoPredictor:
    def __init__(
            self,
            dino_config_path, dino_weight_path, image_bgr,
            device,
            caption, box_threshold=0.25, text_threshold=0.25
    ):
        self.device = device

        self.image_bgr = image_bgr

        self.dino_config_path = dino_config_path
        self.dino_weight_path = dino_weight_path
        self.dino_model = load_dino(self.dino_config_path, self.dino_weight_path)

        self.dino_model.to(self.device)

        self.caption = caption
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        print(f"Loaded Grounding DINO model from {self.dino_weight_path} on {self.device}")


    def predict_dino_model(self, image_tensor, caption, box_threshold, text_threshold):
        """
        Grounding DINO 모델을 사용하여 이미지에서 객체를 예측합니다.
        :param image_tensor: 전처리된 이미지 텐서
        :param caption: 객체 캡션 (예: "analog_gauge" 또는 "needle pointer")
        :return: 예측된 박스, 로짓, 문구
        """
        boxes, logits, phrases = dino_predict(
            model=self.dino_model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        return boxes, logits, phrases

    def run(self):
        image_tensor = Model.preprocess_image(self.image_bgr)

        boxes, logits, phrases = self.predict_dino_model(
            image_tensor, self.caption, self.box_threshold, self.text_threshold
        )
        return boxes, logits, phrases
