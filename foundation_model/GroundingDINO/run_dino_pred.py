import os
import cv2
import torch
import vessl
import numpy as np
from dataclasses import dataclass
from torchvision.ops import box_convert

from groundingdino.util.inference import Model  # Model.preprocess_image 사용
from groundingdino.util.inference import load_model as load_dino, predict as dino_predict


@dataclass
class GroundingDINOParams:
    dino_config: str = os.getenv("DINO_CONFIG", "./config/GroundingDINO_SwinT_OGC.py")
    dino_weights: str = os.getenv("DINO_WEIGHTS", "./public/model/groundingdino_swint_ogc.pth")
    image_dir: str = os.getenv("IMAGE_DIR", "./")
    result_dir: str = os.getenv("RESULT_DIR", "./")
    box_threshold: float = float(os.getenv("BOX_THRESHOLD_GAUGE", 0.3))
    text_threshold: float = float(os.getenv("TEXT_THRESHOLD_GAUGE", 0.25))
    caption: str = os.getenv("ANALOG_GAUGE_CAPTION", "analog gauge dial")
    topk: int = int(os.getenv("TOP_K", 1))


class GroundingDINOInspector:
    def __init__(self, params: GroundingDINOParams):
        self.params = params
        os.makedirs(self.params.result_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print("Loading Grounding DINO model...")
        self.dino_model = load_dino(self.params.dino_config, self.params.dino_weights)
        self.dino_model.to(self.device)

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

    def select_topk_dino_outputs(self, boxes, logits, phrases):
        """
        Grounding DINO 예측 결과 중 top_k 확률(logits)이 가장 높은 객체만 선택합니다.

        Args:
            boxes (Tensor): shape (N, 4) - 박스 좌표
            logits (Tensor): shape (N,) 또는 (N, 1) - 각 박스에 대한 confidence
            phrases (List[str]): shape (N,) - 각 박스에 대응되는 텍스트
            top_k (int): 선택할 top-k 개수

        Returns:
            topk_boxes (Tensor): shape (top_k, 4)
            topk_logits (Tensor): shape (top_k,)
            topk_phrases (List[str]): top_k 개 텍스트
        """
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        # top-k 인덱스 구하기
        topk_values, topk_indices = torch.topk(logits, k=self.params.topk)

        # top-k 요소 선택
        topk_boxes = boxes[topk_indices]
        topk_logits = logits[topk_indices]
        topk_phrases = [phrases[i] for i in topk_indices.tolist()]

        return topk_boxes, topk_logits, topk_phrases

    @staticmethod
    def scale_and_convert_boxes_to_xyxy(boxes: torch.Tensor, image: np.ndarray) -> np.ndarray:
        """
        중심 좌표 기반 바운딩 박스 (cx, cy, w, h)를 (x1, y1, x2, y2) 형식으로 변환하고 이미지 크기에 맞게 스케일링합니다.
        """
        height, width = image.shape[:2]
        scale_factors = torch.tensor(
            [width, height, width, height],
            dtype=boxes.dtype,
            device=boxes.device
        )
        scaled_boxes = boxes * scale_factors
        boxes_xyxy = box_convert(scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy")

        return boxes_xyxy.cpu().numpy()

    def process_image(self, image_name):
        image_path = os.path.join(self.params.image_dir, image_name)
        image_bgr = cv2.imread(image_path)
        print(f"\nProcessing {image_name}...")

        image_tensor = Model.preprocess_image(image_bgr)
        boxes, logits, phrases = dino_predict(
            model=self.dino_model,
            image=image_tensor,
            caption=self.params.caption,
            box_threshold=self.params.box_threshold,
            text_threshold=self.params.text_threshold,
            device=self.device
        )

        topk_boxes, topk_logits, topk_phrases = self.select_topk_dino_outputs(boxes, logits, phrases)
        xyxy_boxes = self.scale_and_convert_boxes_to_xyxy(topk_boxes, image_bgr)

        for box, caption in zip(xyxy_boxes, self.params.caption):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_bgr, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        vessl.log({
            "analog_gauge_result": vessl.Image(
                data=image_bgr,
                caption=f"Grounding DINO results for {image_name}",
            )
        })

    def run(self):
        import torch
        print('torch.cuda.is_available: ', torch.cuda.is_available())
        print('torch.cuda.get_device_name: ', torch.cuda.get_device_name(0))

        for image_name in os.listdir(self.params.image_dir):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            self.process_image(image_name)


if __name__ == "__main__":
    params = GroundingDINOParams()
    inspector = GroundingDINOInspector(params)
    inspector.run()