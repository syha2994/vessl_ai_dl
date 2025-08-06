import os
import cv2
import time
import torch
import numpy as np
import csv
from dataclasses import dataclass
from torchvision.ops import box_convert

# Grounding DINO와 SAM 모델 로드 유틸리티
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import Model  # Model.preprocess_image 사용
from groundingdino.util.inference import load_model as load_dino, predict as dino_predict

# OCR 라이브러리 추가
from paddleocr import PaddleOCR


@dataclass
class AnalogGaugeInspectorParams:
    dino_config: str = os.getenv("DINO_CONFIG", "/Users/seungyeon/PycharmProjects/git/AIagent/ai-agent/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    gauge_dino_weights: str = os.getenv("GAUGE_DINO_WEIGHTS", "/Users/seungyeon/PycharmProjects/git/AIagent/ai-agent/GroundingDINO/public/model/groundingdino_swint_ogc.pth")
    needle_dino_weights: str = os.getenv("NEEDLE_DINO_WEIGHTS", "/Users/seungyeon/PycharmProjects/git/AIagent/ai-agent/GroundingDINO/public/model/groundingdino_swint_ogc.pth")
    sam_checkpoint: str = os.getenv("SAM_CHECKPOINT", "/Users/seungyeon/PycharmProjects/git/AIagent/ai-agent//GroundingDINO/public/model/sam_vit_h_4b8939.pth")
    image_dir: str = os.getenv("IMAGE_DIR", "/Users/seungyeon/CREFLE/2.data/eq0/analog_gauge/various/square_preprocessing")
    result_dir: str = os.getenv("RESULT_DIR", "/Users/seungyeon/CREFLE/2.data/eq0/analog_gauge/various/square_preprocessing/result")
    box_threshold_gauge: float = float(os.getenv("BOX_THRESHOLD_GAUGE", 0.3))
    text_threshold_gauge: float = float(os.getenv("TEXT_THRESHOLD_GAUGE", 0.25))
    box_threshold_needle: float = float(os.getenv("BOX_THRESHOLD_NEEDLE", 0.2))
    text_threshold_needle: float = float(os.getenv("TEXT_THRESHOLD_NEEDLE", 0.2))
    min_value: float = float(os.getenv("MIN_VALUE", -20))
    max_value: float = float(os.getenv("MAX_VALUE", 60))
    tick_interval: float = float(os.getenv("TICK_INTERVAL", 10))
    analog_gauge_caption: str = os.getenv("ANALOG_GAUGE_CAPTION", "analog gauge dial")
    needle_caption: str = os.getenv("NEEDLE_CAPTION", "needle pointer, clock hands, needle")
    mode: str = os.getenv("MODE", "filename")
    paddle_use_textline_orientation: bool = bool(os.getenv("PADDLE_USE_TEXTLINE_ORIENTATION", False))
    paddle_use_doc_unwarping: bool = bool(os.getenv("PADDLE_USE_UNWARPING", False))
    paddle_lang: str = os.getenv("PADDLE_LANG", 'ch')


class AnalogGaugeInspector:
    def __init__(self, params: AnalogGaugeInspectorParams):
        self.params = params
        os.makedirs(self.params.result_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print("Loading Grounding DINO model...")
        self.gauge_dino_model = load_dino(self.params.dino_config, self.params.gauge_dino_weights)
        self.gauge_dino_model.to(self.device)
        self.needle_dino_model = load_dino(self.params.dino_config, self.params.needle_dino_weights)
        self.needle_dino_model.to(self.device)

        print("Loading SAM model...")
        self.sam_model = sam_model_registry["vit_h"](checkpoint=self.params.sam_checkpoint)
        self.sam_model.to(self.device)
        self.sam_predictor = SamPredictor(self.sam_model)

        print("Initializing PaddleOCR reader...")
        self.paddle_ocr = PaddleOCR(
            use_textline_orientation=self.params.paddle_use_textline_orientation,
            use_doc_unwarping=self.params.paddle_use_doc_unwarping,
            lang=self.params.paddle_lang
        )

    @staticmethod
    def scale_and_convert_boxes_to_xyxy(boxes: torch.Tensor, image: np.ndarray) -> np.ndarray:
        """
        중심 좌표 기반 바운딩 박수 (cx, cy, w, h)를 (x1, y1, x2, y2) 형식으로 변환하고 이미지 크기에 맞게 스케일링합니다.
        :param boxes: 텐서 형태의 바운딩 박스 (cx, cy, w, h)
        :param image: 원본 이미지 (높이와 너비 정보 추출용)
        :return: 스케일링된 바운딩 박스 (x1, y1, x2, y2) 형태의 넘파이 배열
        """
        image_shape = image.shape
        height, width = image_shape[:2]
        scale_factors = torch.tensor([width, height, width, height], dtype=boxes.dtype, device=boxes.device)
        scaled_boxes = boxes * scale_factors
        boxes_xyxy = box_convert(scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        return boxes_xyxy

    @staticmethod
    def select_highest_confidence_box(boxes, logits):
        if boxes.shape[0] > 0:
            best_idx = torch.argmax(logits)
            return boxes[best_idx]
        return None

    def select_valid_needle_box(self, boxes: torch.Tensor, logits: torch.Tensor, image):
        """
        가장 높은 confidence를 가진 needle box 중, 이미지 영역의 50%를 넘지 않는 유효한 박스를 선택합니다.
        추가적으로, 박스가 이미지 경계에서 너무 가까운 경우(5% 이내)는 무시합니다.
        """
        if boxes.shape[0] == 0:
            return None

        image_area = image.shape[0] * image.shape[1]
        height, width = image.shape[:2]
        margin_x = width * 0.01
        margin_y = height * 0.01
        sorted_indices = torch.argsort(logits, descending=True)

        for idx in sorted_indices:
            box = boxes[idx]
            box_xyxy = self.scale_and_convert_boxes_to_xyxy(box, image)
            x1, y1, x2, y2 = box_xyxy
            # Skip boxes too close to the image border
            if x1 < margin_x or y1 < margin_y or x2 > width - margin_x or y2 > height - margin_y:
                continue
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < 0.4 * image_area:
                return box

        return None

    def handle_missing_detection(self, image_np, image_name, message, text_color):
        print(f"{image_name}. {message}. Skipping...")
        cv2.putText(image_np, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
        cv2.imwrite(os.path.join(self.params.result_dir, f"ocr_vis_{image_name}"), image_np)

    def make_cannyedge_image(self, image):
        """
        입력 이미지에서 Canny 엣지를 검출하고 그 결과를 이미지에 그려 반환합니다.
        :param image: BGR 이미지 (예: cropped_image_np)
        :return: Canny 엣지가 그려진 이미지
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def find_ellipse_at_edge_image(self, edge_image):
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        # 가장 큰 컨투어를 찾습니다.
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) < 5:
            return None
        # 컨투어에서 타원을 피팅합니다.
        ellipse = cv2.fitEllipse(largest_contour)
        return ellipse

    def process_image(self, image_name):
        image_path = os.path.join(self.params.image_dir, image_name)
        image_bgr = cv2.imread(image_path)
        print(f"\nProcessing {image_name}...")

        if self.params.mode == 'manual':
            try:
                self.params.min_value = float(input("Enter min value: "))
                self.params.max_value = float(input("Enter max value: "))
                self.params.tick_interval = float(input("Enter tick interval: "))
            except ValueError:
                print("Invalid input. Using default values.")

        elif self.params.mode == 'filename':
            try:
                name_part = os.path.splitext(image_name)[0]  # remove extension
                parts = name_part.split("_")
                if len(parts) < 4:
                    raise ValueError("Filename does not contain enough parts for parameter parsing.")
                min_val, max_val, tick_interval, real_value = map(float, parts[:4])
                self.params.min_value = min_val
                self.params.max_value = max_val
                self.params.tick_interval = tick_interval
            except Exception as e:
                print(f"Failed to parse parameters from filename {image_name}: {e}. Using default values.")

        # -------- Step1. Grounding DINO 를 이용해 아날로그 게이지 객체 탐지 --------
        start_step1 = time.time()
        image_tensor = Model.preprocess_image(image_bgr)
        boxes_gauge, logits_gauge, phrases_gauge = dino_predict(
            model=self.gauge_dino_model,
            image=image_tensor,
            caption=self.params.analog_gauge_caption,
            box_threshold=self.params.box_threshold_gauge,
            text_threshold=self.params.text_threshold_gauge,
            device=self.device
        )
        print(f"Step1 (Gauge Detection) time: {time.time() - start_step1:.3f}s")
        box_gauge = self.select_highest_confidence_box(boxes_gauge, logits_gauge)
        if box_gauge is None:
            self.handle_missing_detection(image_bgr, image_name, "No gauge detected", (0, 0, 255))
            return

        gauge_box_xyxy = self.scale_and_convert_boxes_to_xyxy(box_gauge, image_bgr)
        x1, y1, x2, y2 = gauge_box_xyxy.astype(int)
        if 0 <= x1 < x2 <= image_bgr.shape[1] and 0 <= y1 < y2 <= image_bgr.shape[0]:
            cropped_image_np = image_bgr[y1:y2, x1:x2]
        else:
            self.handle_missing_detection(image_bgr, image_name, "Invalid crop box for gauge", (0, 0, 255))
            return

        cropped_image_np_vis = cropped_image_np.copy()

        edge_image = self.make_cannyedge_image(cropped_image_np)
        ellipse = self.find_ellipse_at_edge_image(edge_image)
        if ellipse is not None:
            cv2.ellipse(cropped_image_np_vis, ellipse, (0, 255, 255), 2)
            print(f"Detected ellipse at {ellipse[0]} with axes {ellipse[1]} and angle {ellipse[2]}")

        # -------- Step2. Grounding DINO 를 이용해 바늘 객체 탐지 --------
        start_step2 = time.time()
        cropped_image_tensor = Model.preprocess_image(cropped_image_np)
        boxes_needle, logits_needle, phrases_needle = dino_predict(
            model=self.needle_dino_model,
            image=cropped_image_tensor,
            caption=self.params.needle_caption,
            box_threshold=self.params.box_threshold_needle,
            text_threshold=self.params.text_threshold_needle,
            device=self.device
        )
        print(f"Step2 (Needle Detection) time: {time.time() - start_step2:.3f}s")

        if boxes_needle.shape[0] > 0:
            selected_box = self.select_valid_needle_box(boxes_needle, logits_needle, cropped_image_np)
            if selected_box is None:
                self.handle_missing_detection(cropped_image_np_vis, image_name, "No suitable needle box found", (0, 0, 255))
                return
            else:
                box_needle_xyxy = self.scale_and_convert_boxes_to_xyxy(selected_box, cropped_image_np)
                x1, y1, x2, y2 = box_needle_xyxy.astype(int)
                cv2.rectangle(cropped_image_np_vis, (x1, y1), (x2, y2), color=(0, 200, 0), thickness=2)
        else:
            self.handle_missing_detection(cropped_image_np_vis, image_name, "No needle boxes detected", (0, 0, 255))
            return

        # -------- Step3. SAM 모델을 이용해 바늘 마스크 예측 --------
        start_step3 = time.time()
        self.sam_predictor.set_image(cropped_image_np)
        needle_mask, _, _ = self.sam_predictor.predict(box=box_needle_xyxy, multimask_output=False)
        print(f"Step3 (SAM Mask Predict) time: {time.time() - start_step3:.3f}s")

        # 마스크를 원본 이미지에 덧씌우기
        colored_mask = np.zeros_like(cropped_image_np_vis, dtype=np.uint8)
        colored_mask[needle_mask[0]] = [0, 0, 255]
        cropped_image_np_vis = cv2.addWeighted(cropped_image_np_vis, 1.0, colored_mask, 0.8, 0)

        # -------- Step4. Sam으로 예측된 바늘 마스크에서 컨투어 추출 --------
        start_step4 = time.time()
        contours, _ = cv2.findContours(needle_mask[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Step4 (Contour Extraction) time: {time.time() - start_step4:.3f}s")
        if len(contours) == 0:
            self.handle_missing_detection(cropped_image_np_vis, image_name, "No contours found for needle", (0, 0, 255))
            return

        # 가장 큰 컨투어를 바늘로 간주
        needle_contour = max(contours, key=cv2.contourArea)

        # 바늘 마스크의 무게중심을 계산해 게이지의 중심점으로 사용
        moments = cv2.moments(needle_mask[0].astype(np.uint8))
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            gauge_axis = (center_x, center_y)
            cv2.circle(cropped_image_np_vis, gauge_axis, radius=5, color=(255, 255, 0), thickness=-1)
        else:
            self.handle_missing_detection(cropped_image_np_vis, image_name, "No valid moments found for needle", (0, 0, 255))
            return

        # -------- Step5. PaddleOCR를 이용해 눈금 텍스트 추출 및 시각화 --------
        start_step5 = time.time()
        ocr_result_with_boxes = self.paddle_ocr.ocr(cropped_image_np)[0]
        print(f"Step5 (OCR) time: {time.time() - start_step5:.3f}s")

        if ocr_result_with_boxes is None:
            self.handle_missing_detection(cropped_image_np_vis, image_name, "OCR failed or returned None", (0, 0, 255))
            return

        expected_values = np.arange(self.params.min_value, self.params.max_value + 1e-5, self.params.tick_interval)
        expected_texts = set([str(int(v)) if v == int(v) else f"{v:.1f}" for v in expected_values])

        # OCR 박스 중심 위치와 텍스트 매핑
        ocr_centers = []
        value_angles = []

        for box, (text, score) in ocr_result_with_boxes:
            if text in expected_texts and score > 0.7:
                pts = np.array(box)
                center_x = int(np.mean(pts[:, 0]))
                center_y = int(np.mean(pts[:, 1]))
                try:
                    value = float(text)
                    angle = np.arctan2(center_y - gauge_axis[1], center_x - gauge_axis[0])
                    ocr_centers.append((center_x, center_y))
                    value_angles.append((value, angle))
                    # 시각화
                    cv2.polylines(
                        cropped_image_np_vis, [pts.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2
                    )
                    cv2.circle(
                        cropped_image_np_vis, (center_x, center_y), 7, (180, 105, 255), -1
                    )
                    cv2.line(
                        cropped_image_np_vis, (center_x, center_y), gauge_axis, (180, 105, 255), 3
                    )
                    cv2.putText(
                        cropped_image_np_vis, text,
                        (int(pts[0][0]), int(pts[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )
                except:
                    continue

        # -------- Step6. 바늘과 OCR 중심점 간의 거리 계산 및 바늘 위치 결정 --------
        start_step6 = time.time()
        if len(ocr_centers) < 2:
            self.handle_missing_detection(cropped_image_np_vis, image_name, "Not enough valid OCR values found", (0, 0, 255))
            print(f"Step6 (Needle Point Selection) time: {time.time() - start_step6:.3f}s")
            return

        # 중심으로부터 가장 먼 점을 첫 번째 점으로 선택
        distances = [(pt, np.linalg.norm(np.array(pt[0]) - np.array(gauge_axis))) for pt in needle_contour]
        distances.sort(key=lambda x: x[1], reverse=True)
        needle_point_1, dist1 = distances[0]

        # needle_point_1 기준 반대 방향에 있는 점을 needle_point_2로 선택
        direction_1 = np.array(needle_point_1[0]) - np.array(gauge_axis)
        direction_1_unit = direction_1 / (np.linalg.norm(direction_1) + 1e-6)
        most_opposite_score = 1.0
        needle_point_2 = None
        for pt, dist in distances[1:]:
            direction = np.array(pt[0]) - np.array(gauge_axis)
            direction_unit = direction / (np.linalg.norm(direction) + 1e-6)
            dot_product = np.dot(direction_unit, direction_1_unit)
            # dot_product가 -1에 가까울수록 반대 방향
            if dot_product < most_opposite_score:
                most_opposite_score = dot_product
                needle_point_2 = pt
                dist2 = dist
        if needle_point_2 is None:
            needle_point_2, dist2 = distances[1]

        if abs(dist1 - dist2) / max(dist1, dist2) > 0.03:
            needle_point = needle_point_1
        else:
            print('a')

            def min_dist_to_ocr(pt):
                return min([np.linalg.norm(np.array(pt) - np.array(center_ocr)) for center_ocr in ocr_centers])

            pt1_dist = min_dist_to_ocr(needle_point_1[0])
            pt2_dist = min_dist_to_ocr(needle_point_2[0])
            needle_point = needle_point_1 if pt1_dist < pt2_dist else needle_point_2

        cv2.circle(cropped_image_np_vis, tuple(needle_point[0]), radius=5, color=(0, 255, 0), thickness=-1)
        print(f"Step6 (Needle Point Selection) time: {time.time() - start_step6:.3f}s")

        # -------- Step7. 바늘 각도 계산 및 게이지 값 추정 --------
        start_step7 = time.time()
        needle_angle = np.arctan2(needle_point[0][1] - gauge_axis[1], needle_point[0][0] - gauge_axis[0])

        if len(value_angles) >= 2:
            # 각도 정렬
            value_angles.sort(key=lambda x: x[1])
            values, angles = zip(*value_angles)
            values = np.array(values)
            angles = np.unwrap(np.array(angles))  # angle discontinuity 보정

            # 각도에 대한 보간 함수 생성
            from scipy.interpolate import interp1d
            angle_to_value = interp1d(angles, values, kind='linear', fill_value='extrapolate')

            # 바늘 각도 보정 및 게이지 값 추정
            needle_angle_unwrapped = np.unwrap([angles[0], needle_angle])[1]
            estimated_value = angle_to_value(needle_angle_unwrapped)

            print(f"Estimated gauge value: {estimated_value:.3f}")
            cv2.putText(cropped_image_np_vis, f"{estimated_value:.1f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

            vis_output_path = os.path.join(self.params.result_dir, f"ocr_vis_{image_name}")
            cv2.imwrite(vis_output_path, cropped_image_np_vis)
            print(f"OCR visualization saved to {vis_output_path}")

            # ---------------- VESSL 로그에 이미지 업로드 -------------------
            import vessl
            vessl.init()

            vessl.log({
                "edge_image": vessl.Image(
                    data=edge_image,
                    caption=f"{image_name} - Canny Edge Detection"
                )
            })
            vessl.log({
                "analog_gauge_result": vessl.Image(
                    data=cropped_image_np_vis,
                    caption=f"{image_name} - Analog Gauge Result"
                )
            })
            vessl.log({
                "original_image": vessl.Image(
                    data=image_bgr,
                    caption=f"{image_name} - Original Image"
                )
            })
            # -----------------------------------------------------------
            # --- 정확도 계산 및 결과 반환 ---
            try:
                s, e, _, r = map(float, os.path.splitext(image_name)[0].split("_"))
                R = e - s
                D = abs(estimated_value - r)
                E = D / R
                A = 1 - E
                print(f"[Result] {image_name} | Real: {r} | Predicted: {estimated_value:.2f} | Accuracy: {A*100:.2f}%")
                return image_name, r, estimated_value, A * 100
            except Exception as ex:
                print(f"정확도 계산 실패: {ex}")
                return None
        else:
            print("Not enough valid OCR values for interpolation.")
        print(f"Step7 (Angle & Value Estimation) time: {time.time() - start_step7:.3f}s")
        return None

    def run(self):
        import torch
        print('torch.cuda.is_available: ', torch.cuda.is_available())
        print('torch.cuda.get_device_name: ', torch.cuda.get_device_name(0))

        csv_output_path = os.path.join(self.params.result_dir, "accuracy_results.csv")
        with open(csv_output_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Image Name", "Real Value", "Predicted Value", "Accuracy (%)"])

            total_accuracy = 0
            count = 0

            for image_name in os.listdir(self.params.image_dir):
                if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                print('##################################################')
                start_total = time.time()
                result = self.process_image(image_name)
                print(f"Total time for {image_name}: {time.time() - start_total:.3f}s")
                print('##################################################')

                if result is not None:
                    image_name, real, predicted, accuracy = result
                    writer.writerow([image_name, real, f"{predicted:.2f}", f"{accuracy:.2f}"])
                    total_accuracy += accuracy
                    count += 1

            if count > 0:
                final_accuracy = total_accuracy / count
                print(f"\n✅ 모든 이미지 처리 완료! 평균 정확도: {final_accuracy:.2f}%")
            else:
                print("\n⚠️ 처리된 유효한 이미지가 없습니다.")


if __name__ == "__main__":
    params = AnalogGaugeInspectorParams()
    inspector = AnalogGaugeInspector(params)
    inspector.run()