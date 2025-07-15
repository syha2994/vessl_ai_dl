import os
import cv2
import numpy as np

# 폴더 경로 지정
folder_path = '/Users/syha/2022_CREFLE/00.Project/07.fasion_robotics/20230502-DF-8279/100um/DF-8279/test/bad'

# 폴더 내 이미지 파일들 가져오기
img_files = [f for f in os.listdir(folder_path) if f.endswith('.bmp')]
print(img_files)

for img_file in img_files:
    # 이미지 읽기
    img_path = os.path.join(folder_path, img_file)
    img = cv2.imread(img_path)

    # 이미지 크기 가져오기
    height, width, channels = img.shape

    # 검정색 이미지 생성
    black_img = np.zeros((height, width, channels), dtype=np.uint8)

    # 이미지 저장
    save_path = f'/Users/syha/2022_CREFLE/00.Project/07.fasion_robotics/20230502-DF-8279/100um/DF-8279/ground_truth/bad/{img_file[:-4]}_mask.png'
    cv2.imwrite(save_path, black_img)
