import os
import cv2

# 변환할 디렉토리들의 경로 리스트
dir_list = ['/home/storage_disk2/fabric_data/test_data_16/bad',
            '/home/storage_disk2/fabric_data/test_data_16/good']

# 각 디렉토리마다
for dir_path in dir_list:
    # 디렉토리 내의 파일들을 차례대로 읽어들임
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        # 파일 확장자가 이미지 파일인 경우
        if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print(filename)
            # 이미지를 흑백으로 변환
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            # 변환한 이미지를 저장
            new_filepath = os.path.join(dir_path, filename)
            cv2.imwrite(new_filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
