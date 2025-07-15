from PIL import Image

class Fabric_Split:
    def __init__(self):
        self.x = 1384
        self.y = 0
        self.width = 14000
        self.height = 1000
        self.patch_size = 1000


    def run(self, img_array):
        patches = []
        crop_array = img_array[self.y:self.y + self.height, self.x:self.x + self.width]
        crop_img = Image.fromarray(crop_array)

        for i in range(0, self.height, self.patch_size):
            for j in range(0, self.width, self.patch_size):
                # 패치 영역 지정
                patch_x = self.x + j
                patch_y = self.y + i
                patch_width = min(self.patch_size, self.width - j)
                patch_height = min(self.patch_size, self.height - i)

                # 패치 이미지 자르기
                patch_array = crop_array[i:i + patch_height, j:j + patch_width]
                # patch_img = Image.fromarray(patch_array)
                patches.append(patch_array)
        return patches
