import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class HeatMap:
    def __init__(self):
        self.color_degree = 255.
        self.vmin = 1.
        self.vmax = 255. * 0.4+ self.vmin * 0.6
        self.norm = colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        self.alpha = 0.2
        self.isSave = False

    def draw(self, image, score_map, f_name, idx, threshold=0.5):  # 추가: threshold 값을 설정하여 임계값 이상의 영역을 찾습니다.
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        resized_score_map = cv2.resize(score_map,
                                       dsize=(image.shape[1], image.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)

        resized_score_map = resized_score_map * self.color_degree
        fig = plt.figure(frameon=False)
        x = image.shape[1] / fig.dpi
        y = image.shape[0] / fig.dpi
        fig.set_figwidth(x)
        fig.set_figheight(y)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        plt.imshow(image, interpolation='none', aspect='auto')
        plt.imshow(resized_score_map, norm=self.norm, cmap='jet', interpolation='none', aspect='auto', alpha=self.alpha)
        fig.canvas.draw()
        composed_img = np.array(fig.canvas.renderer._renderer)

        # Find coordinates of regions above the threshold
        thresholded_indices = np.where(resized_score_map >= threshold)
        if len(thresholded_indices[0]) > 0:
            min_row = np.min(thresholded_indices[0])
            max_row = np.max(thresholded_indices[0])
            min_col = np.min(thresholded_indices[1])
            max_col = np.max(thresholded_indices[1])

            # Draw rectangle around the detected region
            cv2.rectangle(composed_img, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)

        composed_img = cv2.cvtColor(composed_img, cv2.COLOR_BGR2RGB)
        composed_img = cv2.putText(composed_img, str(np.max(score_map)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if self.isSave:
            from pathlib import Path
            name = Path(f_name).name
            rename = name.rsplit('.', 1)[0] + '_' + str(idx) + '.png'
            cv2.imwrite(os.path.join('./heatmap', rename), composed_img)

        fig.clear()
        plt.close(fig)

        return composed_img
