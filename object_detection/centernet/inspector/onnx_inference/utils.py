import numpy as np
from scipy.ndimage.filters import maximum_filter


def topk(hm, k):
    batch, cls, h, w = hm.shape
    out = maximum_filter(hm, size=3, mode="constant")
    keep_max = (out == hm).astype(float)
    hm = keep_max * hm

    topk_scores = []
    topk_indexs = []
    topk_cls = []
    topk_xs = []
    topk_ys = []

    for b in range(batch):
        topk_scores_b = []
        topk_indexs_b = []
        topk_cls_b = []
        topk_xs_b = []
        topk_ys_b = []

        for c in range(cls):
            hm_flat = hm[b, c].flatten()
            topk_inds = np.argpartition(hm_flat, -k)[-k:]
            topk_scores_c = hm_flat[topk_inds]
            topk_ys_c, topk_xs_c = np.unravel_index(topk_inds, (h, w))

            topk_scores_b.append(topk_scores_c)
            topk_indexs_b.append(topk_inds)
            topk_cls_b.append(np.full_like(topk_scores_c, c))
            topk_xs_b.append(topk_xs_c)
            topk_ys_b.append(topk_ys_c)

        topk_scores.append(np.concatenate(topk_scores_b))
        topk_indexs.append(np.concatenate(topk_indexs_b))
        topk_cls.append(np.concatenate(topk_cls_b))
        topk_xs.append(np.concatenate(topk_xs_b))
        topk_ys.append(np.concatenate(topk_ys_b))

    topk_scores = np.stack(topk_scores)
    topk_indexs = np.stack(topk_indexs)
    topk_cls = np.stack(topk_cls)
    topk_xs = np.stack(topk_xs)
    topk_ys = np.stack(topk_ys)

    return topk_scores, topk_indexs, topk_cls, topk_xs, topk_ys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def heatmap_bbox(hm, wh, reg, k=100):
    hm = sigmoid(hm)
    scores, indexs, cls, xs, ys = topk(hm, k)
    batch = reg.shape[0]
    reg = reg.reshape(batch, 2, -1).transpose(0, 2, 1)  # (batch,w*h,2)
    reg_indexs = np.repeat(indexs[:, :, np.newaxis], 2, axis=2)  # (batch,k,2)
    reg = np.take_along_axis(reg, reg_indexs, axis=1)  # (batch,k,2)
    xs = xs.astype(float) + reg[:, :, 0]
    ys = ys.astype(float) + reg[:, :, 1]
    wh = wh.reshape(batch, 2, -1).transpose(0, 2, 1)
    wh = np.take_along_axis(wh, reg_indexs, axis=1)  # ((batch,k,2)
    bbox = np.stack(
        (
            xs - wh[:, :, 0] / 2,
            ys - wh[:, :, 1] / 2,
            xs + wh[:, :, 0] / 2,
            ys + wh[:, :, 1] / 2,
        ),
        axis=-1,
    )  # (batch,k,4)
    return bbox, cls, scores
