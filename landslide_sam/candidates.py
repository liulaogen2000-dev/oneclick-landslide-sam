"""Click-derived windows and SAM inference; reference information is not accepted."""
from dataclasses import dataclass
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image

SCENE_DATASETS = ('GVLM_v1_0', 'GVLM_v1_1')
TILE_DATASETS = ('CAS', 'jz', 'bijie')
SCALES = (512, 1024, 2048, 4096)


@dataclass
class Candidate:
    mask: np.ndarray
    window: tuple
    score: float
    stability: float
    logits: np.ndarray
    token: np.ndarray
    nominal_side: float


def windows(shape, point_yx, dataset, gsd=None):
    h, w = shape[:2]
    py, px = point_yx
    if not (0 <= py < h and 0 <= px < w):
        raise ValueError('Click lies outside the image')
    if dataset in TILE_DATASETS:
        return [(0, 0, h, w)]
    if dataset not in SCENE_DATASETS:
        raise ValueError('Unknown dataset mode')
    if dataset == 'GVLM_v1_0':
        rx = ry = .59
    else:
        if gsd is None:
            raise ValueError('GVLM_v1_1 needs native pixel size (x metres, y metres)')
        rx, ry = gsd
    if rx <= 0 or ry <= 0:
        raise ValueError('Pixel size must be positive')
    result = []
    for side in SCALES:
        hh = min(h, max(1, int(np.floor(side * .59 / ry + .5))))
        ww = min(w, max(1, int(np.floor(side * .59 / rx + .5))))
        y = max(0, min(int(py) - hh // 2, h - hh))
        x = max(0, min(int(px) - ww // 2, w - ww))
        result.append((y, x, hh, ww))
    return result


class FrozenSAM:
    """CUDA backend and precision follow final_v1/code/extract.py."""
    def __init__(self, checkpoint, config='configs/sam2.1/sam2.1_hiera_l.yaml'):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        if not torch.cuda.is_available():
            raise RuntimeError('Frozen inference protocol requires a CUDA GPU')
        torch.manual_seed(0)
        np.random.seed(0)
        torch.set_num_threads(4)
        self.predictor = SAM2ImagePredictor(build_sam2(config, str(checkpoint), device='cuda'))
        self.predictor.model.eval()
        for p in self.predictor.model.parameters():
            p.requires_grad_(False)
        self.token = None
        def capture(module, inputs):
            self.token = inputs[0].detach().float().cpu().numpy()[0].copy()
        self.predictor.model.sam_mask_decoder.iou_prediction_head.register_forward_pre_hook(capture)

    def generate(self, rgb, point_yx, dataset, gsd=None):
        import torch
        legacy = dataset in ('jz', 'bijie')
        torch.use_deterministic_algorithms(not legacy)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = not legacy
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = legacy
        result = []
        for s, win in enumerate(windows(rgb.shape, point_yx, dataset, gsd)):
            y, x, h, w = win
            crop = np.ascontiguousarray(rgb[y:y+h, x:x+w])
            if legacy:
                image, rh, rw = crop, h, w
            else:
                rh, rw = max(1, int(h * 1024 / max(h, w) + .5)), max(1, int(w * 1024 / max(h, w) + .5))
                image = np.asarray(Image.fromarray(crop).resize((rw, rh), Image.Resampling.BILINEAR))
            py, px = point_yx
            xy = np.array([[(px-x) * rw/w, (py-y) * rh/h]], np.float32)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=not legacy):
                self.predictor.set_image(image)
                lm, scores, _ = self.predictor.predict(point_coords=xy, point_labels=np.array([1], np.int32),
                                                       multimask_output=True, return_logits=True)
            if not legacy:
                with torch.inference_mode():
                    logits = torch.nn.functional.interpolate(torch.from_numpy(lm).cuda()[None],
                        size=(h, w), mode='bilinear', align_corners=False)[0].cpu().numpy()
            else:
                logits = lm
            if dataset in SCENE_DATASETS:
                nominal = SCALES[s]
            elif dataset == 'CAS':
                if gsd is None:
                    raise ValueError('CAS features need the native GSD')
                nominal = 512 * gsd[0] / .59
            else:
                nominal = max(h, w) * (.8/.59 if dataset == 'bijie' else 1.)
            for k in range(3):
                mask = logits[k] > 0
                stability = int((logits[k] > 1).sum()) / max(int((logits[k] > -1).sum()), 1)
                result.append(Candidate(mask, win, float(scores[k]), stability, logits[k], self.token.copy(), nominal))
        return result
