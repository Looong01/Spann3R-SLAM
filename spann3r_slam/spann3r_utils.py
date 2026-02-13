"""
Spann3R-backed utilities for SLAM.

All public runtime functions use the ``spann3r_*`` prefix.
"""

import os
import sys

import PIL
import einops
import numpy as np
import torch
import torch.nn.functional as F
import lietorch

# Add Spann3R source tree first so ``import dust3r`` / ``import spann3r``
# resolve to the upstream code copied into this repository.
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spann3r_core_dir = os.path.join(_root_dir, "spann3r_core")
if _spann3r_core_dir not in sys.path:
    sys.path.insert(0, _spann3r_core_dir)

import dust3r.utils.path_to_croco  # noqa: F401
from dust3r.utils.image import ImgNorm
from spann3r.model import Spann3R

from spann3r_slam.config import config
import spann3r_slam.matching as matching

DEFAULT_SPANN3R_CKPT = "checkpoints/spann3r.pth"
DEFAULT_DUST3R_CKPT = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"


class _FeatureRetrievalDatabase:
    """Lightweight retrieval without external retrieval dependencies.

    It stores one global descriptor per keyframe (mean pooled patch feature) and
    returns cosine-similar keyframe indices.
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.keyframe_desc = []

    def _get_descriptor(self, frame):
        feat = getattr(frame, "feat", None)
        if feat is None:
            return None
        feat = feat.reshape(-1, feat.shape[-1]).float()
        desc = F.normalize(feat.mean(dim=0), dim=0)
        return desc.detach().cpu()

    def update(self, frame, add_after_query, k, min_thresh=0.0):
        query_desc = self._get_descriptor(frame)
        topk_image_inds = []

        if query_desc is not None and self.keyframe_desc:
            db = torch.stack(self.keyframe_desc, dim=0)  # (N, D)
            scores = torch.mv(db, query_desc)  # cosine similarity

            n = min(int(k), scores.numel())
            if n > 0:
                vals, inds = torch.topk(scores, k=n)
                valid = vals > min_thresh
                topk_image_inds = inds[valid].tolist()

        if add_after_query and query_desc is not None:
            self.keyframe_desc.append(query_desc)

        return topk_image_inds


def _resolve_ckpt(path, default_rel_path):
    if path is not None:
        return path
    return os.path.join(_root_dir, default_rel_path)


def _frame_to_view(frame):
    return {
        "img": frame.img,
        "true_shape": frame.img_true_shape,
    }


def _extract_desc_and_conf(res):
    desc = res.get("desc")
    if desc is None:
        # Graceful fallback if a checkpoint does not include descriptor heads.
        desc = F.normalize(res["pts3d"], dim=-1)

    desc_conf = res.get("desc_conf")
    if desc_conf is None:
        desc_conf = res.get("conf")
    if desc_conf is None:
        desc_conf = torch.ones_like(res["pts3d"][..., 0])
    if desc_conf.dim() == 4 and desc_conf.shape[-1] == 1:
        desc_conf = desc_conf[..., 0]

    return desc, desc_conf


def _extract_outputs(res):
    desc, desc_conf = _extract_desc_and_conf(res)
    conf = res["conf"]
    if conf.dim() == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if desc_conf.dim() == 4 and desc_conf.shape[-1] == 1:
        desc_conf = desc_conf[..., 0]
    return res["pts3d"][0], conf[0], desc[0], desc_conf[0]


def load_spann3r(path=None, device="cuda", dust3r_path=None):
    """Load Spann3R model.

    Args:
        path: Path to Spann3R checkpoint (`*.pth`). Defaults to
              ``checkpoints/spann3r.pth``.
        device: Torch device.
        dust3r_path: Path to DUSt3R checkpoint. Defaults to
              ``checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth``.
    """
    spann3r_ckpt = _resolve_ckpt(path, DEFAULT_SPANN3R_CKPT)
    dust3r_ckpt = _resolve_ckpt(dust3r_path, DEFAULT_DUST3R_CKPT)

    if not os.path.exists(spann3r_ckpt):
        raise FileNotFoundError(f"Spann3R checkpoint not found: {spann3r_ckpt}")
    if not os.path.exists(dust3r_ckpt):
        raise FileNotFoundError(f"DUSt3R checkpoint not found: {dust3r_ckpt}")

    print(f"Loading DUSt3R backbone from {dust3r_ckpt}")
    model = Spann3R(dus3r_name=dust3r_ckpt, use_feat=False).to(device)

    print(f"Loading Spann3R weights from {spann3r_ckpt}")
    checkpoint = torch.load(spann3r_ckpt, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    if msg.missing_keys:
        print(f"[Spann3R] Missing keys: {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        print(f"[Spann3R] Unexpected keys: {len(msg.unexpected_keys)}")

    model.eval()
    return model


def load_spann3r_retriever(spann3r_model, retriever_path=None, device="cuda"):
    """Feature retrieval without MASt3R retrieval checkpoints."""
    _ = (spann3r_model, retriever_path, device)
    return _FeatureRetrievalDatabase(device=device)


@torch.inference_mode()
def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):
    dec1, dec2 = model.decode(feat1, pos1, feat2, pos2)
    with torch.cuda.amp.autocast(enabled=False):
        res1 = model.downstream_head(dec1, shape1, 1)
        res2 = model.downstream_head(dec2, shape2, 2)
    return res1, res2


def downsample(X, C, D, Q):
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        X = X[..., ::downsample, ::downsample, :].contiguous()
        C = C[..., ::downsample, ::downsample].contiguous()
        D = D[..., ::downsample, ::downsample, :].contiguous()
        Q = Q[..., ::downsample, ::downsample].contiguous()
    return X, C, D, Q


def _default_intrinsics(h, w, device, dtype):
    # 60 deg default field of view for unknown intrinsics.
    focal = 0.5 * max(h, w) / np.tan(np.deg2rad(60.0 / 2.0))
    return torch.tensor(
        [[focal, 0.0, (w - 1) * 0.5], [0.0, focal, (h - 1) * 0.5], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def _as_se3_pose(T):
    """Convert a Sim3 pose to SE3 by dropping the scale component."""
    if isinstance(T, lietorch.SE3):
        return T
    if isinstance(T, lietorch.Sim3):
        return lietorch.SE3(T.data[..., :7])
    return T


def _subsample_points(points, colors, max_points=None):
    if points is None or colors is None:
        return None, None
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points, colors

    step = int(np.ceil(points.shape[0] / float(max_points)))
    return points[::step], colors[::step]


def _collect_world_points(frame, conf_thresh, spatial_stride=1, max_points=None):
    if frame is None or frame.X_canon is None:
        return None, None

    X = frame.X_canon
    if X.ndim == 3:
        X = X.squeeze(0)
    if X.numel() == 0:
        return None, None

    C = None
    if getattr(frame, "C", None) is not None:
        N = int(getattr(frame, "N", 0))
        if N > 0:
            C = frame.get_average_conf()
        else:
            C = frame.C
    if C is None:
        C = torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)
    if C.ndim == 3:
        C = C.squeeze(0)

    rgb = frame.uimg.to(device=X.device, dtype=X.dtype).view(-1, 3)
    valid = torch.isfinite(X).all(dim=-1) & torch.isfinite(C[:, 0]) & (X[:, 2] > 1e-6)
    valid = valid & (C[:, 0] > conf_thresh)
    if valid.sum() == 0:
        return None, None

    idx = torch.where(valid)[0]
    if spatial_stride is not None and spatial_stride > 1:
        idx = idx[:: int(spatial_stride)]
    if idx.numel() == 0:
        return None, None

    T_WC = _as_se3_pose(frame.T_WC)
    Xw = T_WC.act(X[idx])
    rgb = rgb[idx]
    return _subsample_points(Xw, rgb, max_points=max_points)


@torch.inference_mode()
def spann3r_collect_world_points(
    frames, conf_thresh=0.0, spatial_stride=1, max_points=None
):
    frames = list(frames)
    per_frame_cap = None
    if max_points is not None and max_points > 0 and len(frames) > 0:
        per_frame_cap = max(1, int(np.ceil(float(max_points) / float(len(frames)))))

    world_pts = []
    world_rgb = []
    for frame in frames:
        Xw, rgb = _collect_world_points(
            frame,
            conf_thresh=conf_thresh,
            spatial_stride=spatial_stride,
            max_points=per_frame_cap,
        )
        if Xw is not None:
            world_pts.append(Xw)
            world_rgb.append(rgb)

    if not world_pts:
        return None, None

    Xw = torch.cat(world_pts, dim=0)
    rgb = torch.cat(world_rgb, dim=0)
    return _subsample_points(Xw, rgb, max_points=max_points)


def _zbuffer_rasterize(points_cam, colors, K, h, w):
    if points_cam is None or points_cam.numel() == 0:
        return np.zeros((h, w, 3), dtype=np.float32)

    x, y, z = points_cam.unbind(dim=-1)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = torch.round(fx * (x / z) + cx).long()
    v = torch.round(fy * (y / z) + cy).long()
    valid = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if valid.sum() == 0:
        return np.zeros((h, w, 3), dtype=np.float32)

    u = u[valid].detach().cpu().numpy().astype(np.int64)
    v = v[valid].detach().cpu().numpy().astype(np.int64)
    z = z[valid].detach().cpu().numpy().astype(np.float32)
    c = colors[valid].detach().cpu().numpy().astype(np.float32)

    pix = v * w + u
    # For each pixel keep the nearest point.
    order = np.lexsort((z, pix))
    pix_sorted = pix[order]
    col_sorted = c[order]
    unique_pix, first_idx = np.unique(pix_sorted, return_index=True)

    img = np.zeros((h * w, 3), dtype=np.float32)
    img[unique_pix] = col_sorted[first_idx]
    return img.reshape(h, w, 3)


@torch.inference_mode()
def spann3r_render(
    model,
    frame,
    ref_frame,
    K=None,
    target_T_WC=None,
    spatial_stride=1,
    max_points=None,
):
    _ = model
    if frame is None or frame.X_canon is None:
        return None

    h, w = map(int, frame.img_shape.flatten().tolist())
    device = frame.X_canon.device
    dtype = frame.X_canon.dtype
    K_use = K if K is not None else _default_intrinsics(h, w, device, dtype)
    target_T_WC = target_T_WC if target_T_WC is not None else frame.T_WC
    target_T_WC = _as_se3_pose(target_T_WC)
    target_T_CW = target_T_WC.inv()

    conf_thresh = float(config["tracking"].get("C_conf", 0.0))
    Xw, rgb = spann3r_collect_world_points(
        (frame, ref_frame),
        conf_thresh=conf_thresh,
        spatial_stride=spatial_stride,
        max_points=max_points,
    )
    if Xw is None:
        return None
    Xc = target_T_CW.act(Xw)

    rendered = _zbuffer_rasterize(Xc, rgb, K_use, h, w)
    out = (
        torch.from_numpy(rendered)
        .to(device=device, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    return out.clamp(0.0, 1.0)


@torch.inference_mode()
def spann3r_symmetric_inference(model, frame_i, frame_j):
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model.encode_image(_frame_to_view(frame_i))
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model.encode_image(_frame_to_view(frame_j))

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape2, shape1)

    X, C, D, Q = zip(*[_extract_outputs(r) for r in [res11, res21, res22, res12]])
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


@torch.inference_mode()
def spann3r_decode_symmetric_batch(
    model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
):
    B = feat_i.shape[0]
    X, C, D, Q = [], [], [], []

    for b in range(B):
        feat1 = feat_i[b][None]
        feat2 = feat_j[b][None]
        pos1 = pos_i[b][None]
        pos2 = pos_j[b][None]

        res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])
        res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])

        Xb, Cb, Db, Qb = zip(*[_extract_outputs(r) for r in [res11, res21, res22, res12]])
        X.append(torch.stack(Xb, dim=0))
        C.append(torch.stack(Cb, dim=0))
        D.append(torch.stack(Db, dim=0))
        Q.append(torch.stack(Qb, dim=0))

    X, C, D, Q = (
        torch.stack(X, dim=1),
        torch.stack(C, dim=1),
        torch.stack(D, dim=1),
        torch.stack(Q, dim=1),
    )
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


@torch.inference_mode()
def spann3r_inference_mono(model, frame):
    if frame.feat is None:
        frame.feat, frame.pos, _ = model.encode_image(_frame_to_view(frame))

    feat, pos = frame.feat, frame.pos
    shape = frame.img_true_shape

    res11, res21 = decoder(model, feat, feat, pos, pos, shape, shape)

    # Spann3R does not output the Gaussian parameters used by the old renderer.
    frame.gaussian_pred = None
    frame.gaussian_pred_cross = None

    X, C, D, Q = zip(*[_extract_outputs(r) for r in [res11, res21]])
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)

    Xii, _ = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, _ = einops.rearrange(C, "b h w -> b (h w) 1")

    return Xii, Cii


def spann3r_match_symmetric(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
    X, C, D, Q = spann3r_decode_symmetric_batch(
        model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
    )

    b = X.shape[1]

    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

    X11 = torch.cat((Xii, Xjj), dim=0)
    X21 = torch.cat((Xji, Xij), dim=0)
    D11 = torch.cat((Dii, Djj), dim=0)
    D21 = torch.cat((Dji, Dij), dim=0)

    idx_1_to_2, valid_match_2 = matching.match(X11, X21, D11, D21)

    match_b = X11.shape[0] // 2
    idx_i2j = idx_1_to_2[:match_b]
    idx_j2i = idx_1_to_2[match_b:]
    valid_match_j = valid_match_2[:match_b]
    valid_match_i = valid_match_2[match_b:]

    return (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        Qji.view(b, -1, 1),
        Qij.view(b, -1, 1),
    )


@torch.inference_mode()
def spann3r_asymmetric_inference(model, frame_i, frame_j):
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model.encode_image(_frame_to_view(frame_i))
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model.encode_image(_frame_to_view(frame_j))

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)

    X, C, D, Q = zip(*[_extract_outputs(r) for r in [res11, res21]])
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q, (res11, res21)


def spann3r_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):
    X, C, D, Q, _ = spann3r_asymmetric_inference(model, frame_i, frame_j)

    # Spann3R does not output the Gaussian parameters used by the old renderer.
    frame_i.gaussian_pred = None
    frame_i.gaussian_pred_cross = None

    b, h, w = X.shape[:-1]
    b = b // 2

    Xii, Xji = X[:b], X[b:]
    Dii, Dji = D[:b], D[b:]

    idx_i2j, valid_match_j = matching.match(
        Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
    )

    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
    Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
    Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")

    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    else:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_img(img, size, square_ok=False, return_transformation=False):
    assert size in (224, 512)

    img = PIL.Image.fromarray(np.uint8(img * 255))
    W1, H1 = img.size
    if size == 224:
        img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    else:
        img = _resize_pil_image(img, size)

    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not square_ok and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    res = {
        "img": ImgNorm(img)[None],
        "true_shape": np.int32([img.size[::-1]]),
        "unnormalized_img": np.asarray(img),
    }

    if return_transformation:
        scale_w = W1 / W
        scale_h = H1 / H
        half_crop_w = (W - img.size[0]) / 2
        half_crop_h = (H - img.size[1]) / 2
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)

    return res
