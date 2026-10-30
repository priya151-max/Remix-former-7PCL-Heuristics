"""
RemixFormer++ architecture.

EXTRACTED VERBATIM from notebook `7pcl-heuristic3 (1).ipynb` (cells 9-23),
which is the notebook that produced the deployed end-to-end checkpoint
(format_version = remixformerpp_7pcl_GDDW_5class_graph_integrated_heuristic_v3).

The only deviation from the notebook source is documented in
`LOAD_IMAGENET_PRETRAINED` below. No layer, dimension, or forward-pass
behaviour has been altered.
"""
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch.utils.checkpoint

from .config import (
    CLINICAL_SIZE, GLOBAL_CLINICAL_SIZE, LOCAL_PATCH_SIZE,
    DERMOSCOPY_SIZE, DERM_WINDOW_SIZE, NUM_DERM_WINDOWS, LSM_K,
    SWIN_EMBED_DIM, SWIN_DEPTHS, SWIN_HEADS, SWIN_WINDOW, SWIN_MLP_RATIO,
    VIT_DIM, VIT_DEPTH, VIT_HEADS, MSTA_TEMPLATES,
    NUM_DIAG_CLASSES, LOAD_IMAGENET_PRETRAINED,
    USE_GRAD_CHECKPOINTING, DERM_WINDOW_CHUNK, LOCAL_CROP_CHUNK,
    LSM_TRAIN_TEMPERATURE,
)

__all__ = ["RemixFormerPP"]

# 6. Window helpers

def window_partition_padded(x, window_size):
    # x: [B,H,W,C]
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    x_pad = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    windows = x_pad.view(
        B,
        Hp // window_size,
        window_size,
        Wp // window_size,
        window_size,
        C
    )
    windows = windows.permute(
        0, 1, 3, 2, 4, 5
    ).contiguous().view(
        -1, window_size * window_size, C
    )
    return windows, Hp, Wp, pad_h, pad_w

def window_reverse_padded(windows, window_size, B, H, W):
    Hp = math.ceil(H / window_size) * window_size
    Wp = math.ceil(W / window_size) * window_size

    x = windows.view(
        B,
        Hp // window_size,
        Wp // window_size,
        window_size,
        window_size,
        -1
    )
    x = x.permute(
        0, 1, 3, 2, 4, 5
    ).contiguous().view(B, Hp, Wp, -1)
    return x[:, :H, :W, :]

def build_shift_mask(H, W, window_size, shift_size, device):
    if shift_size == 0:
        return None

    Hp = math.ceil(H / window_size) * window_size
    Wp = math.ceil(W / window_size) * window_size

    img_mask = torch.zeros((1, Hp, Wp, 1), device=device)

    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows, _, _, _, _ = window_partition_padded(
        img_mask, window_size
    )
    mask_windows = mask_windows.view(
        -1,
        window_size * window_size
    )

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
    attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
    return attn_mask
# 7. Relative-position CLS window attention

class CLSWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        relative_size = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_size, num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )
        coords_flatten = torch.flatten(coords, 1)
        relative = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative.sum(-1)
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False
        )

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _relative_bias(self, n_tokens):
        if n_tokens == self.window_size * self.window_size:
            bias = self.relative_position_bias_table[
                self.relative_position_index.reshape(-1)
            ]
            bias = bias.view(
                self.window_size * self.window_size,
                self.window_size * self.window_size,
                -1
            ).permute(2, 0, 1).contiguous()
            return bias

        # Defensive interpolation for unusual window sizes.
        base = self.relative_position_bias_table.T.view(
            self.num_heads,
            2 * self.window_size - 1,
            2 * self.window_size - 1
        )
        target = int(math.sqrt(n_tokens))
        base = F.interpolate(
            base.unsqueeze(0),
            size=(2 * target - 1, 2 * target - 1),
            mode="bicubic",
            align_corners=False
        ).squeeze(0)
        coords = torch.arange(target, device=base.device)
        coords = torch.stack(
            torch.meshgrid(coords, coords, indexing="ij")
        )
        flat = coords.flatten(1)
        rel = flat[:, :, None] - flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += target - 1
        rel[:, :, 1] += target - 1
        rel[:, :, 0] *= 2 * target - 1
        idx = rel.sum(-1).long()
        return base[:, idx.reshape(-1)].view(
            self.num_heads, target * target, target * target
        )

    def forward(self, patch_windows, cls_token, attn_mask=None):
        # patch_windows: [B*nW, N, C]
        # cls_token: [B, C]
        BNW, N, C = patch_windows.shape

        # Number of windows for this batch.
        B = cls_token.shape[0]
        nW = BNW // B

        cls_rep = cls_token[:, None, :].expand(
            B, nW, C
        ).reshape(BNW, 1, C)

        tokens = torch.cat([cls_rep, patch_windows], dim=1)
        qkv = self.qkv(tokens)
        qkv = qkv.reshape(
            BNW,
            N + 1,
            3,
            self.num_heads,
            C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        logits = q @ k.transpose(-2, -1)

        rel = self._relative_bias(N)
        # Relative bias only applies to patch-patch pairs.
        logits[:, :, 1:, 1:] = logits[:, :, 1:, 1:] + rel.unsqueeze(0)

        if attn_mask is not None:
            logits_patch = logits[:, :, 1:, 1:]
            nW = attn_mask.shape[0]
            logits_patch = logits_patch.view(
                B, nW, self.num_heads, N, N
            )
            logits_patch = logits_patch + attn_mask.unsqueeze(0).unsqueeze(2)
            logits[:, :, 1:, 1:] = logits_patch.view(
                BNW, self.num_heads, N, N
            )

        attn = logits.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(BNW, N + 1, C)
        out = self.proj(out)

        cls_out = out[:, 0]
        patch_out = out[:, 1:]

        # Paper-style LSM score:
        # softmax(Q_cls K_patch^T) element-wise multiplied by
        # softmax((Q_patch K_cls^T)^T), averaged over heads.
        q_cls = q[:, :, 0:1, :]
        k_patch = k[:, :, 1:, :]
        q_patch = q[:, :, 1:, :]
        k_cls = k[:, :, 0:1, :]

        o1 = (q_cls @ k_patch.transpose(-2, -1)).squeeze(2)
        o2 = (q_patch @ k_cls.transpose(-2, -1)).squeeze(-1)

        s1 = F.softmax(o1, dim=-1)
        s2 = F.softmax(o2, dim=-1)
        score = (s1 * s2).mean(dim=1)

        # Keep the LSM score in window form here:
        # [B*nW, N]. The CLS-Swin block converts it back to the
        # spatial [B,H,W] feature-map layout after reversing windows.
        score = score.view(B, nW, N)

        # Aggregate repeated CLS outputs across windows.
        cls_out = cls_out.view(B, nW, C).mean(dim=1)

        if not self.training:
            score = score.detach()

        return patch_out, cls_out, score
# 8. CLS-Swin blocks, patch merging and pretrained-weight transfer

class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)

        return self.reduction(self.norm(x))


class CLSSwinBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0
    ):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = CLSWindowAttention(
            dim,
            heads,
            window_size=window_size
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x, cls_token):
        B, H, W, C = x.shape

        shortcut = x
        x_norm = self.norm1(x)

        if self.shift_size > 0:
            shifted = torch.roll(
                x_norm,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2)
            )
        else:
            shifted = x_norm

        windows, Hp, Wp, pad_h, pad_w = window_partition_padded(
            shifted,
            self.window_size
        )

        attn_mask = build_shift_mask(
            H,
            W,
            self.window_size,
            self.shift_size,
            x.device
        )

        patch_out, cls_out, score_windows = self.attn(
            windows,
            cls_token,
            attn_mask=attn_mask
        )

        x = window_reverse_padded(
            patch_out,
            self.window_size,
            B,
            H,
            W
        )

        # Reconstruct the LSM attention score as a spatial map.
        # [B,nW,N] -> [B,Hp,Wp,1] -> [B,H,W].
        score_map = window_reverse_padded(
            score_windows.reshape(
                -1,
                self.window_size * self.window_size,
                1
            ),
            self.window_size,
            B,
            H,
            W
        ).squeeze(-1)

        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )
            score_map = torch.roll(
                score_map,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x, cls_out, score_map


class CLSSwinStage(nn.Module):
    """
    One Swin stage.

    Important fix:
    The CLS token must always have the SAME feature width as the patch
    tokens entering CLSWindowAttention.

    Original v23 kept the CLS token at 96 dimensions after stage 1.
    Patch merging changed the patch width to 192, producing:
        target [B, nW, 192]
        source [B, 1, 96]

    Here, when a stage downsamples from C -> 2C, the CLS token is
    explicitly projected with cls_downsample: C -> 2C.
    """

    def __init__(
        self,
        dim,
        depth,
        heads,
        window_size=7,
        downsample=True,
        next_dim=None
    ):
        super().__init__()

        self.dim = dim

        self.blocks = nn.ModuleList([
            CLSSwinBlock(
                dim=dim,
                heads=heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=SWIN_MLP_RATIO
            )
            for i in range(depth)
        ])

        self.downsample = (
            PatchMerging(dim)
            if downsample
            else None
        )

        # PatchMerging changes C -> 2C. The CLS token must follow it.
        if self.downsample is not None:
            target_dim = 2 * dim if next_dim is None else next_dim
            if target_dim != 2 * dim:
                raise ValueError(
                    f"PatchMerging expects next dim {2 * dim}, "
                    f"but received {target_dim}."
                )

            self.cls_downsample = nn.Linear(
                dim,
                target_dim,
                bias=True
            )
        else:
            self.cls_downsample = None

    def forward(self, x, cls_token):
        last_score = None

        for block in self.blocks:
            if USE_GRAD_CHECKPOINTING and self.training:

                # Bind the current block as a default argument so that
                # checkpointing cannot accidentally close over a later block.
                def block_fn(x_, c_, block_=block):
                    return block_(x_, c_)

                x, cls_token, last_score = torch.utils.checkpoint.checkpoint(
                    block_fn,
                    x,
                    cls_token,
                    use_reentrant=False
                )
            else:
                x, cls_token, last_score = block(
                    x,
                    cls_token
                )

        if self.downsample is not None:
            x = self.downsample(x)

            # Critical CLS-dimension transition:
            # 96 -> 192 -> 384 -> 768.
            cls_token = self.cls_downsample(cls_token)

        return x, cls_token, last_score


class PatchEmbed(nn.Module):
    # Named to match timm Swin-T keys: patch_embed.proj and patch_embed.norm.
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=4,
            stride=4
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return self.norm(x)


class CLSSwinBackbone(nn.Module):
    def __init__(self, pretrained=True, share_stages=None):
        super().__init__()

        self.embed_dim = SWIN_EMBED_DIM
        self.dims = [
            SWIN_EMBED_DIM,
            SWIN_EMBED_DIM * 2,
            SWIN_EMBED_DIM * 4,
            SWIN_EMBED_DIM * 8,
        ]

        if share_stages is None:
            self.patch_embed = PatchEmbed(
                in_chans=3,
                embed_dim=SWIN_EMBED_DIM
            )

            # CLS starts at the same width as stage 1.
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, SWIN_EMBED_DIM)
            )

            self.layers = nn.ModuleList([
                CLSSwinStage(
                    self.dims[i],
                    SWIN_DEPTHS[i],
                    SWIN_HEADS[i],
                    window_size=SWIN_WINDOW,
                    downsample=(i < 3),
                    next_dim=(
                        self.dims[i + 1]
                        if i < 3
                        else None
                    )
                )
                for i in range(4)
            ])

            self.norm = nn.LayerNorm(self.dims[-1])

            nn.init.trunc_normal_(
                self.cls_token,
                std=0.02
            )

            if pretrained:
                self._load_pretrained_swin()

        else:
            # Local encoder:
            # shared patch embedding + first 3 stages,
            # independent fourth stage.
            self.patch_embed = share_stages.patch_embed
            self.cls_token = share_stages.cls_token

            self.layers = nn.ModuleList([
                share_stages.layers[0],
                share_stages.layers[1],
                share_stages.layers[2],
                CLSSwinStage(
                    self.dims[3],
                    SWIN_DEPTHS[3],
                    SWIN_HEADS[3],
                    window_size=SWIN_WINDOW,
                    downsample=False
                )
            ])

            self.norm = nn.LayerNorm(self.dims[-1])

    def _patch_embed(self, x):
        return self.patch_embed(x)

    def _load_pretrained_swin(self):
        print(
            "Loading ImageNet Swin-T weights where parameter "
            "shapes match..."
        )

        ref = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )

        ref_sd = ref.state_dict()
        own_sd = self.state_dict()

        copied = {}

        for k, v in ref_sd.items():
            if (
                k in own_sd
                and own_sd[k].shape == v.shape
            ):
                copied[k] = v

        missing = sorted(
            set(own_sd) - set(copied)
        )

        self.load_state_dict(
            copied,
            strict=False
        )

        print(
            f"Copied {len(copied)} pretrained tensors; "
            f"{len(missing)} new/modified tensors remain."
        )

        del ref

    def forward(self, x, return_sequence=True):
        x = self._patch_embed(x)
        B = x.shape[0]

        cls = self.cls_token.expand(
            B,
            -1,
            -1
        ).squeeze(1)

        last_score = None

        for stage in self.layers:
            x, cls, score = stage(
                x,
                cls
            )

            if score is not None:
                last_score = score

        x = self.norm(x)
        cls = self.norm(cls)

        patches = x.flatten(1, 2)

        # Final invariant:
        # CLS width == final patch width == 768.
        if cls.shape[-1] != patches.shape[-1]:
            raise RuntimeError(
                "CLS/patch dimension mismatch after Swin stages: "
                f"CLS={cls.shape[-1]}, patches={patches.shape[-1]}"
            )

        return {
            "cls": cls,
            "patches": patches,
            "map": x,
            "lsm_score": last_score
        }

# 9. Corrected paper-aligned Lesion Selection Module

class LesionSelectionModule(nn.Module):
    """
    RemixFormer++ LSM.

    Paper behavior:
      1. obtain S_attn from final-stage CLS/patch Q-K interactions;
      2. Kp=4 largest attention values are used for lesion patches;
      3. attention is computed on the downsampled global feature map;
      4. the selected areas are enlarged to 224x224 on the original 896x896 image;
      5. during training, attention sampling is used to avoid a hard top-K gradient break.

    The paper does not publish the authors' exact sampler source code. We therefore use a
    straight-through Gumbel reparameterization as the differentiable Monte-Carlo approximation,
    while keeping the paper's categorical attention-distribution sampling interpretation.
    In evaluation there is no approximation: top-K over S_attn is used.
    """
    def __init__(self, k=4, crop_size=224):
        super().__init__()
        self.k = k
        self.crop_size = crop_size

    def _training_attention_sample(self, score):
        B, Hs, Ws = score.shape
        logits = score.flatten(1)
        probs = F.softmax(logits, dim=-1)

        hard_vectors = []
        indices = []

        for _ in range(self.k):
            one_hot = F.gumbel_softmax(
                logits,
                tau=LSM_TRAIN_TEMPERATURE,
                hard=True,
                dim=-1
            )
            hard_vectors.append(one_hot)
            indices.append(one_hot.argmax(dim=-1))

        sample_matrix = torch.stack(hard_vectors, dim=1)
        idx = torch.stack(indices, dim=1)
        return idx, probs, sample_matrix

    def _inference_topk(self, score):
        B, Hs, Ws = score.shape
        probs = F.softmax(score.flatten(1), dim=-1)
        k = min(self.k, Hs * Ws)
        idx = torch.topk(probs, k=k, dim=-1).indices
        sample_matrix = F.one_hot(
            idx, num_classes=Hs * Ws
        ).to(probs.dtype)
        return idx, probs, sample_matrix

    def _crop_grid(self, centers_y, centers_x, H, W, device):
        B, K = centers_y.shape
        half = self.crop_size / 2.0

        yy = torch.linspace(
            -half + 0.5, half - 0.5, self.crop_size, device=device
        )
        xx = torch.linspace(
            -half + 0.5, half - 0.5, self.crop_size, device=device
        )
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")

        gy = gy[None, None] + centers_y[:, :, None, None]
        gx = gx[None, None] + centers_x[:, :, None, None]

        gy = gy / max(H - 1, 1) * 2 - 1
        gx = gx / max(W - 1, 1) * 2 - 1

        return torch.stack([gx, gy], dim=-1)

    def forward(self, image, score):
        B, C, H, W = image.shape
        B2, Hs, Ws = score.shape
        if B != B2:
            raise RuntimeError("LSM batch-size mismatch.")

        if self.training:
            idx, probs, sample_matrix = self._training_attention_sample(score)
        else:
            idx, probs, sample_matrix = self._inference_topk(score)

        # The final global Swin map is the map on which S_attn is computed.
        # Each token corresponds to H/Hs by W/Ws pixels in the original image.
        scale_y = H / float(Hs)
        scale_x = W / float(Ws)

        token_y = (
            torch.arange(Hs, device=image.device, dtype=torch.float32) + 0.5
        ) * scale_y
        token_x = (
            torch.arange(Ws, device=image.device, dtype=torch.float32) + 0.5
        ) * scale_x

        yy, xx = torch.meshgrid(token_y, token_x, indexing="ij")

        centers_y = torch.einsum(
            "bkn,n->bk", sample_matrix, yy.reshape(-1)
        )
        centers_x = torch.einsum(
            "bkn,n->bk", sample_matrix, xx.reshape(-1)
        )

        grid = self._crop_grid(
            centers_y, centers_x, H, W, image.device
        )

        crops = []
        K = grid.shape[1]

        for start in range(0, K, LOCAL_CROP_CHUNK):
            g = grid[:, start:start + LOCAL_CROP_CHUNK]
            b, kk = g.shape[:2]
            g = g.reshape(
                b * kk, self.crop_size, self.crop_size, 2
            )

            img_rep = image[:, None].expand(
                B, kk, C, H, W
            ).reshape(B * kk, C, H, W)

            crop = F.grid_sample(
                img_rep,
                g,
                mode="bilinear",
                padding_mode="border",
                align_corners=True
            )
            crops.append(crop)

        crops = torch.cat(crops, dim=0).view(
            B, K, C, self.crop_size, self.crop_size
        )

        # Keep exact centers for a transparent audit.
        centers = torch.stack([centers_x, centers_y], dim=-1)

        return crops, idx, probs, centers

# 10. Clinical branch with CLS_WMSA, LSM K=4 and CSF attention capture
class CrossScaleFusion(nn.Module):
    def __init__(self, dim=768, heads=8):
        super().__init__()
        self.norm_g = nn.LayerNorm(dim)
        self.norm_l = nn.LayerNorm(dim)
        self.g_from_l = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.l_from_g = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.g_proj = nn.Linear(dim, dim)
        self.l_proj = nn.Linear(dim, dim)

    def forward(self, global_seq, local_seq, capture=False):
        g = self.norm_g(global_seq)
        l = self.norm_l(local_seq)
        g_cross, g_attn = self.g_from_l(g,l,l,need_weights=capture,average_attn_weights=False)
        l_cross, l_attn = self.l_from_g(l,g,g,need_weights=capture,average_attn_weights=False)
        g_out = g + self.g_proj(g_cross)
        l_out = l + self.l_proj(l_cross)
        if capture:
            return g_out,l_out,g_attn,l_attn
        return g_out,l_out

class ClinicalBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # DEPLOYMENT DEVIATION (documented): the notebook passes pretrained=True here.
        # At inference the ImageNet weights are fully overwritten by
        # upstream_remix_state_dict, so downloading them is unnecessary. Set
        # LOAD_IMAGENET_PRETRAINED=True in src/config.py to restore notebook behaviour.
        self.global_encoder = CLSSwinBackbone(pretrained=LOAD_IMAGENET_PRETRAINED)
        self.local_encoder = CLSSwinBackbone(pretrained=False, share_stages=self.global_encoder)
        self.lsm = LesionSelectionModule(k=LSM_K, crop_size=LOCAL_PATCH_SIZE)
        self.csf = CrossScaleFusion(dim=768, heads=8)
        self.last_lsm_score = None
        self.last_crop_indices = None
        self.last_crops = None
        self.last_crop_centers = None

    def forward(self,x,capture=False):
        x_global = F.interpolate(x,size=(GLOBAL_CLINICAL_SIZE,GLOBAL_CLINICAL_SIZE),
                                 mode="bicubic",align_corners=False)
        global_out = self.global_encoder(x_global)
        score = global_out["lsm_score"]
        if score is None:
            raise RuntimeError("CLS_WMSA did not return an LSM attention score.")

        crops,crop_idx,probs,crop_centers = self.lsm(x,score)
        self.last_lsm_score = score.detach()
        self.last_crop_indices = crop_idx.detach()
        self.last_crops = crops.detach()
        self.last_crop_centers = crop_centers.detach()

        B,K,C,H,W = crops.shape
        local_out = self.local_encoder(crops.reshape(B*K,C,H,W))
        local_cls = local_out["cls"].view(B,K,-1)
        local_patches = local_out["patches"].view(B,K,-1,768)
        local_seq = torch.cat([local_cls.unsqueeze(2),local_patches],dim=2).reshape(B,-1,768)
        global_seq = torch.cat([global_out["cls"].unsqueeze(1),global_out["patches"]],dim=1)

        if capture:
            g_fused,l_fused,g_attn,l_attn = self.csf(global_seq,local_seq,capture=True)
        else:
            g_fused,l_fused = self.csf(global_seq,local_seq,capture=False)
            g_attn=l_attn=None

        return {
            "global":g_fused[:,0],
            "global_tokens":g_fused,
            "local":l_fused,
            "local_tokens":l_fused,
            "lsm_score":score,
            "crop_indices":crop_idx,
            "crop_probs":probs,
            "crop_centers":crop_centers,
            "csf_global_to_local_attention":g_attn,
            "csf_local_to_global_attention":l_attn,
        }

# 11. MSTA with explicit texture attention capture

class MSTA(nn.Module):
    """
    Paper equation (15)-(17):
      a_jk = y_i,j dot t_k^T
      c_jk = exp(-s_k a_jk) / sum_l exp(-s_l a_jl)
      E_i,k^TE = sum_j c_jk y_i,j
      E_i^HRWE = E_i^WE + E_i^TE
    """
    def __init__(self, dim=384, num_templates=32):
        super().__init__()
        self.dim = dim
        self.num_templates = num_templates
        self.templates = nn.Parameter(torch.randn(num_templates, dim) * 0.02)
        self.scale = nn.Parameter(torch.ones(num_templates))
        self.proj = nn.Linear(num_templates * dim, dim)

    def forward(self, y, return_attention=False):
        # y: [B,W,N,D]
        # The paper uses direct dot-product similarity, not cosine-normalized similarity.
        a = torch.einsum("bwnd,kd->bwnk", y, self.templates)

        scale = self.scale.view(1, 1, 1, self.num_templates)
        c = F.softmax(-scale * a, dim=-1)

        texture = torch.einsum("bwnk,bwnd->bwkd", c, y)
        texture = texture.reshape(
            y.shape[0], y.shape[1], self.num_templates * self.dim
        )
        texture = self.proj(texture)

        if return_attention:
            return texture, a, c
        return texture

# 12. Dermoscopy branch: exact dimensions + explicit MSTA/region attention capture
class RegionTransformerBlock(nn.Module):
    def __init__(self, dim=384, heads=6, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )

    def forward(self, x, return_attention=False):
        q = self.norm1(x)
        a, w = self.attn(q, q, q, need_weights=return_attention, average_attn_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, w

class DermoscopyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = VIT_DIM
        self.num_heads = VIT_HEADS

        self.hrwe = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool="",
            img_size=DERM_WINDOW_SIZE,
            dynamic_img_size=True,
        )
        assert self.hrwe.num_features == VIT_DIM

        self.msta = MSTA(VIT_DIM, MSTA_TEMPLATES)

        self.region_cls = nn.Parameter(torch.zeros(1,1,VIT_DIM))
        self.region_pos = nn.Parameter(torch.zeros(1, NUM_DERM_WINDOWS+1, VIT_DIM))
        self.region_blocks = nn.ModuleList([
            RegionTransformerBlock(VIT_DIM, VIT_HEADS)
            for _ in range(VIT_DEPTH)
        ])
        self.region_norm = nn.LayerNorm(VIT_DIM)

        self.last = {}

    def split_windows(self, x):
        B,C,H,W = x.shape
        if (H,W) != (DERMOSCOPY_SIZE, DERMOSCOPY_SIZE):
            raise RuntimeError(f"Dermoscopy input must be 1024x1024, got {H}x{W}")
        z = x.unfold(2, DERM_WINDOW_SIZE, DERM_WINDOW_SIZE).unfold(3, DERM_WINDOW_SIZE, DERM_WINDOW_SIZE)
        z = z.permute(0,2,3,1,4,5).contiguous()
        return z.reshape(B, NUM_DERM_WINDOWS, C, DERM_WINDOW_SIZE, DERM_WINDOW_SIZE)

    def forward(self, x, capture=False):
        windows = self.split_windows(x)
        B, W, C, H, Wd = windows.shape
        cls_list, patch_list = [], []

        for start in range(0, W, DERM_WINDOW_CHUNK):
            chunk = windows[:, start:start+DERM_WINDOW_CHUNK]
            flat = chunk.reshape(B*chunk.shape[1], C, H, Wd)
            tokens = self.hrwe.forward_features(flat)
            if isinstance(tokens, tuple):
                tokens = tokens[0]
            if tokens.ndim != 3:
                raise RuntimeError(f"HRWE returned {tuple(tokens.shape)}")
            cls_list.append(tokens[:,0].view(B,chunk.shape[1],VIT_DIM))
            patch_list.append(tokens[:,1:].view(B,chunk.shape[1],tokens.shape[1]-1,VIT_DIM))

        window_cls = torch.cat(cls_list, dim=1)
        patch_tokens = torch.cat(patch_list, dim=1)

        texture, msta_similarity, msta_weights = self.msta(
            patch_tokens, return_attention=True
        )
        hrwe_windows = window_cls + texture
        hrwe_windows = F.layer_norm(hrwe_windows, (VIT_DIM,))

        cls = self.region_cls.expand(B,-1,-1)
        seq = torch.cat([cls, hrwe_windows], dim=1) + self.region_pos

        region_attn = []
        for block in self.region_blocks:
            seq, attn = block(seq, return_attention=capture)
            if capture:
                region_attn.append(attn.detach())

        seq = self.region_norm(seq)
        g_d = seq[:,0]
        l_d = seq[:,1:]

        if capture:
            self.last = {
                "window_cls": window_cls.detach(),
                "patch_tokens": patch_tokens.detach(),
                "msta_similarity": msta_similarity.detach(),
                "msta_weights": msta_weights.detach(),
                "window_embeddings": hrwe_windows.detach(),
                "region_tokens": seq.detach(),
                "region_attention": region_attn,
            }

        return {
            "global": g_d,
            "local": l_d,
            "window_embeddings": hrwe_windows,
            "msta_similarity": msta_similarity if capture else None,
            "msta_weights": msta_weights if capture else None,
            "region_tokens": seq,
            "region_attention": region_attn if capture else None,
        }

print("Dermoscopy branch ready: 1024 -> 16x256 -> ViT-S/16 -> MSTA-32 -> 17-token x 12-layer region encoder.")

# 13. Metadata branch

class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        if self.training and x.shape[0] == 1:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps
            )
        return super().forward(x)

class MetadataBranch(nn.Module):
    def __init__(self, input_dim, output_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            SafeBatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)
# 14. Cyclic CMF with explicit attention capture, dimension-safe
class CyclicCMF(nn.Module):
    def __init__(self, fusion_dim=768, heads=8):
        super().__init__()
        self.dim = fusion_dim
        self.heads = heads

        # Dermoscopy branch is 384-D in the paper-aligned implementation.
        # CMF operates at 768-D, so both dermoscopy global and local features
        # are explicitly projected before LayerNorm/MHA.
        self.d_global_to_f = nn.Linear(384, fusion_dim)
        self.d_local_to_f = nn.Linear(384, fusion_dim)

        self.norm_q = nn.LayerNorm(fusion_dim)
        self.norm_kv = nn.LayerNorm(fusion_dim)

        self.attn_cd = nn.MultiheadAttention(fusion_dim, heads, batch_first=True)
        self.attn_dc = nn.MultiheadAttention(fusion_dim, heads, batch_first=True)
        self.attn_mc = nn.MultiheadAttention(fusion_dim, heads, batch_first=True)
        self.attn_md = nn.MultiheadAttention(fusion_dim, heads, batch_first=True)

        self.out_cd = nn.Linear(fusion_dim, fusion_dim)
        self.out_dc = nn.Linear(fusion_dim, fusion_dim)
        self.out_mc = nn.Linear(fusion_dim, fusion_dim)
        self.out_md = nn.Linear(fusion_dim, fusion_dim)

    def update(self, query, source_local, attn, out_proj, capture=False):
        # Both tensors entering LayerNorm are guaranteed to be [B, N, 768].
        if query.ndim != 3 or query.shape[-1] != self.dim:
            raise RuntimeError(
                f"CMF query must be [B,N,{self.dim}], got {tuple(query.shape)}"
            )
        if source_local.ndim != 3 or source_local.shape[-1] != self.dim:
            raise RuntimeError(
                f"CMF source must be [B,N,{self.dim}], got {tuple(source_local.shape)}"
            )

        q = self.norm_q(query)
        kv = self.norm_kv(source_local)

        cross, weights = attn(
            q, kv, kv,
            need_weights=capture,
            average_attn_weights=False
        )
        out = query + out_proj(cross)
        return out, weights

    def forward(self, clinical, dermoscopy, metadata, capture=False):
        g_c = clinical["global"].unsqueeze(1)                  # [B,1,768]
        l_c = clinical["local_tokens"]                         # [B,Nc,768]

        g_d = self.d_global_to_f(
            dermoscopy["global"]
        ).unsqueeze(1)                                         # [B,1,768]

        l_d_raw = dermoscopy["local"]
        if l_d_raw.ndim == 2:
            l_d_raw = l_d_raw.unsqueeze(1)
        l_d = self.d_local_to_f(l_d_raw)                       # [B,Nd,768]

        g_m = metadata.unsqueeze(1)                            # [B,1,768]

        # Explicit dimension audit before any LayerNorm.
        for name, tensor in {
            "clinical_global": g_c,
            "clinical_local": l_c,
            "dermoscopy_global_projected": g_d,
            "dermoscopy_local_projected": l_d,
            "metadata": g_m,
        }.items():
            assert tensor.ndim == 3 and tensor.shape[-1] == 768, (
                f"CMF dimension error for {name}: {tuple(tensor.shape)}"
            )

        g_c_fused, a_cd = self.update(
            g_c, l_d, self.attn_cd, self.out_cd, capture
        )
        g_d_fused, a_dc = self.update(
            g_d, l_c, self.attn_dc, self.out_dc, capture
        )
        g_m_from_c, a_mc = self.update(
            g_m, l_c, self.attn_mc, self.out_mc, capture
        )
        g_m_from_d, a_md = self.update(
            g_m, l_d, self.attn_md, self.out_md, capture
        )

        g_m_fused = 0.5 * (g_m_from_c + g_m_from_d)

        out = {
            "global_clinical": g_c_fused.squeeze(1),
            "local_clinical": l_c.mean(dim=1),
            "global_dermoscopy": g_d_fused.squeeze(1),
            "local_dermoscopy": l_d.mean(dim=1),
            "metadata": g_m_fused.squeeze(1),
        }

        if capture:
            out["attention"] = {
                "clinical_from_dermoscopy": a_cd.detach() if a_cd is not None else None,
                "dermoscopy_from_clinical": a_dc.detach() if a_dc is not None else None,
                "metadata_from_clinical": a_mc.detach() if a_mc is not None else None,
                "metadata_from_dermoscopy": a_md.detach() if a_md is not None else None,
            }

        return out

# 15. Complete RemixFormer++ model
class RemixFormerPP(nn.Module):
    def __init__(self, meta_dim, checklist_specs):
        super().__init__()
        self.clinical = ClinicalBranch()
        self.dermoscopy = DermoscopyBranch()
        self.metadata = MetadataBranch(input_dim=meta_dim, output_dim=768)
        self.cmf = CyclicCMF(fusion_dim=768, heads=8)

        self.classifier = nn.Sequential(
            nn.LayerNorm(768 * 5),
            nn.Linear(768 * 5, 1024),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(1024, NUM_DIAG_CLASSES)
        )

        self.checklist_heads = nn.ModuleDict({
            name: nn.Linear(768*5, spec["num_classes"])
            for name, spec in checklist_specs.items()
        })

    def forward(self, clinical, dermoscopy, metadata, capture=False):
        c = self.clinical(clinical)
        d = self.dermoscopy(dermoscopy, capture=capture)
        m = self.metadata(metadata)
        fused = self.cmf(c, d, m, capture=capture)

        final_feature = torch.cat([
            fused["global_clinical"],
            fused["local_clinical"],
            fused["global_dermoscopy"],
            fused["local_dermoscopy"],
            fused["metadata"],
        ], dim=1)

        diagnosis = self.classifier(final_feature)
        checklist = {name: head(final_feature) for name, head in self.checklist_heads.items()}

        return {
            "diagnosis": diagnosis,
            "checklist": checklist,
            "clinical": c,
            "dermoscopy": d,
            "metadata": m,
            "fused": fused,
            "feature": final_feature
        }
