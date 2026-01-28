
import torch
from ..core.engine import run as engine_run, _to_nchw
from ..core.helpers import to_color3

class KeylightCoreHubV3:
    CATEGORY = "KeylightChromaKeyHub"
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "mask", "mask_image", "image_rgba")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "image": ("IMAGE",),
                "key_mode": (["auto","manual"], {"default":"auto"}),
                "key_color": ("COLORCODE", {"default": "#00FF00"}),
                "background_mode": (["alpha","color","soft_color"], {"default":"alpha"}),
                "bg_color": ("COLORCODE", {"default": "#000000"}),
                "tolerance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "clip_black": ("FLOAT", {"default": -0.02, "min": -1.0, "max": 1.0, "step": 0.001}),
                "clip_white": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 2.0, "step": 0.001}),
            },
            "optional": {
                "sampler_args": ("KEY_SAMPLER_ARGS",),
                "edge_args": ("KEY_EDGE_ARGS",),
                "spill_algo_args": ("KEY_SPILL_ALGO_ARGS",),
                "ph_args": ("KEY_PH_ARGS",),
                "mm_args": ("KEY_MM_ARGS",),
            }
        }

    def _ensure_bchw(self, img):
        if img.ndim == 4 and img.shape[1] in (1,3,4):
            return img
        if img.ndim == 4 and img.shape[-1] in (1,3,4):
            return img.permute(0,3,1,2)
        raise ValueError("Unsupported image tensor shape. Expect [N,C,H,W] or [N,H,W,C].")

    def _auto_key_from_border(self, bhwc, frac=0.08):
        # bhwc: [N,H,W,3]
        N,H,W,_ = bhwc.shape
        g = max(1, int(round(min(H,W) * max(0.0, min(0.45, float(frac))))))
        if g*2 >= min(H,W):
            g = max(1, min(H,W)//4)
        top    = bhwc[:, :g, :, :]
        bottom = bhwc[:, -g:, :, :]
        left   = bhwc[:, :, :g, :]
        right  = bhwc[:, :, -g:, :]
        border = torch.cat([top.reshape(N,-1,3), bottom.reshape(N,-1,3),
                            left.reshape(N,-1,3), right.reshape(N,-1,3)], dim=1)
        mean = border.mean(dim=1)  # [N,3]
        return mean.view(N,3,1,1)  # -> [N,3,1,1]

    def _auto_key_from_rect(self, bhwc, rect):
        # rect: [x,y,w,h] in [0..1], center-based
        N,H,W,_ = bhwc.shape
        x,y,w,h = rect
        cx = int(round(float(x) * W))
        cy = int(round(float(y) * H))
        rw = max(1, int(round(float(w) * W)))
        rh = max(1, int(round(float(h) * H)))
        x0 = max(0, cx - rw//2); x1 = min(W, cx + rw//2)
        y0 = max(0, cy - rh//2); y1 = min(H, cy + rh//2)
        patch = bhwc[:, y0:y1, x0:x1, :]
        if patch.numel() == 0:
            patch = bhwc  # fallback to whole image
        mean = patch.reshape(N,-1,3).mean(dim=1)
        return mean.view(N,3,1,1)

    def apply(self, image, key_mode, key_color, background_mode, bg_color,
              tolerance, clip_black, clip_white,
              sampler_args=None, edge_args=None, spill_algo_args=None, ph_args=None, mm_args=None):
        # Normalize image to BCHW [N,3,H,W]
        x_bchw = self._ensure_bchw(image).float().clamp(0,1)
        if x_bchw.shape[1] == 1:
            x_bchw = x_bchw.repeat(1,3,1,1)
        if x_bchw.shape[1] == 4:
            x_bchw = x_bchw[:,:3,:,:]

        # Prepare sampler-based auto key if requested
        bhwc = x_bchw.permute(0,2,3,1).contiguous()
        if str(key_mode) == "auto":
            s = sampler_args or {}
            mode = str(s.get("mode","auto_border"))
            if mode == "auto_border":
                frac = float(s.get("auto_border_frac", 0.08))
                key_rgb = self._auto_key_from_border(bhwc, frac=frac)
            else:
                rect = s.get("rect", [0.45,0.45,0.1,0.1])
                key_rgb = self._auto_key_from_rect(bhwc, rect=rect)
        else:
            # Manual mode uses UI color
            N = x_bchw.shape[0]
            k1 = to_color3(key_color, device=x_bchw.device, dtype=x_bchw.dtype)  # [1,3,1,1]
            key_rgb = k1.repeat(N,1,1,1)  # [N,3,1,1]

        # Background color tensor (single)
        bg_rgb = to_color3(bg_color, device=x_bchw.device, dtype=x_bchw.dtype)

        # Unpack optional args safely
        edge_soft       = float((edge_args or {}).get("edge_soft", 0.0))
        shrink_expand   = float((edge_args or {}).get("shrink_expand", 0.0))
        defringe        = float((edge_args or {}).get("defringe", 0.0))
        spill_cfg       = spill_algo_args or None
        ph_cfg          = ph_args or None
        mm_cfg          = mm_args or None

        # Call engine
        comp, a, mask_img, _ = engine_run(
            x_bchw, key_rgb, float(tolerance), float(clip_black), float(clip_white),
            edge_soft=edge_soft, shrink_expand=shrink_expand, defringe=defringe,
            spill_algo=spill_cfg, ph=ph_cfg, matte_math=mm_cfg,
            background_mode=str(background_mode), bg_color=bg_rgb,
            use_linear=True, verbose=False
        )

        # Build normalized outputs
        comp_bhwc = comp.permute(0,2,3,1).contiguous()
        mask_hw   = a[:,0,:,:] if a.dim()==4 and a.shape[1]==1 else a.squeeze(1)
        if mask_hw.dim()==2:
            mask_hw = mask_hw.unsqueeze(0)
        alpha     = mask_hw.unsqueeze(-1)  # [N,H,W,1]

        # Always provide RGBA
        try:
            image_rgba = torch.cat([comp_bhwc, alpha], dim=-1).clamp(0.0, 1.0)
        except Exception:
            ones = comp_bhwc.new_ones((*comp_bhwc.shape[:3],1))
            image_rgba = torch.cat([comp_bhwc[...,:3], ones], dim=-1)

        # Primary 'image' output auto-switch
        bgm = str(background_mode).lower() if isinstance(background_mode, str) else "alpha"
        image_out = image_rgba if bgm == "alpha" else comp_bhwc

        mask_image_bhwc = mask_img.permute(0,2,3,1).contiguous()
        return (image_out, mask_hw, mask_image_bhwc, image_rgba)
