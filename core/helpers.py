
import torch, re

def to_color3(value, device="cpu", dtype=torch.float32):
    # Handle INT format (0xRRGGBB) from new ComfyUI color picker
    if isinstance(value, int):
        r = ((value >> 16) & 0xFF) / 255.0
        g = ((value >> 8) & 0xFF) / 255.0
        b = (value & 0xFF) / 255.0
        return torch.tensor([r,g,b], dtype=dtype, device=device).view(1,3,1,1)
    
    # Handle list/tuple format [r,g,b]
    if isinstance(value, (list, tuple)) and len(value)==3:
        arr = [float(x) for x in value]
        return torch.tensor(arr, dtype=dtype, device=device).view(1,3,1,1)
    
    # Handle string formats
    if isinstance(value, str):
        s=value.strip()
        # Hex string format: #RRGGBB or #RGB
        if s.startswith("#") and (len(s)==7 or len(s)==4):
            if len(s)==7:
                r=int(s[1:3],16)/255.0; g=int(s[3:5],16)/255.0; b=int(s[5:7],16)/255.0
            else:
                r=int(s[1]*2,16)/255.0; g=int(s[2]*2,16)/255.0; b=int(s[3]*2,16)/255.0
            return torch.tensor([r,g,b], dtype=dtype, device=device).view(1,3,1,1)
        # Comma-separated format: "r,g,b"
        m=re.match(r"^\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*$", s)
        if m:
            r,g,b = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            if max(r,g,b)>1.0: r,g,b = r/255.0, g/255.0, b/255.0
            return torch.tensor([r,g,b], dtype=dtype, device=device).view(1,3,1,1)
    
    # Default: green color
    return torch.tensor([0.0,1.0,0.0], dtype=dtype, device=device).view(1,3,1,1)
