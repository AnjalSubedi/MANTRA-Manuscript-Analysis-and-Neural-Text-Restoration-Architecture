import cv2
import numpy as np
from PIL import Image

def to_cv2(pil_img):
    """Converts a PIL image to a grayscale OpenCV numpy array."""
    gray_img = pil_img.convert("L")
    return np.array(gray_img)

def to_pil(cv2_img):
    """Converts a grayscale OpenCV numpy array to a PIL Image."""
    return Image.fromarray(cv2_img).convert("RGB")

def estimate_readability_and_continuity(cv_img, orig_img):
    """
    Estimates the readability score of the degraded image and checks foreground continuity.
    Returns: readability_score, continuity_ratio (fraction of foreground pixels preserved)
    """
    std_dev = float(np.std(cv_img))
    fg_mask = cv_img < 160
    bg_mask = cv_img >= 160
    mean_fg = float(np.mean(cv_img[fg_mask])) if np.any(fg_mask) else 0.0
    mean_bg = float(np.mean(cv_img[bg_mask])) if np.any(bg_mask) else 255.0
    contrast = mean_bg - mean_fg
    contrast_score = min(80.0, (contrast / 150.0) * 80.0)
    std_score = min(20.0, (std_dev / 50.0) * 20.0)
    readability_score = float(np.clip(contrast_score + std_score, 0.0, 100.0))
    orig_fg_count = np.sum(orig_img < 160)
    degraded_fg_count = np.sum(fg_mask)
    if orig_fg_count > 0:
        continuity_ratio = degraded_fg_count / orig_fg_count
    else:
        continuity_ratio = 1.0
    return readability_score, continuity_ratio

# --- Composable Modules ---

# 1. Substrate Effects
def apply_uneven_illumination(img, seed=None):
    """Simulates uneven lighting using a linear spatial gradient."""
    if seed is not None:
        np.random.seed(seed)
    h, w = img.shape
    intensity_diff = np.random.uniform(0.15, 0.35)
    direction = np.random.uniform(0, 2 * np.pi)
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    proj = x * np.cos(direction) + y * np.sin(direction)
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-5)
    scale = 1.0 - intensity_diff * proj
    out = (img.astype(float) * scale)
    params = {
        "illum_intensity_diff": float(intensity_diff),
        "illum_direction_rad": float(direction)
    }
    return np.clip(out, 0, 255).astype(np.uint8), params

def apply_paper_texture(img, seed=None):
    """Procedural low-frequency noise to simulate paper texture."""
    if seed is not None:
        np.random.seed(seed)
    h, w = img.shape
    noise_h = max(2, h // 4)
    noise_w = max(2, w // 4)
    noise = np.random.normal(128, 15, (noise_h, noise_w)).astype(np.uint8)
    noise_resized = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    noise_blurred = cv2.GaussianBlur(noise_resized, (15, 15), 0)
    n_min, n_max = noise_blurred.min(), noise_blurred.max()
    noise_norm = 0.88 + 0.12 * (noise_blurred - n_min) / (n_max - n_min + 1e-5)
    out = (img.astype(float) * noise_norm)
    params = {
        "paper_texture_min": float(n_min),
        "paper_texture_max": float(n_max)
    }
    return np.clip(out, 0, 255).astype(np.uint8), params

def sub_stains_blotches(img, intensity=0.2, num_stains=2, seed=None):
    """Generates low-frequency dark gray/brownish stains on the substrate."""
    if seed is not None:
        np.random.seed(seed)
    h, w = img.shape
    out = img.astype(float)
    params = {"sub_stains_count": num_stains}
    
    for i in range(num_stains):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        sx = np.random.uniform(20.0, 80.0)
        sy = np.random.uniform(10.0, 30.0)
        opacity = np.random.uniform(0.1, intensity)
        
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        dist = ((x - cx) ** 2) / (sx ** 2) + ((y - cy) ** 2) / (sy ** 2)
        blotch = np.exp(-dist)
        out = out * (1.0 - opacity * blotch)
        
    return np.clip(out, 0, 255).astype(np.uint8), params

def sub_edge_darkening(img, shadow_width=15, intensity=0.7, edge="both", seed=None):
    """Applies page-shadow or crop-edge shadow bands at margins."""
    if seed is not None:
        np.random.seed(seed)
    h, w = img.shape
    out = img.copy()
    params = {"edge_darkening_shadow_width": shadow_width, "edge_darkening_edge": edge}
    
    if edge in ["top", "both"] and shadow_width > 0:
        for y in range(shadow_width):
            factor = (y / shadow_width)
            val = int(100 + 155 * (1.0 - intensity * (1.0 - factor)))
            out[y, :] = np.minimum(out[y, :], val)
            
    if edge in ["bottom", "both"] and shadow_width > 0:
        for y_offset in range(shadow_width):
            y = h - 1 - y_offset
            factor = (y_offset / shadow_width)
            val = int(100 + 155 * (1.0 - intensity * (1.0 - factor)))
            out[y, :] = np.minimum(out[y, :], val)
            
    return out, params

def draw_showthrough_pattern(h, w, seed=None):
    """Generates mirrored abstract Devanagari-like stroke segments."""
    if seed is not None:
        np.random.seed(seed)
    canvas = np.ones((h, w), dtype=np.uint8) * 255
    
    num_lines = np.random.randint(2, 5)
    for _ in range(num_lines):
        y = np.random.randint(h // 4, 3 * h // 4)
        x1 = np.random.randint(0, w // 2)
        x2 = np.random.randint(w // 2, w)
        thickness = np.random.randint(2, 5)
        cv2.line(canvas, (x1, y), (x2, y), 0, thickness)
        
    num_strokes = np.random.randint(10, 25)
    for _ in range(num_strokes):
        x = np.random.randint(10, w - 10)
        y1 = np.random.randint(h // 4, 3 * h // 4)
        y2 = y1 + np.random.randint(10, 30)
        thickness = np.random.randint(1, 4)
        cv2.line(canvas, (x, y1), (x, y2), 0, thickness)
        
    num_loops = np.random.randint(5, 12)
    for _ in range(num_loops):
        cx = np.random.randint(10, w - 10)
        cy = np.random.randint(h // 4, 3 * h // 4)
        r = np.random.randint(4, 8)
        cv2.circle(canvas, (cx, cy), r, 0, -1)
        
    canvas = cv2.flip(canvas, 1)
    blur_kernel = np.random.choice([9, 11, 13])
    canvas = cv2.GaussianBlur(canvas, (blur_kernel, blur_kernel), 0)
    return canvas

def draw_diffuse_shadows(h, w, seed=None):
    """Generates diffuse abstract reverse-side page patterns."""
    if seed is not None:
        np.random.seed(seed)
    canvas = np.ones((h, w), dtype=np.uint8) * 255
    num_shadows = np.random.randint(2, 5)
    for _ in range(num_shadows):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        rx = np.random.randint(40, 150)
        ry = np.random.randint(20, 60)
        cv2.ellipse(canvas, (cx, cy), (rx, ry), np.random.randint(0, 360), 0, 360, 220, -1)
    canvas = cv2.GaussianBlur(canvas, (31, 31), 0)
    return canvas

def sub_bleedthrough(img, intensity=0.12, seed=None):
    """Applies faint mirrored reverse-side text in the background."""
    h, w = img.shape
    pattern = draw_showthrough_pattern(h, w, seed=seed)
    bg_mask = img >= 160
    out = img.copy().astype(float)
    blend = pattern.astype(float) / 255.0
    factor = 1.0 - (1.0 - blend) * intensity
    out[bg_mask] = out[bg_mask] * factor[bg_mask]
    
    params = {"bleedthrough_intensity": float(intensity)}
    return np.clip(out, 0, 255).astype(np.uint8), params

def sub_reverse_showthrough(img, intensity=0.08, seed=None):
    """Applies diffuse abstract page shadows resembling reverse show-through."""
    h, w = img.shape
    pattern = draw_diffuse_shadows(h, w, seed=seed)
    bg_mask = img >= 160
    out = img.copy().astype(float)
    blend = pattern.astype(float) / 255.0
    factor = 1.0 - (1.0 - blend) * intensity
    out[bg_mask] = out[bg_mask] * factor[bg_mask]
    
    params = {"showthrough_intensity": float(intensity)}
    return np.clip(out, 0, 255).astype(np.uint8), params


# 2. Ink Effects
def ink_gray_fade(img, text_floor=60, bg_ceiling=245):
    """Compresses contrast of ink to fade it into a gray color."""
    out = (text_floor + (img.astype(float) / 255.0) * (bg_ceiling - text_floor)).astype(np.uint8)
    params = {"ink_gray_floor": text_floor, "ink_gray_ceiling": bg_ceiling}
    return out, params

def ink_erosion_thinning(img, iterations=1, blend_weight=1.0):
    """Thins characters using morphological erosion (greyscale dilation), optionally blended with original."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.dilate(img, kernel, iterations=iterations)
    if blend_weight < 1.0:
        out = cv2.addWeighted(img, 1.0 - blend_weight, eroded, blend_weight, 0)
    else:
        out = eroded
    return out, {"ink_thinning_iterations": iterations, "ink_thinning_blend": float(blend_weight)}

def ink_dilation_thickening(img, iterations=1):
    """Thickens characters using morphological dilation (greyscale erosion)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    out = cv2.erode(img, kernel, iterations=iterations)
    return out, {"ink_thickening_iterations": iterations}

def ink_opacity_variation(img, noise_sigma=30, opacity_variation=0.5, seed=None):
    """Applies low-frequency noise strictly on ink pixels to simulate uneven ink flow."""
    if seed is not None:
        np.random.seed(seed)
    h, w = img.shape
    noise = np.random.normal(0, noise_sigma, (h, w))
    noise = cv2.GaussianBlur(noise, (15, 15), 0)
    ink_mask = (img < 220)
    out = img.copy().astype(float)
    out[ink_mask] += noise[ink_mask] * opacity_variation
    params = {
        "ink_opacity_noise_sigma": noise_sigma,
        "ink_opacity_variation": float(opacity_variation)
    }
    return np.clip(out, 0, 255).astype(np.uint8), params

def apply_broken_strokes(img, dropout_rate=0.08, seed=None):
    """Simulates broken strokes using light erosion and pixel dropout."""
    if seed is not None:
        np.random.seed(seed)
    h, w = img.shape
    _, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded_ink = cv2.erode(thresh, kernel, iterations=1)
    
    ink_mask = (eroded_ink > 0)
    dropout_mask = (np.random.rand(h, w) < dropout_rate) & ink_mask
    
    out = img.copy()
    out[dropout_mask] = 255
    thinned = cv2.bitwise_not(eroded_ink)
    blended = cv2.addWeighted(out, 0.7, thinned, 0.3, 0)
    
    params = {
        "broken_strokes_dropout_rate": float(dropout_rate)
    }
    return blended, params

def ink_shirorekha_dropout(img, dropout_rate=0.02, seed=None):
    """Applies random pixel dropouts specifically inside upper portion of characters."""
    if seed is not None:
        np.random.seed(seed)
    h, w = img.shape
    out = img.copy()
    
    _, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    bbox = cv2.boundingRect(thresh)
    
    if bbox[3] > 0:
        y1 = bbox[1] + int(np.random.uniform(0.05, 0.10) * bbox[3])
        y2 = bbox[1] + int(np.random.uniform(0.22, 0.28) * bbox[3])
        y1 = min(h - 1, max(0, y1))
        y2 = min(h, max(y1 + 1, y2))
        
        dropout_mask = (np.random.rand(y2 - y1, w) < dropout_rate)
        region = out[y1:y2, :]
        fg_mask = (region < 180) & dropout_mask
        region[fg_mask] = np.random.randint(235, 255, np.sum(fg_mask)).astype(np.uint8)
        out[y1:y2, :] = region
        
    params = {"shirorekha_dropout_rate": float(dropout_rate)}
    return out, params

def ink_bleed_spread(img, bleed_kernel=3, blend_weight=0.4, seed=None):
    """Simulates ink feathering or bleeding around characters."""
    if seed is not None:
        np.random.seed(seed)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bleed_kernel, bleed_kernel))
    eroded = cv2.erode(img, kernel, iterations=1)
    blurred = cv2.GaussianBlur(eroded, (bleed_kernel, bleed_kernel), 0)
    out = cv2.addWeighted(img, 1.0 - blend_weight, blurred, blend_weight, 0)
    params = {
        "ink_bleed_kernel": bleed_kernel,
        "ink_bleed_blend_weight": float(blend_weight)
    }
    return out, params


# 3. Scan / Capture Effects
def scan_downsample_softness(img, scale_factor=0.7, seed=None):
    """Simulates low-DPI scan softness by resizing down then upsampling."""
    h, w = img.shape
    small_w, small_h = max(2, int(w * scale_factor)), max(2, int(h * scale_factor))
    temp = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    out = cv2.resize(temp, (w, h), interpolation=cv2.INTER_CUBIC)
    params = {"scan_scale_factor": float(scale_factor)}
    return out, params

def apply_motion_blur(img, kernel_size=3, angle=0):
    """Applies motion blur to a grayscale image."""
    if kernel_size < 3:
        return img
    if kernel_size % 2 == 0:
        kernel_size += 1
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)
    return cv2.filter2D(img, -1, kernel)

def scan_blur(img, blur_type="gaussian", sigma=0.5, motion_angle=0, seed=None):
    """Applies blur to simulate scan softness."""
    if blur_type == "gaussian":
        out = cv2.GaussianBlur(img, (3, 3), sigma)
    else:
        out = apply_motion_blur(img, kernel_size=3, angle=motion_angle)
    params = {"scan_blur_type": blur_type, "scan_blur_sigma": float(sigma), "scan_blur_angle": float(motion_angle)}
    return out, params

def scan_streaks(img, num_streaks=2, intensity=0.05, seed=None):
    """Draws thin horizontal scanner lines."""
    if seed is not None:
        np.random.seed(seed)
    h, w = img.shape
    out = img.copy().astype(float)
    params = {"scan_streaks_count": num_streaks}
    
    for _ in range(num_streaks):
        y = np.random.randint(0, h)
        out[y, :] = out[y, :] * (1.0 - intensity)
        
    return np.clip(out, 0, 255).astype(np.uint8), params


# 4. Local Damage Effects
def apply_salt_and_pepper(img, salt_prob=0.003, pepper_prob=0.003, seed=None):
    """Applies salt & pepper noise."""
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    num_salt = np.ceil(salt_prob * img.size)
    coords_y = np.random.randint(0, img.shape[0], int(num_salt))
    coords_x = np.random.randint(0, img.shape[1], int(num_salt))
    out[coords_y, coords_x] = 255
    num_pepper = np.ceil(pepper_prob * img.size)
    coords_y = np.random.randint(0, img.shape[0], int(num_pepper))
    coords_x = np.random.randint(0, img.shape[1], int(num_pepper))
    out[coords_y, coords_x] = 0
    return out

def damage_wormholes(img, num_holes=3, radius_range=(1, 2), seed=None):
    """Punches tiny white circles mostly in background, protecting glyph text bounds."""
    if seed is not None:
        np.random.seed(seed)
    h, w = img.shape
    out = img.copy()
    params = {"wormholes_count": num_holes}
    
    holes_placed = 0
    attempts = 0
    total_fg_loss = 0
    
    r_min = max(1, radius_range[0])
    r_max = max(1, radius_range[1])
    
    while holes_placed < num_holes and attempts < 150:
        attempts += 1
        cx = np.random.randint(10, w - 10)
        cy = np.random.randint(5, h - 5)
        r = np.random.randint(r_min, r_max + 1)
        
        y1, y2 = max(0, cy - r), min(h, cy + r + 1)
        x1, x2 = max(0, cx - r), min(w, cx + r + 1)
        patch = img[y1:y2, x1:x2]
        fg_mask = patch < 160
        fg_loss = int(np.sum(fg_mask))
        
        if fg_loss > 5:
            continue
            
        cv2.circle(out, (cx, cy), r, 255, -1)
        holes_placed += 1
        total_fg_loss += fg_loss
        
    params["wormhole_fg_loss"] = total_fg_loss
    params["wormhole_fg_loss_exceeded"] = (total_fg_loss > 10) or (holes_placed < num_holes)
    return out, params

def damage_fold_crease(img, intensity=0.5, seed=None):
    """Draws a subtle crease shadow/highlight line."""
    if seed is not None:
        np.random.seed(seed)
    h, w = img.shape
    out = img.copy()
    
    x1 = np.random.randint(10, w - 10)
    x2 = x1 + np.random.randint(-15, 15)
    y1, y2 = 0, h - 1
    
    temp = out.astype(float)
    cv2.line(temp, (x1, y1), (x2, y2), 100.0, 1)
    cv2.line(temp, (x1 + 1, y1), (x2 + 1, y2), 255.0, 1)
    
    blurred = cv2.GaussianBlur(temp.astype(np.uint8), (3, 3), 0)
    out = cv2.addWeighted(out, 1.0 - intensity, blurred, intensity, 0)
    
    params = {
        "crease_pos_x1": x1,
        "crease_pos_x2": x2,
        "crease_intensity": float(intensity)
    }
    return out, params


# 5. Geometry Effects
def apply_global_geometry(img, max_rotation=1.5, max_translation=1.5, seed=None):
    """Applies slight rotation, translation, and affine shear."""
    if seed is not None:
        np.random.seed(seed)
    h, w = img.shape
    angle = np.random.uniform(-max_rotation, max_rotation)
    tx = np.random.uniform(-max_translation, max_translation)
    ty = np.random.uniform(-max_translation, max_translation)
    shear_x = np.random.uniform(-0.015, 0.015)
    shear_y = np.random.uniform(-0.015, 0.015)
    
    cx, cy = w / 2.0, h / 2.0
    R = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    R[0, 2] += tx
    R[1, 2] += ty
    R[0, 1] += shear_x
    R[1, 0] += shear_y
    
    out = cv2.warpAffine(img, R, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    params = {
        "geom_rotation_deg": float(angle),
        "geom_translation_x": float(tx),
        "geom_translation_y": float(ty),
        "geom_shear_x": float(shear_x),
        "geom_shear_y": float(shear_y)
    }
    return out, params

def apply_sinusoidal_warp(img, amplitude=1.0, wavelength=150.0, phase=0.0):
    """Sinusoidal warp for baseline distortion."""
    h, w = img.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    dy = amplitude * np.sin(2 * np.pi * grid_x / wavelength + phase)
    map_x = grid_x.astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return warped


# --- Profile Architectures & Mappings ---

def D0_clean_print(img, seed=None):
    out = img.copy()
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    out, geom = apply_global_geometry(out, max_rotation=0.3, max_translation=0.5, seed=seed)
    return out, geom

def D1_light_scan(img, seed=None):
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    out, s = scan_blur(out, blur_type="gaussian", sigma=0.3, seed=seed)
    alpha = np.random.uniform(0.9, 0.98)
    beta = np.random.uniform(2, 8)
    out = np.clip(alpha * out + beta, 0, 255).astype(np.uint8)
    out, geom = apply_global_geometry(out, max_rotation=0.8, max_translation=0.8, seed=seed)
    params = {"contrast_alpha": float(alpha), "contrast_beta": float(beta)}
    params.update(s)
    params.update(geom)
    return out, params

def D2_low_contrast_faded_ink(img, seed=None):
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    out, f = ink_gray_fade(out, text_floor=60, bg_ceiling=240)
    out, s = scan_blur(out, blur_type="gaussian", sigma=0.5, seed=seed)
    out, geom = apply_global_geometry(out, max_rotation=1.0, max_translation=1.0, seed=seed)
    params = {}
    params.update(f)
    params.update(s)
    params.update(geom)
    return out, params

def D3_uneven_background(img, seed=None):
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    out, illum = apply_uneven_illumination(out, seed=seed)
    out, s = scan_blur(out, blur_type="gaussian", sigma=0.4, seed=seed)
    out, geom = apply_global_geometry(out, max_rotation=1.2, max_translation=1.2, seed=seed)
    params = {}
    params.update(illum)
    params.update(s)
    params.update(geom)
    return out, params

def D4_broken_strokes(img, seed=None):
    if seed is not None:
        np.random.seed(seed)
    out, broken = apply_broken_strokes(img, dropout_rate=0.08, seed=seed)
    out, geom = apply_global_geometry(out, max_rotation=1.0, max_translation=1.0, seed=seed)
    params = {}
    params.update(broken)
    params.update(geom)
    return out, params

def D5_noisy_archive_scan(img, seed=None):
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    out, f = ink_gray_fade(out, text_floor=45, bg_ceiling=240)
    out, illum = apply_uneven_illumination(out, seed=seed)
    out, s = scan_blur(out, blur_type="gaussian", sigma=0.6, seed=seed)
    out = apply_salt_and_pepper(out, salt_prob=0.004, pepper_prob=0.004, seed=seed)
    out, geom = apply_global_geometry(out, max_rotation=1.5, max_translation=1.5, seed=seed)
    params = {"salt_prob": 0.004, "pepper_prob": 0.004}
    params.update(f)
    params.update(illum)
    params.update(s)
    params.update(geom)
    return out, params

def D6_blurred_scan(img, seed=None):
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    blur_type = np.random.choice(["gaussian", "motion"])
    if blur_type == "gaussian":
        out, s = scan_blur(out, blur_type="gaussian", sigma=1.1, seed=seed)
    else:
        out, s = scan_blur(out, blur_type="motion", sigma=0.0, motion_angle=np.random.uniform(-45, 45), seed=seed)
    out, f = ink_gray_fade(out, text_floor=30, bg_ceiling=240)
    out, geom = apply_global_geometry(out, max_rotation=1.2, max_translation=1.2, seed=seed)
    params = {}
    params.update(s)
    params.update(f)
    params.update(geom)
    return out, params

def D7_old_paper_texture(img, seed=None):
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    out, paper = apply_paper_texture(out, seed=seed)
    out, f = ink_gray_fade(out, text_floor=50, bg_ceiling=245)
    out, s = scan_blur(out, blur_type="gaussian", sigma=0.4, seed=seed)
    out, geom = apply_global_geometry(out, max_rotation=1.2, max_translation=1.2, seed=seed)
    params = {}
    params.update(paper)
    params.update(f)
    params.update(s)
    params.update(geom)
    return out, params

def D8_old_scan_soft(img, seed=None):
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    scale = np.random.uniform(0.75, 0.85)
    out, ds = scan_downsample_softness(out, scale_factor=scale, seed=seed)
    out, s = scan_blur(out, blur_type="gaussian", sigma=np.random.uniform(0.5, 0.8), seed=seed)
    text_min = np.random.randint(40, 60)
    bg_max = np.random.randint(235, 245)
    out, f = ink_gray_fade(out, text_floor=text_min, bg_ceiling=bg_max)
    
    h, w = out.shape
    pepper_prob = np.random.uniform(0.0005, 0.001)
    num_pepper = np.ceil(pepper_prob * out.size)
    coords_y = np.random.randint(0, h, int(num_pepper))
    coords_x = np.random.randint(0, w, int(num_pepper))
    out[coords_y, coords_x] = np.random.randint(20, 80, int(num_pepper))
    
    out, geom = apply_global_geometry(out, max_rotation=0.8, max_translation=0.8, seed=seed)
    params = {"pepper_prob": float(pepper_prob)}
    params.update(ds)
    params.update(s)
    params.update(f)
    params.update(geom)
    return out, params

def D9_faded_gray_ink(img, seed=None):
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    
    _, thresh = cv2.threshold(out, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(thresh, kernel, iterations=1)
    thinned = cv2.bitwise_not(eroded)
    thin_blend_weight = np.random.uniform(0.3, 0.6)
    out = cv2.addWeighted(out, 1.0 - thin_blend_weight, thinned, thin_blend_weight, 0)
    
    out, opac = ink_opacity_variation(out, noise_sigma=25, opacity_variation=np.random.uniform(0.4, 0.7), seed=seed)
    out, f = ink_gray_fade(out, text_floor=np.random.randint(75, 95), bg_ceiling=255)
    out, s = scan_blur(out, blur_type="gaussian", sigma=np.random.uniform(0.3, 0.5), seed=seed)
    out, geom = apply_global_geometry(out, max_rotation=0.8, max_translation=0.8, seed=seed)
    params = {"thin_blend_weight": float(thin_blend_weight)}
    params.update(opac)
    params.update(f)
    params.update(s)
    params.update(geom)
    return out, params

def D10_noisy_scan_crop(img, seed=None):
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    h, w = out.shape
    out, illum = apply_uneven_illumination(out, seed=seed)
    edge_type = np.random.choice(["top", "bottom", "both", "none"])
    
    _, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    bbox = cv2.boundingRect(thresh)
    text_top = bbox[1] if bbox[3] > 0 else h // 2
    text_bottom = bbox[1] + bbox[3] if bbox[3] > 0 else h // 2
    
    edge_params = {}
    if edge_type in ["top", "both"]:
        max_thickness = min(6, max(0, text_top - 2))
        if max_thickness >= 2:
            thickness = np.random.randint(2, max_thickness + 1)
            out, _ = sub_edge_darkening(out, shadow_width=thickness, intensity=0.7, edge="top", seed=seed)
            edge_params["top_edge_thickness"] = thickness
            
    if edge_type in ["bottom", "both"]:
        max_thickness = min(6, max(0, h - text_bottom - 2))
        if max_thickness >= 2:
            thickness = np.random.randint(2, max_thickness + 1)
            out, _ = sub_edge_darkening(out, shadow_width=thickness, intensity=0.7, edge="bottom", seed=seed)
            edge_params["bottom_edge_thickness"] = thickness
            
    pepper_prob = np.random.uniform(0.001, 0.003)
    out = apply_salt_and_pepper(out, salt_prob=0.0, pepper_prob=pepper_prob, seed=seed)
    out, geom = apply_global_geometry(out, max_rotation=1.0, max_translation=1.5, seed=seed)
    params = {"scan_edge_type": edge_type, "pepper_prob": float(pepper_prob)}
    params.update(illum)
    params.update(edge_params)
    params.update(geom)
    return out, params

def D11_mild_warped_line(img, seed=None):
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    amplitude = np.random.uniform(0.5, 1.2)
    wavelength = np.random.uniform(120.0, 240.0)
    phase = np.random.uniform(0, 2 * np.pi)
    out = apply_sinusoidal_warp(out, amplitude, wavelength, phase)
    out, ds = scan_downsample_softness(out, scale_factor=np.random.uniform(0.8, 0.9), seed=seed)
    out, s = scan_blur(out, blur_type="gaussian", sigma=np.random.uniform(0.4, 0.7), seed=seed)
    out, geom = apply_global_geometry(out, max_rotation=0.8, max_translation=0.8, seed=seed)
    params = {"warp_amplitude": float(amplitude), "warp_wavelength": float(wavelength), "warp_phase": float(phase)}
    params.update(ds)
    params.update(s)
    params.update(geom)
    return out, params

def D12_dark_thick_scan(img, seed=None):
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    out, thick = ink_dilation_thickening(out, iterations=1)
    bg_max = np.random.randint(230, 242)
    out, f = ink_gray_fade(out, text_floor=0, bg_ceiling=bg_max)
    out, s = scan_blur(out, blur_type="gaussian", sigma=np.random.uniform(0.4, 0.8), seed=seed)
    noise_prob = np.random.uniform(0.0005, 0.0015)
    out = apply_salt_and_pepper(out, salt_prob=noise_prob/2.0, pepper_prob=noise_prob/2.0, seed=seed)
    out, geom = apply_global_geometry(out, max_rotation=0.8, max_translation=0.8, seed=seed)
    params = {"noise_prob": float(noise_prob)}
    params.update(thick)
    params.update(f)
    params.update(s)
    params.update(geom)
    return out, params

def D13_washed_out_faint_manuscript(img, seed=None):
    """Simulate very faint old manuscript or badly scanned light ink. Adaptive retry protected."""
    if seed is not None:
        np.random.seed(seed)
        
    out = img.copy()
    h, w = out.shape
    
    for attempt in range(5):
        params = {}
        attenuation = 1.0 - 0.2 * attempt
        
        text_floor = int(np.random.randint(95, 115) - (20 * (1.0 - attenuation)))
        bg_ceiling = int(np.random.randint(235, 245) + (10 * (1.0 - attenuation)))
        bg_ceiling = min(255, bg_ceiling)
        temp_out, f = ink_gray_fade(img, text_floor=text_floor, bg_ceiling=bg_ceiling)
        params.update(f)
        
        temp_out, thin = apply_broken_strokes(temp_out, dropout_rate=0.03 * attenuation, seed=seed)
        params.update(thin)
        
        temp_out, opac = ink_opacity_variation(temp_out, noise_sigma=25, opacity_variation=0.6 * attenuation, seed=seed)
        params.update(opac)
        
        temp_out, shiro = ink_shirorekha_dropout(temp_out, dropout_rate=0.02 * attenuation, seed=seed)
        params.update(shiro)
        
        blur_sigma = np.random.uniform(0.3, 0.5) * attenuation
        if blur_sigma > 0.1:
            temp_out, s = scan_blur(temp_out, blur_type="gaussian", sigma=blur_sigma, seed=seed)
            params.update(s)
            
        noise_prob = np.random.uniform(0.0005, 0.001)
        temp_out = apply_salt_and_pepper(temp_out, salt_prob=noise_prob/2.0, pepper_prob=noise_prob/2.0, seed=seed)
        temp_out, geom = apply_global_geometry(temp_out, max_rotation=0.8, max_translation=0.8, seed=seed)
        params.update(geom)
        
        bg_pixels = temp_out[temp_out >= 160]
        fg_pixels = temp_out[temp_out < 160]
        mean_bg = np.mean(bg_pixels) if bg_pixels.size > 0 else 255.0
        mean_fg = np.mean(fg_pixels) if fg_pixels.size > 0 else 0.0
        contrast = mean_bg - mean_fg
        
        if contrast >= 35.0 or attempt == 4:
            params["attempts"] = attempt + 1
            params["contrast_estimate"] = float(contrast)
            out = temp_out
            break
            
    return out, params

def D14_low_dpi_compressed_scan(img, seed=None):
    """Simulate low-resolution old scan crop or compressed scanned line. Adaptive retry protected."""
    if seed is not None:
        np.random.seed(seed)
        
    out = img.copy()
    h, w = out.shape
    
    for attempt in range(5):
        params = {}
        attenuation = 1.0 - 0.2 * attempt
        
        scale_factor = np.random.uniform(0.55, 0.75) + (0.25 * (1.0 - attenuation))
        scale_factor = min(0.95, scale_factor)
        temp_out, ds = scan_downsample_softness(img, scale_factor=scale_factor, seed=seed)
        params.update(ds)
        
        blur_sigma = np.random.uniform(0.4, 0.6) * attenuation
        if blur_sigma > 0.1:
            temp_out, s = scan_blur(temp_out, blur_type="gaussian", sigma=blur_sigma, seed=seed)
            params.update(s)
            
        text_floor = int(np.random.randint(50, 70) - (20 * (1.0 - attenuation)))
        temp_out, f = ink_gray_fade(temp_out, text_floor=text_floor, bg_ceiling=255)
        params.update(f)
        
        pepper_prob = np.random.uniform(0.0005, 0.001)
        temp_out = apply_salt_and_pepper(temp_out, salt_prob=0.0, pepper_prob=pepper_prob, seed=seed)
        
        if np.random.rand() < 0.5:
            temp_out = apply_motion_blur(temp_out, kernel_size=3, angle=0)
            params["horizontal_softness"] = True
            
        _, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
        bbox = cv2.boundingRect(thresh)
        text_top = bbox[1] if bbox[3] > 0 else h // 2
        text_bottom = bbox[1] + bbox[3] if bbox[3] > 0 else h // 2
        
        edge_type = np.random.choice(["top", "bottom", "none"])
        if edge_type == "top":
            max_thickness = min(4, max(0, text_top - 2))
            if max_thickness >= 2:
                thickness = np.random.randint(2, max_thickness + 1)
                temp_out, p = sub_edge_darkening(temp_out, shadow_width=thickness, intensity=0.7, edge="top", seed=seed)
                params["top_edge_thickness"] = thickness
        elif edge_type == "bottom":
            max_thickness = min(4, max(0, h - text_bottom - 2))
            if max_thickness >= 2:
                thickness = np.random.randint(2, max_thickness + 1)
                temp_out, p = sub_edge_darkening(temp_out, shadow_width=thickness, intensity=0.7, edge="bottom", seed=seed)
                params["bottom_edge_thickness"] = thickness
                
        temp_out, geom = apply_global_geometry(temp_out, max_rotation=0.8, max_translation=0.8, seed=seed)
        params.update(geom)
        
        bg_pixels = temp_out[temp_out >= 160]
        fg_pixels = temp_out[temp_out < 160]
        mean_bg = np.mean(bg_pixels) if bg_pixels.size > 0 else 255.0
        mean_fg = np.mean(fg_pixels) if fg_pixels.size > 0 else 0.0
        contrast = mean_bg - mean_fg
        
        if contrast >= 35.0 or attempt == 4:
            params["attempts"] = attempt + 1
            params["contrast_estimate"] = float(contrast)
            out = temp_out
            break
            
    return out, params

def D15_bleedthrough_archival(img, seed=None):
    """Simulate reverse-side text bleed-through."""
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    out, bleed = sub_bleedthrough(out, intensity=np.random.uniform(0.08, 0.14), seed=seed)
    out, paper = apply_paper_texture(out, seed=seed)
    out, s = scan_blur(out, blur_type="gaussian", sigma=0.4, seed=seed)
    out, geom = apply_global_geometry(out, max_rotation=0.8, max_translation=0.8, seed=seed)
    params = {}
    params.update(bleed)
    params.update(paper)
    params.update(s)
    params.update(geom)
    return out, params

def D16_ink_bleed_spread(img, seed=None):
    """Simulate ink bleeding/spreading in old paper fibers."""
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    out, bleed = ink_bleed_spread(out, bleed_kernel=3, blend_weight=0.35, seed=seed)
    out, paper = apply_paper_texture(out, seed=seed)
    out, f = ink_gray_fade(out, text_floor=45, bg_ceiling=240)
    out, geom = apply_global_geometry(out, max_rotation=0.8, max_translation=0.8, seed=seed)
    params = {}
    params.update(bleed)
    params.update(paper)
    params.update(f)
    params.update(geom)
    return out, params

def D17_local_stain_shadow(img, seed=None):
    """Simulate page stains and shadow patches."""
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    h, w = out.shape
    out, stain = sub_stains_blotches(out, intensity=0.18, num_stains=np.random.randint(1, 3), seed=seed)
    
    thickness = np.random.randint(10, min(30, h // 3))
    out, edge = sub_edge_darkening(out, shadow_width=thickness, intensity=0.4, edge=np.random.choice(["top", "bottom"]), seed=seed)
    
    out, geom = apply_global_geometry(out, max_rotation=0.8, max_translation=0.8, seed=seed)
    params = {}
    params.update(stain)
    params.update(edge)
    params.update(geom)
    return out, params

def D18_fold_crease_damage(img, seed=None):
    """Simulate fold lines crossing text."""
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    out, crease = damage_fold_crease(out, intensity=0.5, seed=seed)
    amplitude = np.random.uniform(0.5, 1.0)
    out = apply_sinusoidal_warp(out, amplitude=amplitude, wavelength=140.0, phase=np.random.uniform(0, 2*np.pi))
    out, paper = apply_paper_texture(out, seed=seed)
    out, geom = apply_global_geometry(out, max_rotation=0.8, max_translation=0.8, seed=seed)
    params = {"warp_amplitude": amplitude}
    params.update(crease)
    params.update(paper)
    params.update(geom)
    return out, params

def D19_weak_shirorekha_dropout(img, seed=None):
    """Simulate weak shirorekha lines with adaptive readability guard."""
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    params = {}
    
    for attempt in range(5):
        attenuation = 1.0 - 0.2 * attempt
        temp_out, shiro = ink_shirorekha_dropout(img, dropout_rate=0.035 * attenuation, seed=seed)
        temp_out, f = ink_gray_fade(temp_out, text_floor=np.random.randint(55, 75), bg_ceiling=245)
        temp_out, s = scan_blur(temp_out, blur_type="gaussian", sigma=0.4, seed=seed)
        temp_out, geom = apply_global_geometry(temp_out, max_rotation=0.8, max_translation=0.8, seed=seed)
        
        bg_pixels = temp_out[temp_out >= 160]
        fg_pixels = temp_out[temp_out < 160]
        mean_bg = np.mean(bg_pixels) if bg_pixels.size > 0 else 255.0
        mean_fg = np.mean(fg_pixels) if fg_pixels.size > 0 else 0.0
        contrast = mean_bg - mean_fg
        
        readability_score, continuity_ratio = estimate_readability_and_continuity(temp_out, img)
        if (contrast >= 35.0 and readability_score >= 75.0 and continuity_ratio >= 0.70) or attempt == 4:
            params = {"shirorekha_attempts": attempt + 1, "readability_score_check": readability_score, "continuity_ratio": continuity_ratio}
            params.update(shiro)
            params.update(f)
            params.update(s)
            params.update(geom)
            out = temp_out
            break
            
    return out, params

def D20_partial_erosion_thin_strokes(img, seed=None):
    """Simulate fine stroke erosion with readability guard."""
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    params = {}
    
    for attempt in range(5):
        attenuation = 1.0 - 0.2 * attempt
        # Use fractional erosion with blend_weight = attenuation to protect thin strokes
        temp_out, thin = ink_erosion_thinning(img, iterations=1, blend_weight=attenuation)
        temp_out, broken = apply_broken_strokes(temp_out, dropout_rate=0.03 * attenuation, seed=seed)
        temp_out, s = scan_blur(temp_out, blur_type="gaussian", sigma=0.3, seed=seed)
        temp_out = apply_salt_and_pepper(temp_out, salt_prob=0.0, pepper_prob=0.0005, seed=seed)
        temp_out, geom = apply_global_geometry(temp_out, max_rotation=0.8, max_translation=0.8, seed=seed)
        
        bg_pixels = temp_out[temp_out >= 160]
        fg_pixels = temp_out[temp_out < 160]
        mean_bg = np.mean(bg_pixels) if bg_pixels.size > 0 else 255.0
        mean_fg = np.mean(fg_pixels) if fg_pixels.size > 0 else 0.0
        contrast = mean_bg - mean_fg
        
        readability_score, continuity_ratio = estimate_readability_and_continuity(temp_out, img)
        if (contrast >= 35.0 and readability_score >= 75.0 and continuity_ratio >= 0.65) or attempt == 4:
            params = {"erosion_attempts": attempt + 1, "readability_score_check": readability_score, "continuity_ratio": continuity_ratio}
            params.update(thin)
            params.update(broken)
            params.update(s)
            params.update(geom)
            out = temp_out
            break
            
    return out, params

def D21_mixed_manuscript_hard(img, seed=None):
    """Combines several moderate effects for high realistic difficulty."""
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    out, paper = apply_paper_texture(out, seed=seed)
    out, f = ink_gray_fade(out, text_floor=np.random.randint(65, 85), bg_ceiling=245)
    
    amplitude = np.random.uniform(0.5, 0.8)
    out = apply_sinusoidal_warp(out, amplitude=amplitude, wavelength=160.0, phase=np.random.uniform(0, 2*np.pi))
    
    out, ds = scan_downsample_softness(out, scale_factor=0.8, seed=seed)
    out, s = scan_blur(out, blur_type="gaussian", sigma=0.4, seed=seed)
    
    out = apply_salt_and_pepper(out, salt_prob=0.0, pepper_prob=0.001, seed=seed)
    out, geom = apply_global_geometry(out, max_rotation=0.8, max_translation=0.8, seed=seed)
    params = {"warp_amplitude": amplitude, "salt_pepper": True}
    params.update(paper)
    params.update(f)
    params.update(ds)
    params.update(s)
    params.update(geom)
    return out, params

def D22_reverse_showthrough_soft(img, seed=None):
    """Simulate thin paper show-through shadow blocks."""
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    out, show = sub_reverse_showthrough(out, intensity=np.random.uniform(0.06, 0.10), seed=seed)
    out, illum = apply_uneven_illumination(out, seed=seed)
    out, s = scan_blur(out, blur_type="gaussian", sigma=0.4, seed=seed)
    out, geom = apply_global_geometry(out, max_rotation=0.8, max_translation=0.8, seed=seed)
    params = {}
    params.update(show)
    params.update(illum)
    params.update(s)
    params.update(geom)
    return out, params

def D23_fragmented_scan_crop(img, seed=None):
    """Simulate crop window shadow bands and misalignment."""
    if seed is not None:
        np.random.seed(seed)
    out = img.copy()
    h, w = out.shape
    
    _, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    bbox = cv2.boundingRect(thresh)
    text_top = bbox[1] if bbox[3] > 0 else h // 2
    text_bottom = bbox[1] + bbox[3] if bbox[3] > 0 else h // 2
    
    edge_params = {}
    max_top_thickness = min(5, max(0, text_top - 2))
    if max_top_thickness >= 2:
        top_thickness = np.random.randint(2, max_top_thickness + 1)
        out, _ = sub_edge_darkening(out, shadow_width=top_thickness, intensity=0.6, edge="top", seed=seed)
        edge_params["top_edge_thickness"] = top_thickness
        
    max_bot_thickness = min(5, max(0, h - text_bottom - 2))
    if max_bot_thickness >= 2:
        bot_thickness = np.random.randint(2, max_bot_thickness + 1)
        out, _ = sub_edge_darkening(out, shadow_width=bot_thickness, intensity=0.6, edge="bottom", seed=seed)
        edge_params["bottom_edge_thickness"] = bot_thickness
        
    out, ds = scan_downsample_softness(out, scale_factor=0.8, seed=seed)
    out = apply_salt_and_pepper(out, salt_prob=0.0, pepper_prob=0.001, seed=seed)
    
    out, geom = apply_global_geometry(out, max_rotation=1.2, max_translation=1.2, seed=seed)
    params = {"edge_dust": True}
    params.update(edge_params)
    params.update(ds)
    params.update(geom)
    return out, params


# Mapping dictionary
DEGRADATION_PROFILES = {
    "D0_clean_print": D0_clean_print,
    "D1_light_scan": D1_light_scan,
    "D2_low_contrast_faded_ink": D2_low_contrast_faded_ink,
    "D3_uneven_background": D3_uneven_background,
    "D4_broken_strokes": D4_broken_strokes,
    "D5_noisy_archive_scan": D5_noisy_archive_scan,
    "D6_blurred_scan": D6_blurred_scan,
    "D7_old_paper_texture": D7_old_paper_texture,
    "D8_old_scan_soft": D8_old_scan_soft,
    "D9_faded_gray_ink": D9_faded_gray_ink,
    "D10_noisy_scan_crop": D10_noisy_scan_crop,
    "D11_mild_warped_line": D11_mild_warped_line,
    "D12_dark_thick_scan": D12_dark_thick_scan,
    "D13_washed_out_faint_manuscript": D13_washed_out_faint_manuscript,
    "D14_low_dpi_compressed_scan": D14_low_dpi_compressed_scan,
    "D15_bleedthrough_archival": D15_bleedthrough_archival,
    "D16_ink_bleed_spread": D16_ink_bleed_spread,
    "D17_local_stain_shadow": D17_local_stain_shadow,
    "D18_fold_crease_damage": D18_fold_crease_damage,
    "D19_weak_shirorekha_dropout": D19_weak_shirorekha_dropout,
    "D20_partial_erosion_thin_strokes": D20_partial_erosion_thin_strokes,
    "D21_mixed_manuscript_hard": D21_mixed_manuscript_hard,
    "D22_reverse_showthrough_soft": D22_reverse_showthrough_soft,
    "D23_fragmented_scan_crop": D23_fragmented_scan_crop
}

DIFFICULTY_MAPPING = {
    "D0_clean_print": "easy",
    "D1_light_scan": "easy",
    "D2_low_contrast_faded_ink": "easy",
    "D3_uneven_background": "easy",
    "D8_old_scan_soft": "easy",
    "D12_dark_thick_scan": "easy",
    
    "D4_broken_strokes": "medium",
    "D5_noisy_archive_scan": "medium",
    "D6_blurred_scan": "medium",
    "D7_old_paper_texture": "medium",
    "D9_faded_gray_ink": "medium",
    "D10_noisy_scan_crop": "medium",
    "D11_mild_warped_line": "medium",
    "D14_low_dpi_compressed_scan": "medium",
    
    "D13_washed_out_faint_manuscript": "hard",
    "D15_bleedthrough_archival": "hard",
    "D16_ink_bleed_spread": "hard",
    "D17_local_stain_shadow": "hard",
    "D18_fold_crease_damage": "hard",
    "D19_weak_shirorekha_dropout": "hard",
    "D20_partial_erosion_thin_strokes": "hard",
    "D21_mixed_manuscript_hard": "hard",
    "D22_reverse_showthrough_soft": "hard",
    "D23_fragmented_scan_crop": "hard"
}

def apply_composed_degradation(cv_img, target_difficulty=None, seed=None):
    """Composed degradation mixing multiple mild effects sequentially, with target difficulty control."""
    if seed is not None:
        np.random.seed(seed)
        
    for attempt in range(50):
        loop_seed = seed + attempt * 1000 if seed is not None else None
        if loop_seed is not None:
            np.random.seed(loop_seed)
            
        out = cv_img.copy()
        params = {}
        
        # 1. Substrate Effect
        sub_type = np.random.choice(["illumination", "texture", "stains", "bleedthrough", "showthrough", "none"])
        params["composed_sub_type"] = sub_type
        if sub_type == "illumination":
            out, p = apply_uneven_illumination(out, seed=loop_seed)
            params.update(p)
        elif sub_type == "texture":
            out, p = apply_paper_texture(out, seed=loop_seed)
            params.update(p)
        elif sub_type == "stains":
            out, p = sub_stains_blotches(out, intensity=0.15, num_stains=np.random.randint(1, 3), seed=loop_seed)
            params.update(p)
        elif sub_type == "bleedthrough":
            out, p = sub_bleedthrough(out, intensity=np.random.uniform(0.08, 0.14), seed=loop_seed)
            params.update(p)
        elif sub_type == "showthrough":
            out, p = sub_reverse_showthrough(out, intensity=np.random.uniform(0.06, 0.10), seed=loop_seed)
            params.update(p)
            
        # 2. Ink Effect
        ink_type = np.random.choice(["fade", "thin", "thick", "broken", "shirorekha", "bleed", "none"])
        params["composed_ink_type"] = ink_type
        if ink_type == "fade":
            out, p = D2_low_contrast_faded_ink(out, seed=loop_seed)
            params.update(p)
        elif ink_type == "thin":
            out, p = apply_broken_strokes(out, dropout_rate=0.03, seed=loop_seed)
            params.update(p)
        elif ink_type == "thick":
            out, p = ink_dilation_thickening(out, iterations=1)
            params.update(p)
        elif ink_type == "broken":
            out, p = apply_broken_strokes(out, dropout_rate=0.06, seed=loop_seed)
            params.update(p)
        elif ink_type == "shirorekha":
            out, p = ink_shirorekha_dropout(out, dropout_rate=0.02, seed=loop_seed)
            params.update(p)
        elif ink_type == "bleed":
            out, p = ink_bleed_spread(out, bleed_kernel=3, blend_weight=0.3, seed=loop_seed)
            params.update(p)
            
        # 3. Scan Effect
        scan_type = np.random.choice(["downsample", "blur", "streaks", "none"])
        params["composed_scan_type"] = scan_type
        if scan_type == "downsample":
            out, p = scan_downsample_softness(out, scale_factor=np.random.uniform(0.75, 0.85), seed=loop_seed)
            params.update(p)
        elif scan_type == "blur":
            out, p = scan_blur(out, blur_type="gaussian", sigma=np.random.uniform(0.4, 0.6), seed=loop_seed)
            params.update(p)
        elif scan_type == "streaks":
            out, p = scan_streaks(out, num_streaks=np.random.randint(1, 3), intensity=0.04, seed=loop_seed)
            params.update(p)
            
        # 4. Optional Damage
        damage_applied = "none"
        if np.random.rand() < 0.5:
            damage_type = np.random.choice(["pepper", "wormholes", "crease"])
            damage_applied = damage_type
            if damage_type == "pepper":
                out = apply_salt_and_pepper(out, salt_prob=0.0, pepper_prob=0.001, seed=loop_seed)
                params["salt_pepper_noise"] = True
            elif damage_type == "wormholes":
                out, p = damage_wormholes(out, num_holes=np.random.randint(1, 3), radius_range=(1, 2), seed=loop_seed)
                params.update(p)
            elif damage_type == "crease":
                out, p = damage_fold_crease(out, intensity=0.4, seed=loop_seed)
                params.update(p)
        params["composed_damage_applied"] = damage_applied
                
        # 5. Optional Geometry
        geom_applied = "none"
        if np.random.rand() < 0.5:
            geom_type = np.random.choice(["affine", "warp"])
            geom_applied = geom_type
            if geom_type == "affine":
                out, p = apply_global_geometry(out, max_rotation=0.8, max_translation=0.8, seed=loop_seed)
                params.update(p)
            elif geom_type == "warp":
                amplitude = np.random.uniform(0.5, 1.0)
                out = apply_sinusoidal_warp(out, amplitude=amplitude, wavelength=150.0, phase=np.random.uniform(0, 2*np.pi))
                params["geom_warp_amplitude"] = amplitude
        params["composed_geom_applied"] = geom_applied
        
        # Calculate active count and check difficulty
        active_count = 0
        for t in [sub_type, ink_type, scan_type, damage_applied, geom_applied]:
            if t != "none":
                active_count += 1
        has_hard_effect = (sub_type in ["bleedthrough", "showthrough"]) or \
                          (ink_type in ["shirorekha", "broken"]) or \
                          (damage_applied in ["wormholes", "crease"]) or \
                          (geom_applied == "warp")
        if has_hard_effect and active_count >= 3:
            difficulty = "hard"
        elif active_count >= 2:
            difficulty = "medium"
        else:
            difficulty = "easy"
            
        if target_difficulty is None or difficulty == target_difficulty:
            params["difficulty_bucket"] = difficulty
            return out, params
            
    # Fallback to the last attempt if difficulty wasn't matched
    params["difficulty_bucket"] = difficulty
    return out, params

def apply_degradation(pil_image, profile_name, target_difficulty=None, seed=None, attempt=0):
    """
    Applies the named degradation profile or composed mode to a PIL image.
    If attempt > 0, blends the degraded image with the clean image to guarantee readability.
    Returns: (degraded PIL Image, parameters_dict).
    """
    cv_img = to_cv2(pil_image)
    if profile_name == "composed":
        degraded_cv, params = apply_composed_degradation(cv_img, target_difficulty=target_difficulty, seed=seed)
        params["profile_name"] = "composed"
        params["degradation_family"] = "composed"
    else:
        if profile_name not in DEGRADATION_PROFILES:
            profile_name = "D0_clean_print"
        degraded_cv, params = DEGRADATION_PROFILES[profile_name](cv_img, seed=seed)
        params["profile_name"] = profile_name
        params["degradation_family"] = "single"
        
    # Standardize difficulty bucket mapping
    if profile_name == "composed":
        # Already set dynamically in apply_composed_degradation
        pass
    else:
        params["difficulty_bucket"] = DIFFICULTY_MAPPING.get(profile_name, "easy")
        
    # Readability retry attenuation/blend safety guard
    # If attempt > 0, we blend the degraded cv image back with the clean cv image
    if attempt > 0:
        alpha = min(0.8, attempt * 0.2)
        degraded_cv = cv2.addWeighted(cv_img, alpha, degraded_cv, 1.0 - alpha, 0)
        params["readability_attenuation_alpha"] = float(alpha)
        
    return to_pil(degraded_cv), params
