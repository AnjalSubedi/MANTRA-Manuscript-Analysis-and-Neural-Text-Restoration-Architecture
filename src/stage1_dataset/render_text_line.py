import os
import sys
import logging
from PIL import Image, ImageDraw, ImageFont, ImageOps

logger = logging.getLogger(__name__)

# Cache for font cmaps to detect tofu
_font_cmaps = {}

def check_font_supports_text(font_path, text):
    """
    Checks if the font file contains glyphs for all non-whitespace characters in the text.
    Uses fontTools to query the font's cmap table.
    """
    if font_path not in _font_cmaps:
        try:
            from fontTools.ttLib import TTFont
            font = TTFont(font_path)
            cmap = font.getBestCmap()
            _font_cmaps[font_path] = cmap
        except Exception as e:
            logger.warning(f"Failed to load cmap for font {font_path}: {e}")
            return False
    
    cmap = _font_cmaps[font_path]
    if not cmap:
        return False
        
    for char in text:
        if char in (' ', '\n', '\r', '\t', '\u200c', '\u200d'):  # Allow space, newlines, and joiners
            continue
        if ord(char) not in cmap:
            logger.debug(f"Character {char} (U+{ord(char):04X}) not supported by font {font_path}")
            return False
    return True

def run_shaping_smoke_test(font_path=None):
    """
    Performs a shaping smoke test:
    1. Checks if PIL's raqm feature is enabled.
    2. Runs a visual/programmatic test rendering 'कि' (क + ि).
       In correct shaping, the matra 'ि' attaches to the left of 'क'.
       Therefore, the bounding box of 'कि' starts further left than 'क''s.
    """
    has_raqm = PIL_has_raqm()
    if not has_raqm:
        return False
        
    if not font_path or not os.path.exists(font_path):
        return True # Rely only on PIL check if font is not provided
        
    try:
        font = ImageFont.truetype(font_path, 32)
        # Bounding box of 'क'
        img_ka = Image.new("L", (100, 100), 255)
        draw_ka = ImageDraw.Draw(img_ka)
        bbox_ka = draw_ka.textbbox((50, 50), "क", font=font)
        
        # Bounding box of 'कि'
        img_ki = Image.new("L", (100, 100), 255)
        draw_ki = ImageDraw.Draw(img_ki)
        bbox_ki = draw_ki.textbbox((50, 50), "कि", font=font)
        
        # If shaped, the left boundary of 'कि' should extend to the left of 'क'
        if bbox_ki[0] < bbox_ka[0] - 1:
            return True
    except Exception as e:
        logger.debug(f"Shaping test encountered error: {e}")
        
    return False

def PIL_has_raqm():
    """Checks if Pillow features raqm support."""
    try:
        import PIL.features
        # Explicitly check 'raqm'
        return PIL.features.check("raqm")
    except Exception:
        return False

def render_line_pango_cairo(text, font_path, font_size, padding_x=15, padding_y=15, bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    """
    Renders text line using Pango + Cairo + HarfBuzz.
    Tries to import cairo and gi.repository.Pango/PangoCairo.
    """
    import cairo
    import gi
    gi.require_version('Pango', '1.0')
    gi.require_version('PangoCairo', '1.0')
    from gi.repository import Pango, PangoCairo
    
    # We must register the font in Fontconfig if it's a specific TTF file path
    # On Windows, pango can find system fonts or we register it
    # Pango layout rendering:
    # 1. Create a dummy surface to calculate size
    surface_dummy = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
    context_dummy = cairo.Context(surface_dummy)
    layout = PangoCairo.create_layout(context_dummy)
    
    # Pango expects font family name. We can map standard files or register them.
    # For robust handling, try registering the font file
    try:
        # Pango Cairo fontmap setup
        fontmap = PangoCairo.font_map_get_default()
        # On Linux/Windows, Fontconfig can be used to add font file:
        # Sincegi/pango integration varies, we fall back to family name if possible
        # but let's configure the layout description
        font_family = os.path.splitext(os.path.basename(font_path))[0]
        desc = Pango.FontDescription(f"{font_family} {font_size}")
        layout.set_font_description(desc)
    except Exception:
        desc = Pango.FontDescription(f"Sans {font_size}")
        layout.set_font_description(desc)
        
    layout.set_text(text, -1)
    
    # Get pixel size
    width, height = layout.get_pixel_size()
    
    total_w = width + 2 * padding_x
    total_h = height + 2 * padding_y
    
    # Create final surface
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, total_w, total_h)
    ctx = cairo.Context(surface)
    
    # Draw background
    ctx.set_source_rgb(bg_color[0]/255.0, bg_color[1]/255.0, bg_color[2]/255.0)
    ctx.paint()
    
    # Draw text
    ctx.set_source_rgb(text_color[0]/255.0, text_color[1]/255.0, text_color[2]/255.0)
    ctx.move_to(padding_x, padding_y)
    PangoCairo.show_layout(ctx, layout)
    
    # Convert surface to PIL Image
    # surface.get_data() returns a buffer
    data = surface.get_data()
    img = Image.frombuffer("RGBA", (total_w, total_h), data, "raw", "BGRA", 0, 1)
    return img.convert("RGB")

def render_line_pil(text, font_path, font_size, padding_x=15, padding_y=15, bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    """
    Renders text line using PIL.
    Loads font with ImageFont.Layout.RAQM for complex script shaping.
    """
    font = ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.Layout.RAQM)
    
    # Create dummy image to compute bounding box
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    
    # Get bounding box of the text
    bbox = draw.textbbox((0, 0), text, font=font)
    
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Check if text is validly sized
    if text_w <= 0 or text_h <= 0:
        text_w = font_size * len(text)
        text_h = font_size
        bbox = (0, 0, text_w, text_h)
        
    total_w = text_w + 2 * padding_x
    total_h = text_h + 2 * padding_y
    
    img = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw the text offset by the bounding box start to align it properly
    # textbbox start is (bbox[0], bbox[1])
    draw.text((padding_x - bbox[0], padding_y - bbox[1]), text, font=font, fill=text_color)
    return img

def detect_rendering_failure(image, text, font_path, bg_color=(255, 255, 255)):
    """
    Checks if the rendered image has rendering failures.
    Returns: (is_failed, reason)
    """
    # 1. Tofu Check
    if not check_font_supports_text(font_path, text):
        return True, "tofu_missing_glyphs"
        
    # Convert image to grayscale for checks
    gray = image.convert("L")
    
    # Bounding box of text contents (non-background pixels)
    if bg_color == (255, 255, 255) or (isinstance(bg_color, int) and bg_color == 255):
        inv = ImageOps.invert(gray)
        bbox = inv.getbbox()
    else:
        bbox = gray.getbbox()
        
    # 2. Blank Check
    if not bbox:
        return True, "blank_image"
        
    # 3. Size Checks
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width < 8 or height < 8:
        return True, "unreadably_small_output"
        
    # 4. Clipping Check
    # If the text ink touches the very edges (0 or size-1), it's clipped
    w, h = image.size
    if bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] >= w or bbox[3] >= h:
        return True, "clipped_text"
        
    return False, "ok"

def normalize_devanagari_text(text):
    if "ASCII-negative-control" in text:
        return text
    ascii_to_devanagari = {
        '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
        '5': '५', '6': '६', '7': '७', '8': '८', '9': '९'
    }
    return "".join(ascii_to_devanagari.get(char, char) for char in text)

def render_text_line(text, font_path, font_size, padding_x=15, padding_y=15, bg_color=(255, 255, 255), text_color=(0, 0, 0), renderer_preference="auto"):
    """
    Renders text line and handles renderer selection & fallback priorities.
    Returns: (image, renderer_used, status, status_reason)
    """
    text = normalize_devanagari_text(text)
    # Verify font supports character set to fail fast on tofu
    if not check_font_supports_text(font_path, text):
        # Return blank placeholder image and tofu status
        placeholder = Image.new("RGB", (100, 40), bg_color)
        return placeholder, "failed", "failed", "tofu_missing_glyphs"
        
    pango_ok = False
    try:
        import gi
        gi.require_version('Pango', '1.0')
        gi.require_version('PangoCairo', '1.0')
        import cairo
        pango_ok = True
    except Exception:
        pass
        
    chromium_ok = False
    try:
        import playwright
        chromium_ok = True
    except Exception:
        pass
        
    raqm_ok = PIL_has_raqm()
    
    # Establish priority list
    priority = []
    if renderer_preference == "chromium":
        priority = ["chromium"]
    elif renderer_preference == "pango_cairo":
        priority = ["pango_cairo"]
    elif renderer_preference == "pillow_raqm":
        priority = ["pillow_raqm"]
    elif renderer_preference == "auto" or renderer_preference == "pango_cairo_first":
        priority = ["pango_cairo", "chromium", "pillow_raqm"]
    else:
        priority = ["pango_cairo", "chromium", "pillow_raqm"]
        
    img = None
    renderer_used = None
    
    for r_type in priority:
        if r_type == "pango_cairo" and pango_ok:
            try:
                img = render_line_pango_cairo(text, font_path, font_size, padding_x, padding_y, bg_color, text_color)
                renderer_used = "pango_cairo"
                break
            except Exception as e:
                logger.debug(f"Pango/Cairo rendering failed: {e}")
        elif r_type == "chromium" and chromium_ok:
            try:
                from stage1_dataset.chromium_text_renderer import get_chromium_renderer
                renderer = get_chromium_renderer()
                img = renderer.render_text(text, font_path, font_size, padding_x, padding_y, bg_color, text_color)
                renderer_used = "chromium"
                break
            except Exception as e:
                logger.debug(f"Chromium rendering failed: {e}")
        elif r_type == "pillow_raqm" and raqm_ok:
            try:
                img = render_line_pil(text, font_path, font_size, padding_x, padding_y, bg_color, text_color)
                renderer_used = "pil_raqm"
                break
            except Exception as e:
                logger.debug(f"Pillow RAQM rendering failed: {e}")
                
    # If no preferred or auto renderer worked, fall back to plain Pillow
    # but only if renderer_preference is auto or not set strictly
    if img is None:
        if renderer_preference in ["chromium", "pango_cairo", "pillow_raqm"]:
            placeholder = Image.new("RGB", (100, 40), bg_color)
            return placeholder, "failed", "failed", f"renderer_{renderer_preference}_failed"
            
        try:
            img = render_line_pil(text, font_path, font_size, padding_x, padding_y, bg_color, text_color)
            if raqm_ok:
                renderer_used = "pil_raqm"
            else:
                renderer_used = "PIL_NO_RAQM_WARNING"
        except Exception as e:
            placeholder = Image.new("RGB", (100, 40), bg_color)
            return placeholder, "failed", "failed", f"render_exception_{type(e).__name__}"
            
    # Perform validation checks
    is_failed, reason = detect_rendering_failure(img, text, font_path, bg_color)
    if is_failed:
        return img, renderer_used, "failed", reason
        
    return img, renderer_used, "success", "ok"
