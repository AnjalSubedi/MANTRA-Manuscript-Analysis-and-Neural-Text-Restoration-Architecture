import base64
import html
import os
import unicodedata
import logging
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

# Singleton instance of the Chromium renderer to reuse browser processes
_shared_renderer = None

class ChromiumTextRenderer:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        
    def start(self):
        """Launches the Playwright Chromium instance if not already running."""
        if not self.browser:
            from playwright.sync_api import sync_playwright
            logger.info("Launching headless Chromium for text rendering...")
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.page = self.browser.new_page()
            
    def close(self):
        """Closes the browser and stops Playwright."""
        if self.page:
            self.page.close()
            self.page = None
        if self.browser:
            self.browser.close()
            self.browser = None
        if self.playwright:
            self.playwright.stop()
            self.playwright = None
        logger.info("Headless Chromium session closed.")
            
    def render_text(self, text, font_path, font_size, padding_x, padding_y, bg_color=(255, 255, 255), text_color=(0, 0, 0)):
        """
        Renders the Devanagari text using Chromium's layout engine.
        Returns: PIL Image of the cropped text container.
        """
        self.start()
        
        # 1. Base64 encode the font to inline it, bypassing filesystem security checks
        with open(font_path, "rb") as f:
            font_data = f.read()
        font_base64 = base64.b64encode(font_data).decode("utf-8")
        
        # 2. NFC normalize and escape HTML characters
        normalized_text = unicodedata.normalize("NFC", text)
        escaped_text = html.escape(normalized_text)
        
        # 3. Form HTML page with correct styling
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
        @font-face {{
            font-family: 'DevanagariCustom';
            src: url('data:font/truetype;charset=utf-8;base64,{font_base64}') format('truetype');
        }}
        body {{
            margin: 0;
            padding: 0;
            background-color: transparent;
        }}
        #text-container {{
            display: inline-block;
            font-family: 'DevanagariCustom';
            font-size: {font_size}px;
            line-height: 1.3;
            padding: {padding_y}px {padding_x}px;
            white-space: nowrap;
            background-color: rgb({bg_color[0]}, {bg_color[1]}, {bg_color[2]});
            color: rgb({text_color[0]}, {text_color[1]}, {text_color[2]});
        }}
        </style>
        </head>
        <body>
        <div id="text-container">{escaped_text}</div>
        </body>
        </html>
        """
        
        # Load custom page
        self.page.set_content(html_content)
        
        # Wait until font is fully loaded in page
        self.page.evaluate("document.fonts.ready")
        
        # Screenshot of the element
        element = self.page.query_selector("#text-container")
        if not element:
            raise RuntimeError("Failed to locate text-container element in Chromium render context.")
            
        screenshot_bytes = element.screenshot()
        img = Image.open(BytesIO(screenshot_bytes))
        return img.convert("RGB")

def get_chromium_renderer():
    """Returns the shared Chromium renderer singleton."""
    global _shared_renderer
    if _shared_renderer is None:
        _shared_renderer = ChromiumTextRenderer()
    return _shared_renderer

def shutdown_chromium_renderer():
    """Shuts down the shared Chromium renderer singleton."""
    global _shared_renderer
    if _shared_renderer is not None:
        _shared_renderer.close()
        _shared_renderer = None
