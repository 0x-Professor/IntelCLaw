"""
OCR Processor - Extract text from screenshots.

Uses pytesseract for OCR processing with optimizations.
"""

import asyncio
from typing import Optional

from loguru import logger

try:
    import pytesseract
    from PIL import Image
    import io
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract/Pillow not available - OCR disabled")


class OCRProcessor:
    """
    OCR text extraction from images.
    
    Features:
    - Multiple language support
    - Image preprocessing for better accuracy
    - Region-based extraction
    - Confidence scoring
    """
    
    def __init__(self, language: str = "eng"):
        """
        Initialize OCR processor.
        
        Args:
            language: Tesseract language code
        """
        self._language = language
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize OCR processor."""
        if not TESSERACT_AVAILABLE:
            logger.warning("OCR not available (pytesseract not installed)")
            return
        
        try:
            # Verify tesseract is installed
            version = await asyncio.to_thread(pytesseract.get_tesseract_version)
            logger.info(f"Tesseract OCR initialized (version {version})")
            self._initialized = True
        except Exception as e:
            logger.warning(f"Tesseract not found: {e}. Install from https://github.com/tesseract-ocr/tesseract")
    
    async def extract_text(
        self,
        image: bytes,
        language: Optional[str] = None,
        config: str = ""
    ) -> Optional[str]:
        """
        Extract text from an image.
        
        Args:
            image: Image bytes (PNG or JPEG)
            language: Override language
            config: Additional tesseract config
            
        Returns:
            Extracted text or None
        """
        if not self._initialized:
            return None
        
        try:
            # Load image
            img = Image.open(io.BytesIO(image))
            
            # Preprocess for better OCR
            img = self._preprocess_image(img)
            
            # Extract text
            lang = language or self._language
            text = await asyncio.to_thread(
                pytesseract.image_to_string,
                img,
                lang=lang,
                config=config
            )
            
            # Clean up text
            text = self._clean_text(text)
            
            return text if text else None
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return None
    
    async def extract_with_boxes(
        self,
        image: bytes,
        language: Optional[str] = None
    ) -> list:
        """
        Extract text with bounding boxes.
        
        Returns list of {"text": str, "box": (x, y, w, h), "confidence": float}
        """
        if not self._initialized:
            return []
        
        try:
            img = Image.open(io.BytesIO(image))
            img = self._preprocess_image(img)
            
            lang = language or self._language
            data = await asyncio.to_thread(
                pytesseract.image_to_data,
                img,
                lang=lang,
                output_type=pytesseract.Output.DICT
            )
            
            results = []
            n_boxes = len(data["text"])
            
            for i in range(n_boxes):
                if int(data["conf"][i]) > 0:  # Has confidence
                    text = data["text"][i].strip()
                    if text:
                        results.append({
                            "text": text,
                            "box": (
                                data["left"][i],
                                data["top"][i],
                                data["width"][i],
                                data["height"][i]
                            ),
                            "confidence": float(data["conf"][i]) / 100.0
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"OCR with boxes failed: {e}")
            return []
    
    def _preprocess_image(self, img: "Image.Image") -> "Image.Image":
        """Preprocess image for better OCR accuracy."""
        # Convert to grayscale
        if img.mode != "L":
            img = img.convert("L")
        
        # Resize if too small
        min_size = 1000
        if img.width < min_size or img.height < min_size:
            scale = max(min_size / img.width, min_size / img.height)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        return img
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines)
    
    @property
    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self._initialized
