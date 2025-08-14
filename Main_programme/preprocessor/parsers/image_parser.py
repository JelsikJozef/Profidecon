import os
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from .base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)

class ImageParser(BaseParser):
    """Parser for image files with multi-language OCR and auto-rotation capabilities."""

    suffixes = (".jpg", ".jpeg", ".png")

    def __init__(self):
        # Configuration from environment
        self.ocr_langs = os.getenv("OCR_LANGS", "eng+slk+deu+ces+pol+spa+ita+fra+por+rus")
        self.max_dimension = int(os.getenv("OCR_MAX_DIMENSION", "2000"))
        self.enable_auto_lang = self.ocr_langs.lower() == "auto"

        # In-memory cache for OSD results during processing session
        self._osd_cache: Dict[str, int] = {}

        logger.info(f"ImageParser initialized with OCR_LANGS='{self.ocr_langs}', max_dimension={self.max_dimension}")

    def _get_image_hash(self, img: Image.Image) -> str:
        """Generate hash for image caching."""
        # Convert to bytes for hashing
        img_bytes = img.tobytes()
        return hashlib.md5(img_bytes).hexdigest()[:16]

    def _detect_rotation_from_exif(self, img: Image.Image) -> tuple[Image.Image, int, str]:
        """Apply EXIF-based rotation if available."""
        try:
            # Apply EXIF orientation correction
            rotated_img = ImageOps.exif_transpose(img)
            if rotated_img is not img:  # Image was rotated
                logger.info("Applied EXIF-based rotation correction")
                return rotated_img, 0, "exif"  # Already corrected, no additional rotation needed
            else:
                return img, 0, "none"
        except Exception as e:
            logger.warning(f"Failed to apply EXIF rotation: {e}")
            return img, 0, "none"

    def _detect_rotation_from_osd(self, img: Image.Image) -> tuple[int, str]:
        """Detect rotation using Tesseract OSD (Orientation and Script Detection)."""
        img_hash = self._get_image_hash(img)

        # Check cache first
        if img_hash in self._osd_cache:
            cached_rotation = self._osd_cache[img_hash]
            logger.debug(f"Using cached OSD rotation: {cached_rotation}°")
            return cached_rotation, "osd_cached"

        try:
            # Use OSD to detect orientation
            osd_data = pytesseract.image_to_osd(img)
            logger.debug(f"OSD data: {osd_data}")

            # Parse rotation from OSD output
            rotation = 0
            for line in osd_data.split('\n'):
                if 'Rotate:' in line:
                    rotation = int(line.split(':')[1].strip())
                    break

            # Cache the result
            self._osd_cache[img_hash] = rotation
            logger.info(f"Detected rotation from OSD: {rotation}°")
            return rotation, "osd"

        except Exception as e:
            logger.warning(f"Failed to detect rotation using OSD: {e}")
            return 0, "osd_failed"

    def _apply_rotation(self, img: Image.Image, rotation: int) -> Image.Image:
        """Apply rotation to image."""
        if rotation == 0:
            return img

        # Convert counter-clockwise rotation to clockwise for PIL
        if rotation == 90:
            return img.rotate(-90, expand=True)
        elif rotation == 180:
            return img.rotate(180, expand=True)
        elif rotation == 270:
            return img.rotate(90, expand=True)
        else:
            logger.warning(f"Unexpected rotation value: {rotation}°, skipping rotation")
            return img

    def _preprocess_for_ocr(self, img: Image.Image) -> Image.Image:
        """Preprocess image to improve OCR accuracy."""
        try:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize if too large
            width, height = img.size
            if min(width, height) > self.max_dimension:
                ratio = self.max_dimension / min(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")

            # Convert to grayscale
            img = img.convert('L')

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)

            # Simple binarization using point transformation
            # Convert to binary (threshold at 128)
            img = img.point(lambda x: 0 if x < 128 else 255, '1')

            return img

        except Exception as e:
            logger.warning(f"Failed to preprocess image: {e}")
            return img

    def _detect_text_language(self, text: str) -> Optional[str]:
        """Detect the primary language of extracted text."""
        if len(text.strip()) < 20:  # Too short for reliable detection
            return None

        try:
            detected_lang = detect(text)
            logger.info(f"Detected text language: {detected_lang}")
            return detected_lang
        except LangDetectException as e:
            logger.warning(f"Failed to detect language: {e}")
            return None

    def _get_extended_lang_config(self, detected_lang: str) -> str:
        """Get extended language configuration based on detected language."""
        # Mapping from langdetect codes to Tesseract language codes
        lang_mapping = {
            'sk': 'slk',  # Slovak
            'cs': 'ces',  # Czech
            'de': 'deu',  # German
            'pl': 'pol',  # Polish
            'es': 'spa',  # Spanish
            'it': 'ita',  # Italian
            'fr': 'fra',  # French
            'pt': 'por',  # Portuguese
            'ru': 'rus',  # Russian
            'en': 'eng',  # English
            'hu': 'hun',  # Hungarian
            'ro': 'ron',  # Romanian
            'uk': 'ukr',  # Ukrainian
        }

        tesseract_lang = lang_mapping.get(detected_lang, 'eng')

        # Create extended language set with detected language prioritized
        base_langs = ['eng', 'slk', 'deu', 'ces']  # Common languages
        if tesseract_lang not in base_langs:
            base_langs.insert(0, tesseract_lang)

        return '+'.join(base_langs)

    def _perform_ocr(self, img: Image.Image) -> tuple[str, list[str]]:
        """Perform OCR with multi-language support."""
        if self.enable_auto_lang:
            # Two-step OCR for automatic language detection
            logger.info("Performing two-step OCR with automatic language detection")

            # Step 1: Conservative OCR with basic languages
            try:
                conservative_langs = "eng+slk"
                text_first = pytesseract.image_to_string(img, lang=conservative_langs).strip()
                logger.debug(f"First pass OCR result length: {len(text_first)}")

                if len(text_first) < 10:
                    logger.warning("First pass OCR returned minimal text")
                    return text_first, [conservative_langs]

                # Step 2: Detect language and perform extended OCR
                detected_lang = self._detect_text_language(text_first)
                if detected_lang:
                    extended_langs = self._get_extended_lang_config(detected_lang)
                    text_final = pytesseract.image_to_string(img, lang=extended_langs).strip()
                    logger.info(f"Final OCR with extended languages: {extended_langs}")
                    return text_final, [conservative_langs, extended_langs]
                else:
                    return text_first, [conservative_langs]

            except Exception as e:
                logger.error(f"Auto-language OCR failed: {e}")
                # Fallback to simple OCR
                try:
                    fallback_text = pytesseract.image_to_string(img, lang="eng").strip()
                    return fallback_text, ["eng"]
                except Exception as e2:
                    logger.error(f"Fallback OCR also failed: {e2}")
                    return "", ["failed"]
        else:
            # Direct OCR with configured languages
            try:
                text = pytesseract.image_to_string(img, lang=self.ocr_langs).strip()
                logger.info(f"OCR completed with languages: {self.ocr_langs}")
                return text, [self.ocr_langs]
            except Exception as e:
                logger.error(f"OCR failed with configured languages: {e}")
                # Fallback to English only
                try:
                    fallback_text = pytesseract.image_to_string(img, lang="eng").strip()
                    return fallback_text, ["eng"]
                except Exception as e2:
                    logger.error(f"English fallback OCR also failed: {e2}")
                    return "", ["failed"]

    def parse(self, path: Path) -> ParsedDocument:
        """Parse image file and extract text using OCR with auto-rotation."""
        logger.info(f"Parsing image file: {path}")

        try:
            # Load image
            with Image.open(path) as img:
                # Make a copy to avoid issues with file handle
                img = img.copy()

                logger.info(f"Loaded image: {img.size}, mode: {img.mode}")

                # Step 1: Apply EXIF-based rotation
                img, exif_rotation, exif_source = self._detect_rotation_from_exif(img)

                # Step 2: Detect additional rotation using OSD if EXIF didn't rotate
                osd_rotation = 0
                osd_source = "none"
                if exif_source == "none":
                    osd_rotation, osd_source = self._detect_rotation_from_osd(img)
                    if osd_rotation != 0:
                        img = self._apply_rotation(img, osd_rotation)

                # Calculate total rotation applied
                total_rotation = exif_rotation + osd_rotation
                rotation_source = exif_source if exif_source != "none" else osd_source

                # Step 3: Preprocess image for OCR
                processed_img = self._preprocess_for_ocr(img)

                # Step 4: Perform OCR
                extracted_text, used_langs = self._perform_ocr(processed_img)

                # Prepare metadata
                metadata = {
                    "source": str(path),
                    "ocr": True,
                    "rotation_applied": total_rotation,
                    "rotation_source": rotation_source,
                    "ocr_langs": used_langs,
                    "image_size": f"{img.size[0]}x{img.size[1]}",
                    "image_mode": img.mode,
                }

                # Check if OCR returned empty text
                if not extracted_text:
                    metadata["ocr_empty"] = True
                    logger.warning(f"OCR returned empty text for {path}")

                logger.info(f"Successfully parsed image: {len(extracted_text)} characters extracted")

                return ParsedDocument(
                    text=extracted_text,
                    metadata=metadata
                )

        except Exception as e:
            logger.error(f"Failed to parse image {path}: {e}")
            # Return minimal document with error information
            return ParsedDocument(
                text="",
                metadata={
                    "source": str(path),
                    "ocr": True,
                    "error": str(e),
                    "ocr_empty": True,
                    "rotation_applied": 0,
                    "rotation_source": "failed",
                    "ocr_langs": ["failed"]
                }
            )
