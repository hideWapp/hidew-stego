"""
hidew-stego — password-scattered LSB steganography for PNG images.

    >>> from hidew_stego import hide, reveal
    >>> stego = hide("photo.png", b"my secret", "correct horse battery staple")
    >>> stego.save("photo-stego.png")
    >>> reveal("photo-stego.png", "correct horse battery staple")
    b'my secret'
"""

from .container import NotAHideWImage
from .stego import CapacityError, ExtractionError, capacity, hide, read_header, reveal

__version__ = "1.0.0"

__all__ = [
    "CapacityError",
    "ExtractionError",
    "NotAHideWImage",
    "capacity",
    "hide",
    "read_header",
    "reveal",
]
