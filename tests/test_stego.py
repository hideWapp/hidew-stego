import io

import numpy as np
import pytest
from PIL import Image

from hidew_stego import CapacityError, ExtractionError, NotAHideWImage, capacity, hide, read_header, reveal
from hidew_stego.analysis import compare
from hidew_stego.container import RESERVED_PIXELS, Header
from hidew_stego.permutation import PixelOrder

# A light Argon2 setting keeps the suite fast; the scheme is identical.
FAST = {"memory_kib": 8 * 1024, "time_cost": 1}
PASSWORD = "Sunset#Harbour42"
SECRET = b"Meet at the old lighthouse at 6am. Bring the blue notebook."


@pytest.fixture
def cover() -> Image.Image:
    rng = np.random.default_rng(7)
    return Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8), mode="RGB")


def png_roundtrip(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return Image.open(buffer)


def test_round_trip_through_a_png_file(cover):
    stego = png_roundtrip(hide(cover, SECRET, PASSWORD, **FAST))
    assert reveal(stego, PASSWORD) == SECRET


def test_changes_are_invisible(cover):
    result = compare(cover, hide(cover, SECRET, PASSWORD, **FAST))
    assert result.max_difference == 1  # LSB only
    assert result.psnr_db > 60


def test_wrong_password_finds_nothing(cover):
    stego = hide(cover, SECRET, PASSWORD, **FAST)
    with pytest.raises(ExtractionError):
        reveal(stego, "not the password")


def test_tampering_with_the_payload_is_detected(cover):
    stego = hide(cover, SECRET, PASSWORD, **FAST)
    pixels = np.array(stego)
    pixels.reshape(-1, 3)[RESERVED_PIXELS:] ^= 1  # flip every payload LSB
    with pytest.raises(ExtractionError):
        reveal(Image.fromarray(pixels), PASSWORD)


def test_tampering_with_the_header_is_detected(cover):
    stego = hide(cover, SECRET, PASSWORD, **FAST)
    pixels = np.array(stego)
    # Flip one salt bit (header byte 10 starts at bit 80 → pixel 26, channel 2).
    pixels.reshape(-1, 3)[26, 2] ^= 1
    with pytest.raises(ExtractionError):
        reveal(Image.fromarray(pixels), PASSWORD)


def test_lossy_recompression_destroys_the_secret(cover):
    stego = hide(cover, SECRET, PASSWORD, **FAST)
    buffer = io.BytesIO()
    stego.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    with pytest.raises((ExtractionError, NotAHideWImage)):
        reveal(Image.open(buffer), PASSWORD)


def test_plain_images_have_no_header(cover):
    with pytest.raises(NotAHideWImage):
        reveal(cover, PASSWORD)


def test_capacity_is_enforced(cover):
    with pytest.raises(CapacityError):
        hide(cover, bytes(capacity(cover) + 1), PASSWORD, **FAST)
    assert reveal(hide(cover, bytes(capacity(cover)), PASSWORD, **FAST), PASSWORD) == bytes(capacity(cover))


def test_alpha_channel_is_untouched():
    rgba = Image.fromarray(np.full((64, 64, 4), 200, dtype=np.uint8), mode="RGBA")
    stego = hide(rgba, b"hi", PASSWORD, **FAST)
    assert stego.mode == "RGBA"
    assert np.array_equal(np.array(stego)[..., 3], np.array(rgba)[..., 3])


def test_header_is_public_but_salted(cover):
    a = read_header(hide(cover, SECRET, PASSWORD, **FAST))
    b = read_header(hide(cover, SECRET, PASSWORD, **FAST))
    assert a.memory_kib == FAST["memory_kib"] and a.time_cost == 1
    assert a.salt != b.salt
    assert Header.unpack(a.pack()) == a


def test_same_secret_never_produces_the_same_image(cover):
    first = np.array(hide(cover, SECRET, PASSWORD, **FAST))
    second = np.array(hide(cover, SECRET, PASSWORD, **FAST))
    assert not np.array_equal(first, second)


class TestPixelOrder:
    def test_is_a_permutation_of_the_payload_region(self):
        order = PixelOrder(total=5000, start=1024, seed=b"k" * 32).take(5000 - 1024)
        assert sorted(order) == list(range(1024, 5000))

    def test_is_deterministic_and_prefix_stable(self):
        full = PixelOrder(4000, 1024, b"s" * 32).take(500)
        lazy = PixelOrder(4000, 1024, b"s" * 32)
        assert lazy.take(11) == full[:11]
        assert lazy.take(500) == full

    def test_depends_on_the_key(self):
        a = PixelOrder(4000, 1024, b"a" * 32).take(200)
        b = PixelOrder(4000, 1024, b"b" * 32).take(200)
        assert a != b

    def test_spreads_across_the_image(self):
        order = PixelOrder(100_000, 1024, b"x" * 32).take(1000)
        # A sequential writer would stay below pixel 2,024; a shuffle does not.
        assert max(order) > 90_000 and min(order) < 10_000


def test_secret_is_stored_as_is(cover):
    """The original bytes sit in the LSBs of the password-derived pixels."""
    from hidew_stego import lsb
    from hidew_stego.keys import order_seed

    stego = hide(cover, SECRET, PASSWORD, **FAST)
    rgb = np.array(stego).reshape(-1)
    order = PixelOrder(128 * 128, RESERVED_PIXELS, order_seed(PASSWORD, read_header(stego)))
    bits = (4 + len(SECRET)) * 8
    data = lsb.from_bits(lsb.read(rgb, order.take(lsb.pixels_for(bits)), bits))
    assert data[4:] == SECRET
