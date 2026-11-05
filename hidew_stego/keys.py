"""
Password → pixel-order seed.

The secret itself is stored as-is. The password decides *where* its bits go:
Argon2id turns it into a 32-byte seed for the pixel shuffle, so without the
password a reader does not know which pixels to look at. Argon2id is
memory-hard, so each password guess costs real memory and time.
"""

from __future__ import annotations

import os

from argon2.low_level import Type, hash_secret_raw

from .container import Header

#: Argon2id defaults: 64 MiB, 3 passes.
DEFAULT_MEMORY_KIB = 64 * 1024
DEFAULT_TIME_COST = 3
DEFAULT_PARALLELISM = 1


def new_header(
    memory_kib: int = DEFAULT_MEMORY_KIB,
    time_cost: int = DEFAULT_TIME_COST,
    parallelism: int = DEFAULT_PARALLELISM,
) -> Header:
    return Header(memory_kib, time_cost, parallelism, salt=os.urandom(16))


def order_seed(password: str, header: Header) -> bytes:
    return hash_secret_raw(
        secret=password.encode("utf-8"),
        salt=header.salt,
        time_cost=header.time_cost,
        memory_cost=header.memory_kib,
        parallelism=header.parallelism,
        hash_len=32,
        type=Type.ID,
    )
