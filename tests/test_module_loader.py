"""Tests for the ModuleLoader utilities."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import logging
import math
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

numpy_stub = types.ModuleType("numpy")
numpy_stub.array = lambda data, copy=None: data  # type: ignore[assignment]
sys.modules.setdefault("numpy", numpy_stub)

yam_processor_stub = types.ModuleType("yam_processor")
yam_processor_stub.__path__ = [str(PROJECT_ROOT / "yam_processor")]  # type: ignore[attr-defined]
sys.modules.setdefault("yam_processor", yam_processor_stub)

core_stub = types.ModuleType("yam_processor.core")
core_stub.__path__ = [str(PROJECT_ROOT / "yam_processor" / "core")]  # type: ignore[attr-defined]
sys.modules.setdefault("yam_processor.core", core_stub)

MODULE_LOADER_PATH = PROJECT_ROOT / "yam_processor" / "core" / "module_loader.py"
SPEC = importlib.util.spec_from_file_location(
    "yam_processor.core.module_loader", MODULE_LOADER_PATH
)
assert SPEC and SPEC.loader is not None
module_loader_module = importlib.util.module_from_spec(SPEC)
sys.modules["yam_processor.core.module_loader"] = module_loader_module
SPEC.loader.exec_module(module_loader_module)
ModuleLoader = module_loader_module.ModuleLoader
ModuleSignatureVerifier = module_loader_module.ModuleSignatureVerifier
signature_path_for = module_loader_module.signature_path_for

SMALL_PRIMES = [
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
]

DIGESTINFO_SHA256_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")


@dataclass
class RSAKeyPair:
    modulus: int
    public_exponent: int
    private_exponent: int
    public_pem: str


def _encode_length(length: int) -> bytes:
    if length < 0x80:
        return bytes([length])
    length_bytes = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(length_bytes)]) + length_bytes


def _encode_integer(value: int) -> bytes:
    data = value.to_bytes((value.bit_length() + 7) // 8 or 1, "big")
    if data[0] & 0x80:
        data = b"\x00" + data
    return b"\x02" + _encode_length(len(data)) + data


def _encode_sequence(components: list[bytes]) -> bytes:
    body = b"".join(components)
    return b"\x30" + _encode_length(len(body)) + body


def _encode_null() -> bytes:
    return b"\x05\x00"


def _encode_object_identifier(oid: tuple[int, ...]) -> bytes:
    if len(oid) < 2:
        raise ValueError("OID must contain at least two components")
    encoded = [oid[0] * 40 + oid[1]]
    for component in oid[2:]:
        if component == 0:
            encoded.append(0)
            continue
        stack: list[int] = []
        while component:
            stack.append(component & 0x7F)
            component >>= 7
        for index, chunk in enumerate(reversed(stack)):
            if index < len(stack) - 1:
                encoded.append(chunk | 0x80)
            else:
                encoded.append(chunk)
    body = bytes(encoded)
    return b"\x06" + _encode_length(len(body)) + body


def _encode_bit_string(data: bytes) -> bytes:
    return b"\x03" + _encode_length(len(data) + 1) + b"\x00" + data


def _rsa_public_key_der(modulus: int, exponent: int) -> bytes:
    return _encode_sequence([
        _encode_integer(modulus),
        _encode_integer(exponent),
    ])


def _subject_public_key_info_der(modulus: int, exponent: int) -> bytes:
    algorithm_identifier = _encode_sequence([
        _encode_object_identifier((1, 2, 840, 113549, 1, 1, 1)),
        _encode_null(),
    ])
    rsa_key = _rsa_public_key_der(modulus, exponent)
    return _encode_sequence([
        algorithm_identifier,
        _encode_bit_string(rsa_key),
    ])


def _to_pem(der_bytes: bytes, label: str) -> str:
    b64 = base64.b64encode(der_bytes).decode("ascii")
    lines = [b64[i : i + 64] for i in range(0, len(b64), 64)]
    return f"-----BEGIN {label}-----\n" + "\n".join(lines) + f"\n-----END {label}-----\n"


def _generate_prime(bit_length: int, rng: random.Random) -> int:
    while True:
        candidate = rng.getrandbits(bit_length) | (1 << (bit_length - 1)) | 1
        if any(candidate % prime == 0 for prime in SMALL_PRIMES):
            continue
        if _is_probable_prime(candidate, rng):
            return candidate


def _is_probable_prime(value: int, rng: random.Random) -> bool:
    if value % 2 == 0:
        return False
    d = value - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(8):
        a = rng.randrange(2, value - 2)
        x = pow(a, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, value)
            if x == value - 1:
                break
        else:
            return False
    return True


def _generate_rsa_keypair(bit_size: int = 512) -> RSAKeyPair:
    rng = random.SystemRandom()
    half_bits = bit_size // 2
    while True:
        p = _generate_prime(half_bits, rng)
        q = _generate_prime(half_bits, rng)
        if p == q:
            continue
        modulus = p * q
        phi = (p - 1) * (q - 1)
        e = 65537
        if math.gcd(e, phi) != 1:
            continue
        d = pow(e, -1, phi)
        public_pem = _to_pem(
            _subject_public_key_info_der(modulus, e),
            "PUBLIC KEY",
        )
        return RSAKeyPair(modulus, e, d, public_pem)


def _sign_module(data: bytes, key_pair: RSAKeyPair) -> bytes:
    digest = hashlib.sha256(data).digest()
    digest_info = DIGESTINFO_SHA256_PREFIX + digest
    key_size = (key_pair.modulus.bit_length() + 7) // 8
    padding_length = key_size - len(digest_info) - 3
    if padding_length < 8:
        raise ValueError("RSA key too small for PKCS#1 v1.5 signature")
    encoded = b"\x00\x01" + (b"\xFF" * padding_length) + b"\x00" + digest_info
    signature_int = pow(int.from_bytes(encoded, "big"), key_pair.private_exponent, key_pair.modulus)
    return signature_int.to_bytes(key_size, "big")


@pytest.fixture
def module_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample_plugin.py"
    file_path.write_text("VALUE = 42\n", encoding="utf-8")
    return file_path


def test_discover_from_module_file(module_file: Path) -> None:
    loader = ModuleLoader(packages=[], module_paths=[module_file])

    discovered = loader.discover()

    values = [getattr(module, "VALUE", None) for module in discovered]
    assert 42 in values


def test_discover_from_directory_logs_warning_on_failure(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    good_file = tmp_path / "good_plugin.py"
    good_file.write_text("FLAG = 'ok'\n", encoding="utf-8")

    bad_file = tmp_path / "bad_plugin.py"
    bad_file.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    caplog.set_level(logging.WARNING)

    loader = ModuleLoader(packages=[], module_paths=[tmp_path])

    discovered = loader.discover()

    assert any(getattr(module, "FLAG", None) == "ok" for module in discovered)

    warning_records = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and getattr(record, "module_path", "") == str(bad_file)
    ]
    assert warning_records, "Expected a warning log entry for the failing module import"


@pytest.fixture()
def rsa_keypair(tmp_path: Path) -> tuple[RSAKeyPair, Path]:
    key_pair = _generate_rsa_keypair()
    public_key_path = tmp_path / "public_key.pem"
    public_key_path.write_text(key_pair.public_pem, encoding="utf-8")
    return key_pair, public_key_path


def _write_signed_module(module_path: Path, key_pair: RSAKeyPair, content: str) -> None:
    module_path.write_text(content, encoding="utf-8")
    signature = _sign_module(module_path.read_bytes(), key_pair)
    signature_path_for(module_path).write_bytes(signature)


def test_signed_module_is_loaded(tmp_path: Path, rsa_keypair: tuple[RSAKeyPair, Path]) -> None:
    key_pair, public_key_path = rsa_keypair
    module_path = tmp_path / "signed_module.py"
    _write_signed_module(module_path, key_pair, "VALUE = 'signed'\n")

    verifier = ModuleSignatureVerifier([public_key_path])
    loader = ModuleLoader(
        packages=[],
        module_paths=[module_path],
        signature_verifier=verifier,
    )

    modules = loader.discover()

    assert any(getattr(module, "VALUE", None) == "signed" for module in modules)


def test_tampered_module_is_rejected(tmp_path: Path, rsa_keypair: tuple[RSAKeyPair, Path]) -> None:
    key_pair, public_key_path = rsa_keypair
    module_path = tmp_path / "tampered_module.py"
    _write_signed_module(module_path, key_pair, "VALUE = 'original'\n")

    module_path.write_text("VALUE = 'tampered'\n", encoding="utf-8")

    verifier = ModuleSignatureVerifier([public_key_path])
    loader = ModuleLoader(
        packages=[],
        module_paths=[module_path],
        signature_verifier=verifier,
    )

    modules = loader.discover()

    assert modules == []


def test_unsigned_module_is_rejected(tmp_path: Path, rsa_keypair: tuple[RSAKeyPair, Path]) -> None:
    _key_pair, public_key_path = rsa_keypair
    module_path = tmp_path / "unsigned_module.py"
    module_path.write_text("VALUE = 'unsigned'\n", encoding="utf-8")

    verifier = ModuleSignatureVerifier([public_key_path])
    loader = ModuleLoader(
        packages=[],
        module_paths=[module_path],
        signature_verifier=verifier,
    )

    modules = loader.discover()

    assert modules == []
