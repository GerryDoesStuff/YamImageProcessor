"""Utilities for verifying signed plugin modules."""
from __future__ import annotations

import base64
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


LOGGER = logging.getLogger(__name__)


class SignatureVerificationError(RuntimeError):
    """Base class for signature verification failures."""


class TrustStoreError(SignatureVerificationError):
    """Raised when the configured trust store cannot be loaded."""


class MissingSignatureError(SignatureVerificationError):
    """Raised when the expected signature data is not present."""


class InvalidSignatureError(SignatureVerificationError):
    """Raised when signature verification fails for all trusted keys."""


_PEM_BLOCK_PATTERN = re.compile(
    br"-----BEGIN ([^-]+)-----\s*(.*?)\s*-----END \1-----",
    re.DOTALL,
)

_RSA_SHA256_DIGESTINFO_PREFIX = bytes.fromhex(
    "3031300d060960864801650304020105000420"
)

RSAOID = (1, 2, 840, 113549, 1, 1, 1)

PublicKey = Tuple[int, int]


class _DERReader:
    """Minimal DER reader supporting structures required for RSA parsing."""

    def __init__(self, data: bytes) -> None:
        self._data = memoryview(data)
        self._pos = 0

    def _read_length(self) -> int:
        if self._pos >= len(self._data):
            raise ValueError("DER: unexpected end of data when reading length")
        first = self._data[self._pos]
        self._pos += 1
        if first < 0x80:
            return int(first)
        num_bytes = first & 0x7F
        if num_bytes == 0:
            raise ValueError("DER: indefinite lengths are not supported")
        if self._pos + num_bytes > len(self._data):
            raise ValueError("DER: truncated length field")
        length = int.from_bytes(self._data[self._pos : self._pos + num_bytes], "big")
        self._pos += num_bytes
        return length

    def read_tlv(self) -> Tuple[int, bytes]:
        if self._pos >= len(self._data):
            raise ValueError("DER: no more data available")
        tag = int(self._data[self._pos])
        self._pos += 1
        length = self._read_length()
        end = self._pos + length
        if end > len(self._data):
            raise ValueError("DER: truncated value")
        value = self._data[self._pos : end].tobytes()
        self._pos = end
        return tag, value

    def read_sequence(self) -> bytes:
        tag, value = self.read_tlv()
        if tag != 0x30:
            raise ValueError("DER: expected SEQUENCE tag")
        return value

    def read_integer(self) -> int:
        tag, value = self.read_tlv()
        if tag != 0x02:
            raise ValueError("DER: expected INTEGER tag")
        if not value:
            raise ValueError("DER: empty INTEGER value")
        return int.from_bytes(value, "big", signed=False)

    def read_bit_string(self) -> bytes:
        tag, value = self.read_tlv()
        if tag != 0x03:
            raise ValueError("DER: expected BIT STRING tag")
        if not value:
            raise ValueError("DER: empty BIT STRING value")
        unused_bits = value[0]
        if unused_bits != 0:
            raise ValueError("DER: BIT STRING with unused bits is unsupported")
        return value[1:]

    def read_object_identifier(self) -> Tuple[int, ...]:
        tag, value = self.read_tlv()
        if tag != 0x06:
            raise ValueError("DER: expected OBJECT IDENTIFIER tag")
        if not value:
            raise ValueError("DER: empty OBJECT IDENTIFIER value")
        first = value[0]
        oid = [first // 40, first % 40]
        current = 0
        for byte in value[1:]:
            current = (current << 7) | (byte & 0x7F)
            if byte & 0x80:
                continue
            oid.append(current)
            current = 0
        if current:
            oid.append(current)
        return tuple(oid)

    def skip(self) -> None:
        _tag, _value = self.read_tlv()

    def peek_tag(self) -> int:
        if self._pos >= len(self._data):
            raise ValueError("DER: no more data to peek at")
        return int(self._data[self._pos])

    def has_data(self) -> bool:
        return self._pos < len(self._data)


def _iter_pem_blocks(pem_data: bytes) -> Iterable[Tuple[str, bytes]]:
    """Yield PEM block types and decoded DER payloads."""

    for match in _PEM_BLOCK_PATTERN.finditer(pem_data):
        block_type = match.group(1).decode("ascii", errors="ignore").strip()
        body = match.group(2).replace(b"\r", b"")
        try:
            der = base64.b64decode(body)
        except (base64.binascii.Error, ValueError):
            LOGGER.debug("Failed to decode PEM block", extra={"block_type": block_type})
            continue
        yield block_type, der


def _encode_length(length: int) -> bytes:
    if length < 0x80:
        return bytes([length])
    length_bytes = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(length_bytes)]) + length_bytes


def _parse_rsa_public_key(der_bytes: bytes) -> PublicKey:
    reader = _DERReader(der_bytes)
    sequence = reader.read_sequence()
    seq_reader = _DERReader(sequence)
    modulus = seq_reader.read_integer()
    exponent = seq_reader.read_integer()
    return modulus, exponent


def _parse_subject_public_key_info(der_bytes: bytes) -> PublicKey | None:
    reader = _DERReader(der_bytes)
    spki_body = reader.read_sequence()
    spki_reader = _DERReader(spki_body)
    algorithm_sequence = spki_reader.read_sequence()
    algorithm_reader = _DERReader(algorithm_sequence)
    oid = algorithm_reader.read_object_identifier()
    if oid != RSAOID:
        LOGGER.debug("Unsupported algorithm identifier", extra={"oid": oid})
        return None
    if algorithm_reader.has_data() and algorithm_reader.peek_tag() == 0x05:  # NULL parameters
        algorithm_reader.skip()
    bit_string = spki_reader.read_bit_string()
    return _parse_rsa_public_key(bit_string)


def _parse_certificate_for_key(der_bytes: bytes) -> PublicKey | None:
    reader = _DERReader(der_bytes)
    certificate_sequence = reader.read_sequence()
    certificate_reader = _DERReader(certificate_sequence)
    tbs_certificate = certificate_reader.read_sequence()
    certificate_reader.read_sequence()  # signatureAlgorithm - ignored
    certificate_reader.read_bit_string()  # signatureValue - ignored

    tbs_reader = _DERReader(tbs_certificate)
    if tbs_reader.peek_tag() == 0xA0:  # version (context-specific)
        tbs_reader.skip()
    tbs_reader.skip()  # serialNumber
    tbs_reader.read_sequence()  # signature
    tbs_reader.read_sequence()  # issuer
    tbs_reader.read_sequence()  # validity
    tbs_reader.read_sequence()  # subject
    tag, spki_value = tbs_reader.read_tlv()
    if tag != 0x30:
        LOGGER.debug(
            "Unexpected tag for subjectPublicKeyInfo",
            extra={"tag": tag},
        )
        return None
    spki_der = bytes([tag]) + _encode_length(len(spki_value)) + spki_value
    return _parse_subject_public_key_info(spki_der)


def _load_public_key(block_type: str, der_bytes: bytes) -> PublicKey | None:
    normalized = block_type.upper()
    if normalized == "PUBLIC KEY":
        return _parse_subject_public_key_info(der_bytes)
    if normalized == "CERTIFICATE":
        return _parse_certificate_for_key(der_bytes)
    LOGGER.debug("Unsupported PEM block type in trust store", extra={"type": block_type})
    return None


def _verify_rsa_signature(key: PublicKey, data: bytes, signature: bytes) -> bool:
    modulus, exponent = key
    if modulus <= 0 or exponent <= 0:
        return False

    key_size = (modulus.bit_length() + 7) // 8
    if key_size == 0 or len(signature) != key_size:
        return False

    signature_int = int.from_bytes(signature, "big")
    decrypted = pow(signature_int, exponent, modulus)
    em = decrypted.to_bytes(key_size, "big")

    if not em.startswith(b"\x00\x01"):
        return False
    try:
        separator_index = em.index(b"\x00", 2)
    except ValueError:
        return False
    padding = em[2:separator_index]
    if len(padding) < 8 or any(byte != 0xFF for byte in padding):
        return False

    digest_info = em[separator_index + 1 :]
    expected_digest = hashlib.sha256(data).digest()
    expected_t = _RSA_SHA256_DIGESTINFO_PREFIX + expected_digest
    return digest_info == expected_t


@dataclass
class ModuleSignatureVerifier:
    """Verify module signatures against a configured trust store."""

    trust_store_paths: Sequence[Path | str]
    _public_keys: List[PublicKey] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:  # pragma: no cover - trivial dataclass hook
        self._load_trust_store()

    def _load_trust_store(self) -> None:
        if not self.trust_store_paths:
            raise TrustStoreError("At least one trust store path must be configured.")

        keys: List[PublicKey] = []
        for path_like in self.trust_store_paths:
            path = Path(path_like)
            try:
                pem_data = path.read_bytes()
            except OSError as exc:  # pragma: no cover - defensive logging path
                LOGGER.warning("Failed to read trust store entry", exc_info=exc)
                raise TrustStoreError(f"Unable to read trust store file: {path}") from exc

            blocks = list(_iter_pem_blocks(pem_data))
            if not blocks:
                raise TrustStoreError(
                    f"No PEM entries found in trust store file: {path}"
                )
            for block_type, der_bytes in blocks:
                key = _load_public_key(block_type, der_bytes)
                if key is None:
                    continue
                keys.append(key)

        if not keys:
            raise TrustStoreError("No public keys could be loaded from the trust store.")

        self._public_keys = keys

    def verify(self, data: bytes, signature: bytes) -> None:
        """Verify ``signature`` for ``data`` against trusted keys."""

        if not signature:
            raise MissingSignatureError("Signature data is empty.")

        for key in self._public_keys:
            if _verify_rsa_signature(key, data, signature):
                return

        raise InvalidSignatureError("Signature could not be validated with trusted keys.")


def signature_path_for(module_path: Path, extension: str = ".sig") -> Path:
    """Return the expected signature path for ``module_path``."""

    if extension.startswith("."):
        extension = extension[1:]

    if module_path.suffix:
        return module_path.with_suffix(f"{module_path.suffix}.{extension}")
    return module_path.with_name(f"{module_path.name}.{extension}")
