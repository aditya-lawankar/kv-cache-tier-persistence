"""
Custom raw binary serialization format.
"""

import struct
import zlib
from typing import Dict, Tuple
import numpy as np

from ..config import ModelConfig
from .serializer import CacheSerializer

# Header layout (version 2):
# magic (8 bytes): 'KVCACHE\x00'
# version (2 bytes): 2
# num_layers (2 bytes)
# num_heads (2 bytes)
# head_dim (2 bytes)
# token_count (4 bytes)
# dtype enum (1 byte): 0=float16, 1=float32
# payload crc32 (4 bytes): zlib.crc32 of everything after the header
# padding (39 bytes) to 64 bytes total
#
# Version 1 (legacy, still readable): identical up to the dtype byte,
# then 43 bytes of padding and no checksum.
HEADER_FORMAT_V2 = "<8s H H H H I B I 39s"
HEADER_FORMAT_V1 = "<8s H H H H I B 43s"
HEADER_SIZE = 64
MAGIC = b'KVCACHE\x00'
VERSION = 2

class ChecksumError(ValueError):
    """Payload bytes do not match the checksum recorded at serialization time."""

class RawBinarySerializer(CacheSerializer):
    """
    Custom raw binary format.
    Maximum speed for hot <-> warm tier transfers by skipping complex parsing.
    Payload integrity is protected by a CRC32 checksum: these bytes round-trip
    through NVMe and object storage, where silent corruption must be detected
    rather than fed back into a model as attention state.
    """

    @property
    def format_name(self) -> str:
        return "raw_binary"

    def serialize(self, kv_data: Dict[int, Tuple[np.ndarray, np.ndarray]], model_config: ModelConfig) -> bytes:
        if not kv_data:
            raise ValueError("Cannot serialize empty kv_data")

        # The wire format is dense: deserialize reads exactly num_layers
        # (K, V) pairs, so a sparse dict would silently corrupt the read.
        missing = [l for l in range(model_config.num_layers) if l not in kv_data]
        if missing:
            raise ValueError(
                f"kv_data is missing layers {missing}; "
                f"expected all layers 0..{model_config.num_layers - 1}"
            )

        # Get token_count from first tensor
        layer_0_k = kv_data[0][0]
        token_count = layer_0_k.shape[1]

        dtype_enum = 0 if model_config.dtype in ("float16", "fp16") else 1

        # Collect raw payload bytes in order
        payload_parts = []
        for layer in range(model_config.num_layers):
            k, v = kv_data[layer]
            payload_parts.append(k.tobytes())
            payload_parts.append(v.tobytes())
        payload = b"".join(payload_parts)

        header = struct.pack(
            HEADER_FORMAT_V2,
            MAGIC,
            VERSION,
            model_config.num_layers,
            model_config.num_heads,
            model_config.head_dim,
            token_count,
            dtype_enum,
            zlib.crc32(payload),
            b'\x00' * 39
        )

        return header + payload

    def deserialize(self, data: bytes) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], ModelConfig]:
        if len(data) < HEADER_SIZE:
            raise ValueError("Data too small to contain valid header")

        # Common prefix: magic + version
        magic, version = struct.unpack("<8s H", data[:10])
        if magic != MAGIC:
            raise ValueError(f"Invalid magic signature: {magic}")

        if version == 2:
            (_, _, num_layers, num_heads, head_dim, token_count,
             dtype_enum, expected_crc, _) = struct.unpack(HEADER_FORMAT_V2, data[:HEADER_SIZE])
            actual_crc = zlib.crc32(data[HEADER_SIZE:])
            if actual_crc != expected_crc:
                raise ChecksumError(
                    f"Payload CRC32 mismatch: expected {expected_crc:#010x}, "
                    f"got {actual_crc:#010x} — data corrupted in storage or transit"
                )
        elif version == 1:
            (_, _, num_layers, num_heads, head_dim, token_count,
             dtype_enum, _) = struct.unpack(HEADER_FORMAT_V1, data[:HEADER_SIZE])
        else:
            raise ValueError(f"Unsupported format version: {version}")

        dtype_str = "float16" if dtype_enum == 0 else "float32"
        np_dtype = np.float16 if dtype_enum == 0 else np.float32

        model_config = ModelConfig(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype_str
        )

        # Calculate tensor sizes
        tensor_elements = num_heads * token_count * head_dim
        tensor_bytes = tensor_elements * np_dtype().itemsize

        expected_size = HEADER_SIZE + 2 * num_layers * tensor_bytes
        if len(data) < expected_size:
            raise ValueError(
                f"Truncated payload: expected {expected_size} bytes, got {len(data)}"
            )

        kv_data = {}
        offset = HEADER_SIZE

        for layer in range(num_layers):
            # Key tensor
            k_bytes = data[offset : offset + tensor_bytes]
            offset += tensor_bytes
            # Value tensor
            v_bytes = data[offset : offset + tensor_bytes]
            offset += tensor_bytes

            k_tensor = np.frombuffer(k_bytes, dtype=np_dtype).reshape(num_heads, token_count, head_dim)
            v_tensor = np.frombuffer(v_bytes, dtype=np_dtype).reshape(num_heads, token_count, head_dim)

            kv_data[layer] = (k_tensor.copy(), v_tensor.copy()) # copy to allow free underlying buffer

        return kv_data, model_config
