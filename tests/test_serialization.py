import pytest
from kv_cache_tier.serialization import (
    SafetensorsSerializer, 
    RawBinarySerializer,
    LZ4Compressor,
    ZstdCompressor,
    CompressedSerializer
)
from kv_cache_tier.utils.tensor_utils import kv_caches_equal

def test_safetensors_round_trip(model_config, sample_kv_data):
    ser = SafetensorsSerializer()
    data = ser.serialize(sample_kv_data, model_config)
    restored_kv, restored_config = ser.deserialize(data)
    
    assert restored_config.num_layers == model_config.num_layers
    assert kv_caches_equal(sample_kv_data, restored_kv)

def test_raw_binary_round_trip(model_config, sample_kv_data):
    ser = RawBinarySerializer()
    data = ser.serialize(sample_kv_data, model_config)
    restored_kv, restored_config = ser.deserialize(data)
    
    assert restored_config.num_layers == model_config.num_layers
    assert kv_caches_equal(sample_kv_data, restored_kv)

def test_lz4_compression_round_trip():
    comp = LZ4Compressor()
    original = b"A" * 1000
    compressed = comp.compress(original)
    assert len(compressed) < len(original)
    restored = comp.decompress(compressed)
    assert restored == original

def test_compressed_serializer_round_trip(model_config, sample_kv_data):
    ser = CompressedSerializer(RawBinarySerializer(), LZ4Compressor())
    data = ser.serialize(sample_kv_data, model_config)
    restored_kv, _ = ser.deserialize(data)
    assert kv_caches_equal(sample_kv_data, restored_kv)
