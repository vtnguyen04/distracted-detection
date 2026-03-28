import numpy as np
import pytest
from src.infrastructure.shared_memory_manager import (
    SharedFrameReader,
    SharedFrameWriter,
    SharedMemoryManager,
    SharedState,
)


def test_shared_state_read_write() -> None:
    """Test atomic read/write on SharedState."""
    state = SharedState()
    state.add_value("fps", "f", 0.0)
    assert state.get("fps") == 0.0
    state.set("fps", 30.5)
    assert state.get("fps") == 30.5
    state.increment("fps", 1.5)
    assert state.get("fps") == 32.0


def test_shared_state_missing_key() -> None:
    """Test accessing missing key raises KeyError."""
    state = SharedState()
    with pytest.raises(KeyError):
        state.get("nonexistent")


def test_shared_frame_writer() -> None:
    """Test reading and writing to SharedFrameWriter."""
    shape = (10, 10, 3)
    writer = SharedFrameWriter(shape=shape, dtype=np.uint8)
    reader = SharedFrameReader(name=writer.name, shape=shape, dtype=np.uint8)
    try:
        test_data = np.ones(shape, dtype=np.uint8) * 128
        writer.write(test_data)
        read_data = reader.read()
        assert read_data.shape == shape
        assert read_data.dtype == np.uint8
        assert np.array_equal(read_data, test_data)
        assert not np.shares_memory(read_data, reader._array)
    finally:
        reader.close()
        writer.close()


def test_shared_memory_manager_context() -> None:
    """Test that SharedMemoryManager cleans up properly."""
    with SharedMemoryManager(frame_shape=(10, 10, 3)) as mgr:
        assert isinstance(mgr.frame_writer, SharedFrameWriter)
        assert isinstance(mgr.state, SharedState)
        assert mgr.state.get("running") == 0
        assert mgr.state.get("fps") == 0
