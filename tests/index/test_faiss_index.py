import numpy as np

from playlist_continue.index.faiss_index import FaissIndex


def _unit_vecs(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def test_search_returns_shape():
    vecs = _unit_vecs(20, 16)
    idx = FaissIndex(dim=16)
    idx.build(vecs)
    result = idx.search(vecs[0:1], k=5)
    assert result.shape == (1, 5)


def test_top1_is_self():
    vecs = _unit_vecs(20, 16)
    idx = FaissIndex(dim=16)
    idx.build(vecs)
    result = idx.search(vecs[3:4], k=1)
    assert result[0, 0] == 3


def test_save_and_load_gives_same_results(tmp_path):
    vecs = _unit_vecs(20, 16)
    idx = FaissIndex(dim=16)
    idx.build(vecs)
    path = str(tmp_path / "test.index")
    idx.save(path)
    idx2 = FaissIndex(dim=16)
    idx2.load(path)
    r1 = idx.search(vecs[0:1], k=3)
    r2 = idx2.search(vecs[0:1], k=3)
    np.testing.assert_array_equal(r1, r2)
