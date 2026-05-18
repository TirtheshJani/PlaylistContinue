from typing import TYPE_CHECKING, Any

from playlist_continue.index.base import BaseIndex
from playlist_continue.index.faiss_index import FaissIndex

if TYPE_CHECKING:
    from playlist_continue.index.hnswlib_index import HnswlibIndex

__all__ = ["BaseIndex", "FaissIndex", "HnswlibIndex"]


# WHY: hnswlib lacks a prebuilt wheel for some interpreter versions; defer the import so
# the package still loads when only FAISS is needed.
def __getattr__(name: str) -> Any:
    if name == "HnswlibIndex":
        from playlist_continue.index.hnswlib_index import HnswlibIndex

        return HnswlibIndex
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
