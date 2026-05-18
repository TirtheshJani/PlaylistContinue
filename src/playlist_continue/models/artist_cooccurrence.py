"""Artist co-occurrence recommender: seed artists -> co-occurring artists -> their tracks."""

from __future__ import annotations

from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.compute as pc

from playlist_continue.models.base import Recommender


class ArtistCooccurrenceRecommender(Recommender):
    """Recommends tracks from artists that frequently co-occur with seed track artists."""

    def __init__(self) -> None:
        self._track_to_artist: dict[int, int] = {}
        self._cooccurrence: dict[int, Counter[int]] = defaultdict(Counter)
        self._artist_tracks: dict[int, list[int]] = defaultdict(list)
        self._global_popularity: Counter[int] = Counter()

    def fit(self, events: pa.Table) -> None:
        # Build track -> artist and artist -> tracks mappings.
        track_ids = events.column("track_id").to_pylist()
        artist_ids = events.column("artist_id").to_pylist()
        for t, a in zip(track_ids, artist_ids, strict=False):
            self._track_to_artist[t] = a
        self._global_popularity = Counter(track_ids)

        # artist -> tracks sorted by popularity descending.
        for t, a in zip(track_ids, artist_ids, strict=False):
            if t not in self._artist_tracks[a]:
                self._artist_tracks[a].append(t)
        for a in self._artist_tracks:
            self._artist_tracks[a].sort(key=lambda t: -self._global_popularity[t])

        # Build artist co-occurrence from sessions.
        session_ids = pc.unique(events.column("session_id")).to_pylist()  # type: ignore[attr-defined]
        for sid in session_ids:
            mask = pc.equal(events.column("session_id"), sid)  # type: ignore[attr-defined]
            session = events.filter(mask)
            session_artists = list(set(session.column("artist_id").to_pylist()))
            for i, a1 in enumerate(session_artists):
                for a2 in session_artists[i + 1 :]:
                    self._cooccurrence[a1][a2] += 1
                    self._cooccurrence[a2][a1] += 1

    def recommend(self, seed_tracks: list[int], n: int = 500) -> list[int]:
        seed_set = set(seed_tracks)
        seed_artists = {self._track_to_artist[t] for t in seed_tracks if t in self._track_to_artist}

        # Aggregate co-occurrence scores for non-seed artists.
        artist_scores: Counter[int] = Counter()
        for a in seed_artists:
            for co_artist, count in self._cooccurrence[a].items():
                if co_artist not in seed_artists:
                    artist_scores[co_artist] += count

        # If no co-occurring artists, fall back to all artists by popularity.
        if not artist_scores:
            all_artists = {a for t, a in self._track_to_artist.items() if t not in seed_set}
            artist_scores = Counter({a: self._global_popularity.get(a, 0) for a in all_artists})

        result: list[int] = []
        for artist, _ in artist_scores.most_common():
            for t in self._artist_tracks.get(artist, []):
                if t not in seed_set:
                    result.append(t)
                    if len(result) >= n:
                        return result

        # If still not enough, fill with globally popular tracks.
        for t, _ in self._global_popularity.most_common():
            if t not in seed_set and t not in result:
                result.append(t)
                if len(result) >= n:
                    break

        return result[:n]
