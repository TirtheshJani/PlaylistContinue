import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

interface RecommendResponse {
  recommendations: number[];
}

function App() {
  const [seedInput, setSeedInput] = useState("");
  const [nRecs, setNRecs] = useState(20);
  const [recommendations, setRecommendations] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const seedTracks = seedInput
        .split(",")
        .map((s) => s.trim())
        .filter((s) => s.length > 0)
        .map((s) => parseInt(s, 10))
        .filter((n) => !isNaN(n));

      const resp = await fetch(`${API_URL}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed_tracks: seedTracks, n: nRecs }),
      });
      if (!resp.ok) {
        throw new Error(`Server error: ${resp.status}`);
      }
      const data: RecommendResponse = await resp.json();
      setRecommendations(data.recommendations);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{ fontFamily: "monospace", maxWidth: 600, margin: "40px auto", padding: "0 16px" }}>
      <h1>PlaylistContinue</h1>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: 12 }}>
          <label htmlFor="seeds">Seed track IDs (comma-separated):</label>
          <br />
          <input
            id="seeds"
            type="text"
            value={seedInput}
            onChange={(e) => setSeedInput(e.target.value)}
            placeholder="e.g. 42, 17, 8"
            style={{ width: "100%", padding: "6px 8px", marginTop: 4, boxSizing: "border-box" }}
          />
        </div>
        <div style={{ marginBottom: 12 }}>
          <label htmlFor="n">Number of recommendations:</label>
          <br />
          <input
            id="n"
            type="number"
            min={1}
            max={500}
            value={nRecs}
            onChange={(e) => setNRecs(Number(e.target.value))}
            style={{ padding: "6px 8px", marginTop: 4 }}
          />
        </div>
        <button type="submit" disabled={loading} style={{ padding: "8px 20px" }}>
          {loading ? "Loading..." : "Get Recommendations"}
        </button>
      </form>

      {error && (
        <p style={{ color: "red", marginTop: 16 }}>Error: {error}</p>
      )}

      {recommendations.length > 0 && (
        <section style={{ marginTop: 24 }}>
          <h2>Recommendations ({recommendations.length})</h2>
          <ol>
            {recommendations.map((trackId, i) => (
              <li key={i}>Track {trackId}</li>
            ))}
          </ol>
        </section>
      )}
    </main>
  );
}

export default App;
