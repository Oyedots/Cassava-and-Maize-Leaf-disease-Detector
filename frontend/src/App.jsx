import { useState } from "react";
import "./App.css";

function App() {
  const [crop, setCrop] = useState("cassava");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handlePredict = async () => {
    if (!file) {
      setError("Please select a leaf image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/predict/${crop}`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        "Could not connect to the prediction server. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="card">
        <h1>🌿 Cassava & Maize</h1>
        <h2>Leaf Disease Detector</h2>

        <p className="description">
          Upload a cassava or maize leaf image to identify possible diseases.
        </p>

        <label>Choose crop</label>

        <select value={crop} onChange={(e) => setCrop(e.target.value)}>
          <option value="cassava">Cassava</option>
          <option value="maize">Maize</option>
        </select>

        <label>Upload leaf image</label>

        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            setFile(e.target.files[0]);
            setResult(null);
            setError("");
          }}
        />

        {file && (
          <p className="filename">
            Selected: {file.name}
          </p>
        )}

        <button onClick={handlePredict} disabled={loading}>
          {loading ? "Analyzing..." : "Predict Disease"}
        </button>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="result">
            <h3>Prediction Result</h3>

            <p>
              <strong>Crop:</strong> {result.crop}
            </p>

            <p>
              <strong>Prediction:</strong> {result.prediction}
            </p>

            <p>
              <strong>Confidence:</strong>{" "}
              {Number(result.confidence).toFixed(2)}%
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;