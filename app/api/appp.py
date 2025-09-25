// frontend/src/components/DeepfakeCheck.tsx
import React, { useState, ChangeEvent } from "react";
import axios from "axios";

const DeepfakeCheck: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<string>("");

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) setFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await axios.post(
        "http://localhost:8000/api/predict/",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setResult(JSON.stringify(resp.data.result, null, 2));
    } catch (error) {
      setResult("Failed to analyze: " + error);
    }
  };

  return (
    <div>
      <input type="file" accept="image/*,video/*" onChange={handleFileChange} />
      <button onClick={handleSubmit}>Analyze</button>
      <pre>{result}</pre>
    </div>
  );
};

export default DeepfakeCheck;
