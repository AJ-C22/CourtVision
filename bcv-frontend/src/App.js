import React, { useState } from "react";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [output, setOutput] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("http://localhost:8000/process-video/", {
      method: "POST",
      body: formData,
    });
    const data = await response.blob();
    const imageUrl = URL.createObjectURL(data);
    setOutput(imageUrl);
  };

  return (
    <div className="App">
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
      {output && <img src={output} alt="Processed Output" />}
    </div>
  );
}

export default App;
