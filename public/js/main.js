document.addEventListener("DOMContentLoaded", () => {
  const textarea = document.getElementById("userInput");
  const modeRadios = document.getElementsByName("mode");
  const resultDiv = document.getElementById("analysisResult");
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const themeIcon = document.getElementById("themeIcon");

  // Theme toggle logic
  const savedTheme = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  themeIcon.textContent = savedTheme === 'dark' ? '☀️' : '🌙';

  themeToggleBtn.addEventListener('click', () => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const next = isDark ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    themeIcon.textContent = next === 'dark' ? '☀️' : '🌙';
  });

  // Inference on Enter
  textarea.addEventListener("keydown", async (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();

      const text = textarea.value.trim();
      const selectedMode = Array.from(modeRadios).find(r => r.checked).value;

      if (selectedMode === "analyze" && text.length > 0) {
        try {
          const res = await fetch("http://127.0.0.1:5000/analyze", {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify({ text })
          });

          if (!res.ok) throw new Error("Request failed");
          const data = await res.json();

          const sentimentLabel = data.sentiment.label;
          const sentimentScore = (data.sentiment.score * 100).toFixed(2);
          const topEmotion = data.emotions[0];
          const emotionLabel = topEmotion.emotion;
          const emotionScore = (topEmotion.score * 100).toFixed(2);

          resultDiv.innerHTML = `
            <div class="card">
              <h2>Analysis Results</h2>
              <p><strong>Sentiment:</strong> ${sentimentLabel} (${sentimentScore}%)</p>
              <p><strong>Top Emotion:</strong> ${emotionLabel} (${emotionScore}%)</p>
            </div>
          `;
        } catch (err) {
          console.error("Analysis failed:", err);
          resultDiv.textContent = "Failed to analyze input.";
        }
      } else {
        resultDiv.textContent = "";
      }
    }
  });
});
