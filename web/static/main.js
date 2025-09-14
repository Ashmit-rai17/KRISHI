// web/static/main.js
const form = document.getElementById('uploadForm');
const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const result = document.getElementById('result');

imageInput.addEventListener('change', () => {
  preview.innerHTML = '';
  const f = imageInput.files[0];
  if (!f) return;
  const img = document.createElement('img');
  img.src = URL.createObjectURL(f);
  preview.appendChild(img);
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!imageInput.files[0]) { alert('Please pick an image'); return; }
  result.innerHTML = '<div class="spinner"></div><p>Analyzing...</p>';
  const formData = new FormData();
  formData.append('image', imageInput.files[0]);
  try {
    const resp = await fetch('/predict', { method:'POST', body: formData });
    const data = await resp.json();
    if (resp.ok) {
      result.innerHTML = data.predictions.map(p => 
        `<div class="result-item"><strong>${p.label}</strong> — ${(p.probability*100).toFixed(1)}%</div>`
      ).join('');
    } else {
      result.innerText = JSON.stringify(data);
    }
  } catch (err) {
    result.innerText = 'Error: ' + err.toString();
  }
});
