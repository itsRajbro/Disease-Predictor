// ── Navbar scroll effect ──────────────────────────────────────────────────────
window.addEventListener('scroll', () => {
  document.getElementById('navbar').classList.toggle('scrolled', window.scrollY > 20);
});

// ── Mobile menu ───────────────────────────────────────────────────────────────
function toggleMenu() {
  document.getElementById('navMobile').classList.toggle('open');
}

// ── Disease card selection ────────────────────────────────────────────────────
document.querySelectorAll('.disease-card').forEach(card => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.disease-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    card.querySelector('input[type=radio]').checked = true;
  });
});

// ── Upload zone ───────────────────────────────────────────────────────────────
const uploadZone        = document.getElementById('uploadZone');
const imageInput        = document.getElementById('imageInput');
const previewImg        = document.getElementById('previewImg');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const analyzeBtn        = document.getElementById('analyzeBtn');

uploadZone.addEventListener('click', () => imageInput.click());

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
imageInput.addEventListener('change', () => {
  if (imageInput.files[0]) handleFile(imageInput.files[0]);
});

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewImg.style.display = 'block';
    uploadPlaceholder.style.display = 'none';
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

// ── Predict ───────────────────────────────────────────────────────────────────
async function predict() {
  const file    = imageInput.files[0];
  const disease = document.querySelector('input[name="disease"]:checked').value;

  if (!file) { alert('Please upload an image first.'); return; }

  // Show loading state
  document.getElementById('resultEmpty').style.display    = 'none';
  document.getElementById('resultContent').style.display  = 'none';
  document.getElementById('loadingOverlay').style.display = 'flex';
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append('image', file);
  formData.append('disease', disease);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) { alert('Error: ' + data.error); resetLoading(); return; }
    showResult(data, file);
  } catch (err) {
    alert('Something went wrong. Make sure the Flask server is running.');
    resetLoading();
  }
}

function showResult(data, file) {
  document.getElementById('loadingOverlay').style.display = 'none';
  document.getElementById('resultContent').style.display  = 'block';

  // Disease type label
  const typeLabels = {
    cxr:     'Chest X-Ray Analysis',
    malaria: 'Malaria Blood Smear',
    ocular:  'Ocular Disease Scan',
    brain:   'Brain MRI Scan'
  };
  document.getElementById('resultDiseaseType').textContent =
    typeLabels[data.disease_type] || data.disease_type.toUpperCase();

  // Verdict text
  document.getElementById('resultVerdict').textContent = data.prediction;

  // Color coding
  const verdictBlock = document.getElementById('verdictBlock');
  const tag          = document.getElementById('resultTag');
  verdictBlock.classList.remove('danger', 'safe');
  tag.classList.remove('danger', 'safe', 'neutral');

  const normalKeywords = ['normal', 'uninfected', 'no tumor'];
  const isNormal = normalKeywords.some(k => data.prediction.toLowerCase().includes(k));

  if (isNormal) {
    verdictBlock.classList.add('safe');
    tag.classList.add('safe');
    tag.textContent = 'No Disease Detected';
  } else {
    verdictBlock.classList.add('danger');
    tag.classList.add('danger');
    tag.textContent = 'Condition Detected';
  }

  // Confidence bar
  document.getElementById('confidenceValue').textContent = data.confidence + '%';
  document.getElementById('confidenceBar').style.width = '0%';
  setTimeout(() => {
    document.getElementById('confidenceBar').style.width = data.confidence + '%';
  }, 100);

  // Image preview
  const reader = new FileReader();
  reader.onload = e => { document.getElementById('resultPreviewImg').src = e.target.result; };
  reader.readAsDataURL(file);

  const existingHint = document.getElementById('saveHint');
  if (existingHint) existingHint.remove();

  if (!data.saved) {
    const hint = document.createElement('div');
    hint.id = 'saveHint';
    hint.style.cssText = `
      margin-top: 10px; padding: 10px 14px;
      background: #eff6ff; border: 1px solid #bfdbfe;
      border-radius: 8px; font-size: 0.8rem; color: #1e40af;
      display: flex; align-items: center; gap: 8px;
    `;
    hint.innerHTML = '💾 <span><a href="/login" style="font-weight:600;color:#1d4ed8;">Log in</a> or <a href="/signup" style="font-weight:600;color:#1d4ed8;">Sign up</a> to save this scan to your history.</span>';
    document.getElementById('resultContent').appendChild(hint);
  }
  analyzeBtn.disabled = false;

  // Scroll to result on mobile
  if (window.innerWidth < 860) {
    document.getElementById('resultPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function resetLoading() {
  document.getElementById('loadingOverlay').style.display = 'none';
  document.getElementById('resultEmpty').style.display    = 'flex';
  analyzeBtn.disabled = false;
}

function resetTool() {
  imageInput.value = '';
  previewImg.src = '';
  previewImg.style.display = 'none';
  uploadPlaceholder.style.display = 'block';
  analyzeBtn.disabled = true;
  document.getElementById('resultContent').style.display  = 'none';
  document.getElementById('resultEmpty').style.display    = 'flex';
  document.getElementById('confidenceBar').style.width    = '0%';
}
