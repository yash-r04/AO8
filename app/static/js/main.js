document.querySelectorAll('.drop-zone').forEach(zone => {
  const targetId = zone.dataset.target;
  const input = document.getElementById(targetId);
  const preview = zone.querySelector('.drop-preview');

  zone.addEventListener('click', () => input.click());

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('dragover');
  });

  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));

  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file, zone, preview, targetId);
  });

  input && input.addEventListener('change', () => {
    if (input.files[0]) handleFile(input.files[0], zone, preview, targetId);
  });
});

function handleFile(file, zone, preview, targetId) {
  const allowedModel = ['.pt', '.pkl'];
  const allowedData  = ['.csv'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  const isModel   = targetId === 'modelInput';
  const allowed   = isModel ? allowedModel : allowedData;

  if (!allowed.includes(ext)) {
    flashZone(zone, 'error');
    return;
  }

  zone.classList.add('has-file');
  if (preview) {
    preview.textContent = `✓ ${file.name} (${formatSize(file.size)})`;
  }
  updateValidation(targetId);
  checkUploadReady();
}

function flashZone(zone, type) {
  zone.style.borderColor = type === 'error' ? 'var(--attack)' : 'var(--accent)';
  setTimeout(() => { zone.style.borderColor = ''; }, 1500);
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + 'B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + 'KB';
  return (bytes / (1024 * 1024)).toFixed(1) + 'MB';
}

//UPLOAD VALIDATION
function updateValidation(inputId) {
  const map = {
    'modelInput':   'vModel',
    'datasetInput': 'vDataset',
  };
  const el = document.getElementById(map[inputId]);
  if (el) el.classList.add('valid');
}

function checkUploadReady() {
  const btn = document.getElementById('uploadBtn');
  if (!btn) return;

  const modelOk   = document.getElementById('modelInput')?.files?.length > 0;
  const datasetOk = document.getElementById('datasetInput')?.files?.length > 0;
  const shapeOk   = document.getElementById('input_shape')?.value?.trim() !== '';
  const classOk   = document.getElementById('num_classes')?.value?.trim() !== '';

  const vConfig = document.getElementById('vConfig');
  if (vConfig && shapeOk && classOk) vConfig.classList.add('valid');

  btn.disabled = !(modelOk && datasetOk && shapeOk && classOk);
}

/* Watch config fields */
['input_shape', 'num_classes'].forEach(id => {
  const el = document.getElementById(id);
  el && el.addEventListener('input', checkUploadReady);
});

/* === ATTACK CARDS TOGGLE === */
document.querySelectorAll('.attack-card').forEach(card => {
  card.addEventListener('click', () => {
    card.classList.toggle('checked');
    updateSelectionSummary();
  });
});

/* === DEFENSE ITEMS TOGGLE === */
document.querySelectorAll('.defense-item').forEach(item => {
  item.addEventListener('click', () => {
    item.classList.toggle('checked');
    updateSelectionSummary();
  });
});

function updateSelectionSummary() {
  const attacks  = [...document.querySelectorAll('.attack-card.checked')]
    .map(c => c.querySelector('.attack-name')?.textContent || '')
    .filter(Boolean);
  const defenses = [...document.querySelectorAll('.defense-item.checked')]
    .map(d => d.querySelector('.defense-name')?.textContent || '')
    .filter(Boolean);

  const selAttacks  = document.getElementById('selAttacks');
  const selDefenses = document.getElementById('selDefenses');
  const selTime     = document.getElementById('selTime');

  if (selAttacks)  selAttacks.textContent  = attacks.length  ? attacks.join(', ')  : 'none';
  if (selDefenses) selDefenses.textContent = defenses.length ? defenses.join(', ') : 'none';

  // Rough time estimate
  if (selTime) {
    let mins = 0;
    if (attacks.includes('FGSM')) mins += 0.5;
    if (attacks.includes('PGD'))  mins += 2;
    if (attacks.includes('C&W'))  mins += 10;
    defenses.forEach(() => mins += 1.5);
    selTime.textContent = mins < 1 ? '~seconds' : `~${Math.round(mins)} min`;
  }
}

/* === BENCHMARK RUN === */
function startBenchmark() {
  const console_ = document.getElementById('console');
  const body = document.getElementById('consoleBody');
  const btn  = document.getElementById('runBtn');

  if (!console_ || !body || !btn) return;

  const attacks = [...document.querySelectorAll('.attack-card.checked')]
    .map(c => c.querySelector('.attack-name')?.textContent)
    .filter(Boolean);

  if (attacks.length === 0) {
    addConsoleLine(body, '✗ select at least one attack before running', 'err');
    console_.style.display = 'block';
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Running...';
  console_.style.display = 'block';
  body.innerHTML = '';

  const lines = [
    '> initializing AO8 benchmark pipeline...',
    '> loading model from upload session...',
    '> loading dataset...',
    '> wrapping model with ART PyTorchClassifier...',
    '> running baseline evaluation on clean data...',
    attacks.includes('FGSM') ? '> running FGSM attack (ε=0.3)...' : null,
    attacks.includes('PGD')  ? '> running PGD attack (20 steps, ε=0.3)...' : null,
    attacks.includes('C&W')  ? '> running C&W attack (this may take a while)...' : null,
    '> applying selected defenses...',
    '> computing robustness score...',
    '> generating report...',
    '> done. redirecting to report →',
  ].filter(Boolean);

  let i = 0;
  const interval = setInterval(() => {
    if (i >= lines.length) {
      clearInterval(interval);
      btn.disabled = false;
      btn.textContent = 'Run benchmark';
      // In production: window.location.href = '/report';
      return;
    }
    addConsoleLine(body, lines[i], i === lines.length - 1 ? '' : 'dim');
    i++;
  }, 600);
}

function addConsoleLine(container, text, cls = '') {
  const div = document.createElement('div');
  div.className = `console-line mono ${cls}`;
  div.textContent = text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

/* === REPORT PAGE — demo populate === */
function populateReport() {
  const data = {
    score:       60,
    clean:       '94.2%',
    fgsm:        '61.8%',
    pgd:         '43.5%',
    cw:          'n/a',
    preClean:    92, preRobust:  74,
    smClean:     91, smRobust:   68,
    advClean:    87, advRobust:  79,
  };

  setText('mClean', data.clean);
  setText('mFgsm',  data.fgsm);
  setText('mPgd',   data.pgd);
  setText('mCw',    data.cw);
  setText('scoreNum', data.score);

  const tag = document.getElementById('scoreTag');
  if (tag) {
    if (data.score >= 75)      tag.textContent = 'robust';
    else if (data.score >= 50) tag.textContent = 'moderate risk';
    else                       tag.textContent = 'high risk';
  }

  // Animate score arc
  const arc = document.getElementById('scoreArc');
  if (arc) {
    const circumference = 326.7;
    const offset = circumference - (data.score / 100) * circumference;
    setTimeout(() => {
      arc.style.transition = 'stroke-dashoffset 1.2s ease';
      arc.setAttribute('stroke-dashoffset', offset);
    }, 300);
  }

  // Bar fills
  setBar('drPreClean',  'drPreCleanVal',  data.preClean);
  setBar('drPreRobust', 'drPreRobustVal', data.preRobust);
  setBar('drSmClean',   'drSmCleanVal',   data.smClean);
  setBar('drSmRobust',  'drSmRobustVal',  data.smRobust);
  setBar('drAdvClean',  'drAdvCleanVal',  data.advClean);
  setBar('drAdvRobust', 'drAdvRobustVal', data.advRobust);

  // Log table
  setText('lFgsmSamples', '10,000');
  setText('lFgsmAcc',     '61.8%');
  setText('lFgsmF1',      '0.58');
  setStatus('lFgsmStatus', 'done');

  setText('lPgdSamples', '10,000');
  setText('lPgdAcc',     '43.5%');
  setText('lPgdF1',      '0.41');
  setStatus('lPgdStatus', 'done');

  setText('lCwSamples', '--');
  setText('lCwAcc',     '--');
  setText('lCwF1',      '--');
  setStatus('lCwStatus', 'skipped');
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setBar(barId, valId, pct) {
  const bar = document.getElementById(barId);
  const val = document.getElementById(valId);
  if (bar) setTimeout(() => { bar.style.width = pct + '%'; }, 400);
  if (val) val.textContent = pct + '%';
}

function setStatus(id, status) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = status;
  el.className = `log-status mono ${status}`;
}

function exportReport(format) {
  alert(`Export as ${format.toUpperCase()} will be available once the pipeline is wired up.`);
}

/* === INIT === */
document.addEventListener('DOMContentLoaded', () => {
  updateSelectionSummary();
  if (document.getElementById('scoreNum')) populateReport();
});
