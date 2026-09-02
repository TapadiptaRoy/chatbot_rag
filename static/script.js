const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');
const docList = document.getElementById('doc-list');
const docCount = document.getElementById('doc-count');
const conversation = document.getElementById('conversation');
const queryForm = document.getElementById('query-form');
const questionInput = document.getElementById('question-input');
const askButton = document.getElementById('ask-button');

// ---------- Documents ----------

async function loadDocuments() {
  const res = await fetch('/documents');
  const docs = await res.json();

  docCount.textContent = docs.length;

  if (docs.length === 0) {
    docList.innerHTML = '<li class="doc-empty">Nothing indexed yet.</li>';
    return;
  }

  docList.innerHTML = '';
  docs.forEach(doc => {
    const li = document.createElement('li');
    li.className = 'doc-item';
    li.innerHTML = `
      <span class="doc-name" title="${escapeHtml(doc.source)}">${escapeHtml(doc.source)}</span>
      <span class="doc-meta">${doc.chunks}c</span>
      <button class="doc-remove" title="Remove" data-id="${doc.doc_id}">&times;</button>
    `;
    docList.appendChild(li);
  });

  docList.querySelectorAll('.doc-remove').forEach(btn => {
    btn.addEventListener('click', async () => {
      await fetch(`/documents/${btn.dataset.id}/delete`, { method: 'POST' });
      loadDocuments();
    });
  });
}

fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file) return;

  uploadStatus.textContent = `Indexing ${file.name}…`;
  uploadStatus.className = 'upload-status';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      uploadStatus.textContent = data.error || 'Upload failed.';
      uploadStatus.className = 'upload-status err';
    } else {
      uploadStatus.textContent = `Added — ${data.chunks_indexed} chunks indexed.`;
      uploadStatus.className = 'upload-status ok';
      loadDocuments();
    }
  } catch (e) {
    uploadStatus.textContent = 'Upload failed — is the server running?';
    uploadStatus.className = 'upload-status err';
  }

  fileInput.value = '';
});

// ---------- Query ----------

queryForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  const emptyState = conversation.querySelector('.empty-state');
  if (emptyState) emptyState.remove();

  const turn = document.createElement('div');
  turn.className = 'turn';
  turn.innerHTML = `
    <div class="turn-question">${escapeHtml(question)}</div>
    <div class="turn-answer loading">Searching your library…</div>
  `;
  conversation.appendChild(turn);
  conversation.scrollTop = conversation.scrollHeight;

  questionInput.value = '';
  askButton.disabled = true;

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    const answerEl = turn.querySelector('.turn-answer');

    if (!res.ok) {
      answerEl.textContent = data.error || 'Something went wrong.';
      answerEl.classList.remove('loading');
      return;
    }

    answerEl.textContent = data.answer;
    answerEl.classList.remove('loading');

    if (data.sources && data.sources.length) {
      const uniqueSources = [...new Set(data.sources)];
      const sourcesEl = document.createElement('div');
      sourcesEl.className = 'sources';
      sourcesEl.innerHTML = uniqueSources
        .map(s => `<span class="source-tag">${escapeHtml(s)}</span>`)
        .join('');
      turn.appendChild(sourcesEl);
    }
  } catch (err) {
    const answerEl = turn.querySelector('.turn-answer');
    answerEl.textContent = 'Request failed — is the server running?';
    answerEl.classList.remove('loading');
  } finally {
    askButton.disabled = false;
    conversation.scrollTop = conversation.scrollHeight;
  }
});

questionInput.addEventListener('input', () => {
  questionInput.style.height = 'auto';
  questionInput.style.height = questionInput.scrollHeight + 'px';
});

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

loadDocuments();