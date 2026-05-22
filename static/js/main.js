/* global state */
const state = {
  history: [],       // [{role, content}]
  pendingImg: null,  // {b64, mediaType, dataUrl}
  streaming: false,
};

/* elements */
const chatArea   = document.getElementById('chat-area');
const welcome    = document.getElementById('welcome');
const messages   = document.getElementById('messages');
const msgInput   = document.getElementById('msg-input');
const sendBtn    = document.getElementById('send-btn');
const fileInput  = document.getElementById('file-input');
const imgPreview = document.getElementById('image-preview');
const imgWrap    = document.getElementById('image-preview-wrap');
const removeImg  = document.getElementById('remove-img');

/* auto-grow textarea */
msgInput.addEventListener('input', () => {
  msgInput.style.height = 'auto';
  msgInput.style.height = msgInput.scrollHeight + 'px';
  sendBtn.disabled = !msgInput.value.trim() && !state.pendingImg;
});

msgInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
});

sendBtn.addEventListener('click', submit);

/* chips */
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    msgInput.value = chip.dataset.prompt;
    msgInput.dispatchEvent(new Event('input'));
    if (!chip.dataset.image) submit();
  });
});

/* file input */
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    const [header, b64] = dataUrl.split(',');
    const mediaType = header.match(/:(.*?);/)[1];
    state.pendingImg = { b64, mediaType, dataUrl };
    imgPreview.src = dataUrl;
    imgWrap.style.display = 'block';
    sendBtn.disabled = false;
  };
  reader.readAsDataURL(file);
  fileInput.value = '';
});

removeImg.addEventListener('click', () => {
  state.pendingImg = null;
  imgWrap.style.display = 'none';
  imgPreview.src = '';
  sendBtn.disabled = !msgInput.value.trim();
});

/* drag-and-drop onto chat area */
chatArea.addEventListener('dragover', (e) => e.preventDefault());
chatArea.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    fileInput.files = e.dataTransfer.files;
    fileInput.dispatchEvent(new Event('change'));
  }
});

/* main submit */
async function submit() {
  const text = msgInput.value.trim();
  if ((!text && !state.pendingImg) || state.streaming) return;

  welcome.style.display = 'none';
  const img = state.pendingImg;
  state.pendingImg = null;
  imgWrap.style.display = 'none';
  imgPreview.src = '';

  msgInput.value = '';
  msgInput.style.height = 'auto';
  sendBtn.disabled = true;
  state.streaming = true;

  /* render user message */
  appendMessage('user', text, img?.dataUrl);

  /* build history snapshot for the request (before appending new user msg) */
  const historySnapshot = state.history.slice();

  /* append user turn to history */
  const userContent = img
    ? [
        { type: 'image', source: { type: 'base64', media_type: img.mediaType, data: img.b64 } },
        { type: 'text', text: text || 'What do you see in this image?' },
      ]
    : text;
  historySnapshot.push({ role: 'user', content: userContent });
  state.history.push({ role: 'user', content: userContent });

  /* render bot bubble */
  const botBubble = appendMessage('bot', '', null, true);
  let reply = '';

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        image_b64: img?.b64 ?? null,
        media_type: img?.mediaType ?? null,
        history: historySnapshot.slice(0, -1), // exclude last user msg; server appends it
      }),
    });

    if (!res.ok) throw new Error(await res.text());

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      for (const line of chunk.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') break;
        try {
          const { token } = JSON.parse(payload);
          reply += token;
          botBubble.innerHTML = renderMarkdown(reply);
          botBubble.classList.add('cursor');
          scrollBottom();
        } catch {}
      }
    }
  } catch (err) {
    botBubble.textContent = `Error: ${err.message}`;
  } finally {
    botBubble.classList.remove('cursor');
    state.history.push({ role: 'assistant', content: reply });
    state.streaming = false;
    sendBtn.disabled = !msgInput.value.trim();
    scrollBottom();
  }
}

/* DOM helpers */
function appendMessage(role, text, imgDataUrl, isBubble = false) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'U' : '✦';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  if (imgDataUrl) {
    const img = document.createElement('img');
    img.src = imgDataUrl;
    bubble.appendChild(img);
  }

  if (text) bubble.innerHTML = renderMarkdown(text);

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  messages.appendChild(wrap);
  scrollBottom();
  return bubble;
}

function scrollBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

/* minimal markdown: code blocks, inline code, bold, italic, newlines */
function renderMarkdown(md) {
  let s = md
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  /* fenced code blocks */
  s = s.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code class="language-${lang}">${code.trimEnd()}</code></pre>`
  );
  /* inline code */
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  /* bold */
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  /* italic */
  s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
  /* newlines */
  s = s.replace(/\n/g, '<br/>');

  return s;
}
