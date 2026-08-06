// NoLlama Web UI

const chat = document.getElementById('chat');
const input = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const modelSelect = document.getElementById('model-select');
const statusDot = document.getElementById('status-dot');
const fileInput = document.getElementById('file-input');
const attachBtn = document.getElementById('attach-btn');
const imagePreview = document.getElementById('image-preview');
const previewImg = document.getElementById('preview-img');
const removeImageBtn = document.getElementById('remove-image');
const dropOverlay = document.getElementById('drop-overlay');
const newChatBtn = document.getElementById('new-chat-btn');
const temperatureSlider = document.getElementById('temperature');
const tempValue = document.getElementById('temp-value');
const noThinkCheckbox = document.getElementById('no-think');

// Temperature slider display
temperatureSlider.addEventListener('input', () => {
    tempValue.textContent = (temperatureSlider.value / 100).toFixed(1);
});

let chatHistory = [];
let attachedImage = null; // base64 data URI
let thinkExpanded = false; // track think block expand state across re-renders
let isGenerating = false;
let abortController = null;

// Send-button doubles as a visible Stop while generating (Escape still works).
function setGenerating(on) {
    isGenerating = on;
    sendBtn.textContent = on ? 'Stop' : 'Send';
    sendBtn.classList.toggle('stop', on);
}

function cancelGeneration() {
    if (abortController) abortController.abort();
    fetch('/v1/cancel', { method: 'POST' }).catch(() => {});
}

function shouldAutoScroll() {
    return chat.scrollHeight - chat.scrollTop - chat.clientHeight < 80;
}

function scrollToBottom() {
    if (shouldAutoScroll()) chat.scrollTop = chat.scrollHeight;
}

// --- Init ---

async function init() {
    await checkHealth();
    await loadModels();
    setInterval(checkHealth, 15000);
    input.focus();
}

async function checkHealth() {
    try {
        const resp = await fetch('/health');
        const data = await resp.json();
        statusDot.className = 'status-dot ' + data.status;
        statusDot.title = data.status;
    } catch {
        statusDot.className = 'status-dot error';
        statusDot.title = 'disconnected';
    }
}

async function loadModels() {
    try {
        const resp = await fetch('/v1/models');
        const data = await resp.json();
        modelSelect.innerHTML = '';
        for (const m of data.data) {
            const opt = document.createElement('option');
            opt.value = m.id;  // e.g. "qwen3-8b-int4-cw@NPU"
            // Display as "qwen3-8b-int4-cw (NPU)"
            const parts = m.id.split('@');
            const name = parts[0];
            const device = parts[1] || m.owned_by.replace('local-', '').toUpperCase();
            opt.textContent = `${name} (${device})`;
            modelSelect.appendChild(opt);
        }
    } catch {}
}

// --- Request builder ---

function buildRequestBody(overrides) {
    const temp = temperatureSlider.value / 100;
    const noThink = noThinkCheckbox.checked;

    let messages = [...chatHistory];
    if (noThink) {
        // Prepend no-think system prompt
        messages = [
            { role: 'system', content: 'Respond directly and concisely. Do not use <think> blocks or internal reasoning.' },
            ...messages.filter(m => m.role !== 'system'),
        ];
    }

    const body = {
        model: modelSelect.value,
        messages: messages,
        stream: true,
        max_tokens: 16384,
        // The web UI is the "does it work" surface, and a thinking-loop on a
        // slow iGPU reads as "it doesn't". Ollama's 1.1 breaks loops faster
        // than the server default (1.05, kept mild for coding agents).
        repetition_penalty: 1.1,
    };

    if (temp > 0) {
        body.temperature = temp;
    }

    return { ...body, ...overrides };
}

// --- Just answer me, dammit! ---

async function justAnswerMe(event) {
    event.stopPropagation();

    // Abort current generation and tell server to stop
    if (abortController) {
        abortController.abort();
    }
    fetch('/v1/cancel', { method: 'POST' }).catch(() => {});

    // Find the last user message
    let lastUserMsg = null;
    for (let i = chatHistory.length - 1; i >= 0; i--) {
        if (chatHistory[i].role === 'user') {
            lastUserMsg = chatHistory[i];
            break;
        }
    }
    if (!lastUserMsg) return;

    // Remove the aborted assistant message from history (it was never complete)
    // The DOM bubble will stay but we'll add a new one below

    // Wait for abort to settle
    await new Promise(r => setTimeout(r, 100));

    // Mark the old bubble as cancelled
    const lastBubble = chat.querySelector('.message.assistant:last-child');
    if (lastBubble) {
        const meta = lastBubble.querySelector('.meta');
        if (!meta) {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'meta';
            metaDiv.innerHTML = '<span style="color:var(--text-dim)">[retrying without thinking]</span>';
            lastBubble.appendChild(metaDiv);
        }
    }

    // Create new assistant bubble and send
    const assistantDiv = addMessage('assistant', '');
    assistantDiv.innerHTML = '<span class="typing-indicator"></span>';
    setGenerating(true);
    const t0 = performance.now();

    try {
        abortController = new AbortController();
        const resp = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: abortController.signal,
            body: JSON.stringify(buildRequestBody({
                messages: [
                    { role: 'system', content: 'Respond directly and concisely. Do not use <think> blocks or internal reasoning.' },
                    ...chatHistory.filter(m => m.role !== 'system'),
                ],
            })),
        });

        const device = resp.headers.get('X-Device') || '';

        if (!resp.ok) {
            const err = await resp.json();
            assistantDiv.innerHTML = `<span style="color:var(--error)">${escapeHtml(err.error?.message || 'Error')}</span>`;
            return;
        }

        const contentType = resp.headers.get('content-type') || '';
        if (contentType.includes('text/event-stream')) {
            let fullText = '';
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const data = line.slice(6);
                    if (data === '[DONE]') continue;
                    try {
                        const chunk = JSON.parse(data);
                        const delta = chunk.choices?.[0]?.delta?.content;
                        if (delta) {
                            fullText += delta;
                            assistantDiv.innerHTML = renderMarkdown(fullText, true);
                            scrollToBottom();
                        }
                    } catch {}
                }
            }

            assistantDiv.innerHTML = renderMarkdown(fullText, false);
            const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
            const metaDiv = document.createElement('div');
            metaDiv.className = 'meta';
            metaDiv.innerHTML = device
                ? `<span class="device-tag">${device}</span> ${elapsed}s (no-think)`
                : `${elapsed}s (no-think)`;
            assistantDiv.appendChild(metaDiv);
            chatHistory.push({ role: 'assistant', content: fullText });
        } else {
            const data = await resp.json();
            const text = data.choices?.[0]?.message?.content || '';
            assistantDiv.innerHTML = renderMarkdown(text, false);
            const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
            const metaDiv = document.createElement('div');
            metaDiv.className = 'meta';
            metaDiv.innerHTML = device
                ? `<span class="device-tag">${device}</span> ${elapsed}s (no-think)`
                : `${elapsed}s (no-think)`;
            assistantDiv.appendChild(metaDiv);
            chatHistory.push({ role: 'assistant', content: text });
        }
    } catch (err) {
        if (err.name !== 'AbortError') {
            assistantDiv.innerHTML = `<span style="color:var(--error)">${escapeHtml(err.message)}</span>`;
        }
    } finally {
        setGenerating(false);
        abortController = null;
        input.focus();
    }
}
window.justAnswerMe = justAnswerMe;

// --- Chat ---

function addMessage(role, content, meta) {
    const div = document.createElement('div');
    div.className = `message ${role}`;

    if (typeof content === 'string') {
        div.innerHTML = renderMarkdown(content);
    } else {
        div.innerHTML = content;
    }

    if (meta) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'meta';
        metaDiv.innerHTML = meta;
        div.appendChild(metaDiv);
    }

    chat.appendChild(div);
    scrollToBottom();
    return div;
}

function renderMarkdown(text, isStreaming) {
    // Handle <think>...</think> blocks BEFORE escaping HTML
    // These are raw model output tags, not user HTML
    let thinkHtml = '';
    let mainText = text;

    // Complete: <think>...</think> followed by the actual answer
    let thinkMatch = text.match(/^<think>([\s\S]*?)<\/think>\s*([\s\S]*)$/);
    // Partial: <think> started but no closing tag yet (streaming)
    let thinkOpen = !thinkMatch && text.match(/^<think>([\s\S]*)$/);
    // Very early: just the opening tag arriving character by character
    let thinkStarting = !thinkMatch && !thinkOpen && /^<(?:t(?:h(?:i(?:n(?:k)?)?)?)?)?$/.test(text.trim());

    if (thinkMatch) {
        const thinkContent = thinkMatch[1].trim();
        mainText = thinkMatch[2].trim();
        // Skip empty think blocks (no-think mode sometimes emits <think></think>)
        if (thinkContent) {
            const lines = thinkContent.split('\n');
            const preview = lines.slice(-3).join('\n');
            const cls = thinkExpanded ? '' : 'collapsed';
            thinkHtml = `<div class="think-block ${cls}" data-think-toggle>
                <div class="think-header">Thinking... <span class="think-toggle">(click to expand)</span></div>
                <div class="think-full">${escapeHtml(thinkContent).replace(/\n/g, '<br>')}</div>
                <div class="think-preview">${escapeHtml(preview).replace(/\n/g, '<br>')}</div>
            </div>`;
        }
    } else if (thinkOpen) {
        // Still thinking — show content live
        const thinkContent = thinkOpen[1].trim();
        if (thinkContent) {
            const lines = thinkContent.split('\n');
            if (lines.length > 4) {
                // Enough lines — expandable + just-answer button
                const preview = lines.slice(-4).join('\n');
                const justAnswerBtn = lines.length > 8
                    ? `<button class="just-answer" data-just-answer>Just answer me, dammit!</button>`
                    : '';
                const cls = thinkExpanded ? '' : 'collapsed';
                thinkHtml = `<div class="think-block streaming ${cls}" data-think-toggle>
                    <div class="think-header">Thinking... <span class="think-toggle">(click to expand)</span></div>
                    <div class="think-full">${escapeHtml(thinkContent).replace(/\n/g, '<br>')}</div>
                    <div class="think-preview">${escapeHtml(preview).replace(/\n/g, '<br>')}</div>
                    ${justAnswerBtn}
                </div>`;
            } else {
                // Few lines — show all, no collapse needed
                thinkHtml = `<div class="think-block streaming">
                    <div class="think-header">Thinking...</div>
                    <div class="think-preview">${escapeHtml(thinkContent).replace(/\n/g, '<br>')}</div>
                </div>`;
            }
        } else {
            thinkHtml = `<div class="think-block streaming">
                <div class="think-header">Thinking...</div>
            </div>`;
        }
        mainText = '';
    } else if (thinkStarting && isStreaming) {
        // Partial <think> tag still arriving
        thinkHtml = `<div class="think-block streaming">
            <div class="think-header">Thinking...</div>
        </div>`;
        mainText = '';
    }

    // Render the main text as markdown (escape HTML first)
    let html = escapeHtml(mainText);

    // Code blocks: ```...```
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code>${code.trim()}</code><button class="copy-btn" onclick="copyCode(this)">copy</button></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Line breaks
    html = html.replace(/\n/g, '<br>');

    return thinkHtml + html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function copyCode(btn) {
    const code = btn.parentElement.querySelector('code').textContent;
    navigator.clipboard.writeText(code);
    btn.textContent = 'copied';
    setTimeout(() => btn.textContent = 'copy', 1500);
}
// Make copyCode available globally
window.copyCode = copyCode;

async function sendMessage() {
    const text = input.value.trim();
    if (!text && !attachedImage) return;
    if (isGenerating) return;
    thinkExpanded = false; // reset for new message

    // Build user message content
    let userContent;
    let displayHtml = '';

    if (attachedImage) {
        userContent = [];
        if (text) userContent.push({ type: 'text', text: text });
        userContent.push({ type: 'image_url', image_url: { url: attachedImage } });
        displayHtml = escapeHtml(text);
        displayHtml += `<img src="${attachedImage}" alt="attached">`;
    } else {
        userContent = text;
        displayHtml = escapeHtml(text);
    }

    // Show user message
    addMessage('user', displayHtml.replace(/\n/g, '<br>'));
    chatHistory.push({ role: 'user', content: userContent });

    // Clear input
    input.value = '';
    input.style.height = 'auto';
    clearImage();

    // Create assistant bubble with waiting indicator
    const assistantDiv = addMessage('assistant', '');
    assistantDiv.innerHTML = '<span class="typing-indicator"></span>';
    setGenerating(true);
    const t0 = performance.now();

    // After 3s with no response, check if a model is reloading and show that
    const reloadCheckTimer = setTimeout(async () => {
        try {
            const r = await fetch('/health');
            const data = await r.json();
            const reloading = Object.values(data.devices || {}).some(
                d => d.status === 'loading' || d.status === 'warming_up'
            );
            if (reloading && isGenerating) {
                assistantDiv.innerHTML =
                    '<span class="typing-indicator"></span> ' +
                    '<span style="color:var(--text-dim)">Reloading model…</span>';
            }
        } catch {}
    }, 3000);

    try {
        abortController = new AbortController();
        const resp = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: abortController.signal,
            body: JSON.stringify(buildRequestBody()),
        });
        clearTimeout(reloadCheckTimer);

        const device = resp.headers.get('X-Device') || '';
        const model = resp.headers.get('X-Model') || '';

        if (!resp.ok) {
            const err = await resp.json();
            assistantDiv.innerHTML = `<span style="color:var(--error)">${escapeHtml(err.error?.message || 'Error')}</span>`;
            return;
        }

        // Check if streaming (SSE) or single response
        const contentType = resp.headers.get('content-type') || '';
        if (contentType.includes('text/event-stream')) {
            // Streaming
            let fullText = '';
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                const lines = buffer.split('\n');
                buffer = lines.pop(); // keep incomplete line

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const data = line.slice(6);
                    if (data === '[DONE]') continue;
                    try {
                        const chunk = JSON.parse(data);
                        const delta = chunk.choices?.[0]?.delta?.content;
                        if (delta) {
                            fullText += delta;
                            assistantDiv.innerHTML = renderMarkdown(fullText, true);
                            scrollToBottom();
                        }
                    } catch {}
                }
            }

            // Re-render with streaming=false to collapse think block
            assistantDiv.innerHTML = renderMarkdown(fullText, false);

            const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
            const metaHtml = device
                ? `<span class="device-tag">${device}</span> ${elapsed}s`
                : `${elapsed}s`;
            const metaDiv = document.createElement('div');
            metaDiv.className = 'meta';
            metaDiv.innerHTML = metaHtml;
            assistantDiv.appendChild(metaDiv);

            chatHistory.push({ role: 'assistant', content: fullText });
        } else {
            // Non-streaming (VLM)
            const data = await resp.json();
            const text = data.choices?.[0]?.message?.content || '';
            const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

            assistantDiv.innerHTML = renderMarkdown(text);
            const metaHtml = device
                ? `<span class="device-tag">${device}</span> ${elapsed}s`
                : `${elapsed}s`;
            const metaDiv = document.createElement('div');
            metaDiv.className = 'meta';
            metaDiv.innerHTML = metaHtml;
            assistantDiv.appendChild(metaDiv);

            chatHistory.push({ role: 'assistant', content: text });
        }
    } catch (err) {
        if (err.name === 'AbortError') {
            assistantDiv.innerHTML += '<br><span style="color:var(--text-dim)">[cancelled]</span>';
        } else {
            assistantDiv.innerHTML = `<span style="color:var(--error)">${escapeHtml(err.message)}</span>`;
        }
    } finally {
        setGenerating(false);
        abortController = null;
        input.focus();
    }
}

function newChat() {
    chatHistory = [];
    chat.innerHTML = '';
    input.focus();
}

// --- Image handling ---

function attachImage(file) {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = () => {
        attachedImage = reader.result;
        previewImg.src = attachedImage;
        imagePreview.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    attachedImage = null;
    previewImg.src = '';
    imagePreview.style.display = 'none';
}

// --- Event listeners ---

// Event delegation for think blocks (survives DOM re-renders)
chat.addEventListener('click', (e) => {
    // "Just answer me" button
    if (e.target.closest('[data-just-answer]')) {
        e.stopPropagation();
        justAnswerMe(e);
        return;
    }
    // Think block expand/collapse
    const thinkBlock = e.target.closest('[data-think-toggle]');
    if (thinkBlock) {
        thinkExpanded = !thinkExpanded;
        thinkBlock.classList.toggle('collapsed');
    }
});

// Send
sendBtn.addEventListener('click', () => {
    if (isGenerating) cancelGeneration();
    else sendMessage();
});

// No-think defaults ON (slow devices + thinking models = runaway loops);
// the user's choice sticks across sessions.
noThinkCheckbox.checked = localStorage.getItem('nollama-no-think') !== 'off';
noThinkCheckbox.addEventListener('change', () => {
    localStorage.setItem('nollama-no-think', noThinkCheckbox.checked ? 'on' : 'off');
});
input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea
input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 150) + 'px';
});

// New chat
newChatBtn.addEventListener('click', newChat);

// File attach
attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) attachImage(fileInput.files[0]);
    fileInput.value = '';
});
removeImageBtn.addEventListener('click', clearImage);

// Paste image
document.addEventListener('paste', (e) => {
    for (const item of e.clipboardData.items) {
        if (item.type.startsWith('image/')) {
            e.preventDefault();
            attachImage(item.getAsFile());
            return;
        }
    }
});

// Drag and drop
let dragCounter = 0;
document.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragCounter++;
    if (e.dataTransfer.types.includes('Files')) {
        dropOverlay.classList.add('active');
    }
});
document.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter <= 0) {
        dropOverlay.classList.remove('active');
        dragCounter = 0;
    }
});
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => {
    e.preventDefault();
    dropOverlay.classList.remove('active');
    dragCounter = 0;
    const file = e.dataTransfer.files[0];
    if (file) attachImage(file);
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'n') {
        e.preventDefault();
        newChat();
    }
    if (e.key === 'Escape' && isGenerating && abortController) {
        abortController.abort();
        fetch('/v1/cancel', { method: 'POST' }).catch(() => {});
    }
});

// --- Start ---
init();
