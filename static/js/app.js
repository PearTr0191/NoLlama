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
// The last sentence is Muse Glimmer's native reasoning control (its chat
// template defers to a system-prompt 'Reasoning strength:' line, default
// high); other models read it as ordinary prose. Never mention '<think>'
// here: the model mimics the literal tags into its answer text (observed
// on Glimmer 2026-08-13).
const NO_THINK_PROMPT = 'Respond directly and concisely, with no internal reasoning preamble. Reasoning strength: minimal.';

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

// --- Sticky-bottom scroll for the live thinking block while streaming ---
// The DOM is wholesale re-rendered on every token (innerHTML = renderMarkdown),
// which normally destroys a scrollable .think-full and snaps its scroll to 0.
// streamState survives the redraw: `pinned` means "the user is watching the
// live tail", so each redraw re-pins to the bottom; a single scroll-up clears
// the flag, after which redraws preserve the user's relative view ("freed").
const STREAM_THRESHOLD = 32; // px from the bottom == "pinned to the stream"
let streamState = { thinkFull: null, pinned: true, onScroll: null };

function attachThinkScroll(thinkFull) {
    // (Re)bind a scroll listener to the current think-full node. The node is
    // recreated on every innerHTML redraw during streaming i.e. detach from the
    // old one and bind to the new one. `pinned` lives on streamState and is
    // read at redraw time, not captured by the listener closure's creation, so
    // it persists correctly across redraws.
    if (streamState.onScroll && streamState.thinkFull) {
        streamState.thinkFull.removeEventListener('scroll', streamState.onScroll);
    }
    streamState.thinkFull = thinkFull;
    streamState.onScroll = function () {
        const tf = streamState.thinkFull;
        if (!tf) return;
        streamState.pinned = tf.scrollHeight - tf.scrollTop - tf.clientHeight <= STREAM_THRESHOLD;
    };
    thinkFull.addEventListener('scroll', streamState.onScroll, { passive: true });
}

function updateStreamBubble(assistantDiv, fullText) {
    // A scrollable .think-full already exists so only its inner content is swapped, allowing
    // to preserve the user's scrollTop natively — no element recreation,
    // manual restore, thus no fighting the user's wheel. Pinned re-pins to the
    // tail; freed leaves the position untouched.
    const prev = streamState.thinkFull;
    if (prev) {
        const scratch = document.createElement('div');
        scratch.innerHTML = renderMarkdown(fullText, true);
        const newFull = scratch.querySelector('.think-full');
        if (newFull) {
            prev.innerHTML = newFull.innerHTML; // same element; scroll preserved
            const block = prev.closest('.think-block');
            const newBlock = scratch.querySelector('.think-block');
            if (block && newBlock) {
                const preview = block.querySelector('.think-preview');
                const newPreview = newBlock.querySelector('.think-preview');
                if (preview && newPreview) preview.innerHTML = newPreview.innerHTML;
                const header = block.querySelector('.think-header');
                const newHeader = newBlock.querySelector('.think-header');
                if (header && newHeader) header.innerHTML = newHeader.innerHTML;
                block.classList.toggle('collapsed', newBlock.classList.contains('collapsed'));
                block.classList.toggle('streaming', newBlock.classList.contains('streaming'));
                // Just-answer button lives after the think block; sync it below.
            }
            syncAnswerNodes(assistantDiv, scratch);
            if (streamState.pinned) prev.scrollTop = prev.scrollHeight;
        } else {
            // Think block closed — rebuild the whole bubble.
            assistantDiv.innerHTML = renderMarkdown(fullText, true);
            const tf = assistantDiv.querySelector('.think-full');
            if (tf) attachThinkScroll(tf);
        }
    } else {
        // No scrollable think block yet — normal full re-render, then attach
        // the scroll listener the first time a .think-full appears.
        assistantDiv.innerHTML = renderMarkdown(fullText, true);
        const thinkFull = assistantDiv.querySelector('.think-full');
        if (thinkFull) {
            attachThinkScroll(thinkFull);
            if (streamState.pinned) thinkFull.scrollTop = thinkFull.scrollHeight;
        }
    }
    scrollToBottom(); // keep the outer chat at the bottom when viewing the tail
}

// Keep the surviving .think-block, drop everything else in assistantDiv, then
// re-append the answer nodes (and just-answer button) from the scratch render.
function syncAnswerNodes(assistantDiv, scratch) {
    const keepBlock = assistantDiv.querySelector('.think-block');
    Array.from(assistantDiv.children).forEach((c) => { if (c !== keepBlock) c.remove(); });
    Array.from(scratch.children).forEach((c) => {
        if (!c.classList || !c.classList.contains('think-block')) assistantDiv.appendChild(c);
    });
}

function resetStreamState() {
    if (streamState.onScroll && streamState.thinkFull) {
        streamState.thinkFull.removeEventListener('scroll', streamState.onScroll);
    }
    streamState.thinkFull = null;
    streamState.pinned = true;
    streamState.onScroll = null;
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
            { role: 'system', content: NO_THINK_PROMPT },
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
                    { role: 'system', content: NO_THINK_PROMPT },
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
            resetStreamState();
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
                            updateStreamBubble(assistantDiv, fullText);
                        }
                    } catch {}
                }
            }

            assistantDiv.innerHTML = renderMarkdown(fullText, false);
            resetStreamState();
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
        // Skip empty think blocks (no-think mode sometimes emits empty tags)
        if (thinkContent) {
            thinkHtml = renderThinkingBlock(thinkContent, false, thinkExpanded ? '' : 'collapsed');
        }
    } else if (thinkOpen) {
        // Still thinking — show content live
        const thinkContent = thinkOpen[1].trim();
        if (thinkContent) {
            const lines = thinkContent.split('\n');
            if (lines.length > 4) {
                // Enough lines — expandable + just-answer button
                const justAnswerBtn = lines.length > 8
                    ? `<button class="just-answer" data-just-answer>Just answer me, dammit!</button>`
                    : '';
                const cls = thinkExpanded ? '' : 'collapsed';
                thinkHtml = renderThinkingBlock(thinkContent, true, cls) + justAnswerBtn;
            } else {
                // Few lines — show all, no collapse needed
                thinkHtml = `<div class="think-block streaming">
                    <div class="think-header">Thinking...</div>
                    <div class="think-preview">${mdEscapeAndRender(thinkContent)}</div>
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

    // Render the main text as markdown
    return thinkHtml + mdEscapeAndRender(mainText);
}

// Renders the inner HTML for a thinking block's full/preview content.
// Uses the same markdown renderer as the main answer so markdown syntax
// inside thinking (headers, lists, bold, code blocks) is rendered, not
// leaked as raw text.
function renderThinkingBlock(content, streaming, extraClass) {
    const full = mdEscapeAndRender(content);
    const preview = mdEscapeAndRender(content.split('\n').slice(-3).join('\n'));
    const cls = streaming ? 'streaming ' + extraClass : extraClass;
    return `<div class="think-block ${cls}" data-think-toggle>
        <div class="think-header">Thinking... <span class="think-toggle">(click to expand)</span></div>
        <div class="think-full">${full}</div>
        <div class="think-preview">${preview}</div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Markdown rendering (self-contained, no dependencies)
// ---------------------------------------------------------------------------
// Order-of-operations:
//   1. escapeHtml the whole input to guarantee raw model HTML can never execute.
//      All passes below operate on the escaped text.
//   2. Pull fenced code blocks into protected placeholders so the inline
//      bold/italic/code passes below never touch code content (the old
//      renderer mangled `**x**` and `*x*` inside code blocks).
//   3. Block passes, line by line: code blocks, headers, blockquotes, lists,
//      paragraphs. Each line's text goes through mdInline exactly ONCE, inside
//      the block pass — running it on the whole text first and again per line
//      corrupts _ and * inside href/src attributes generated by the first
//      pass, and lets *...* match across newlines, eating `* ` bullet lists.
//   4. Inline passes (mdInline): inline code, images, links (scheme-checked),
//      bold, italic — with code spans and attribute values protected.
function mdEscapeAndRender(text) {
    if (!text) return '';
    let html = escapeHtml(text);
    const codeBlocks = [];
    html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
        const i = codeBlocks.length;
        codeBlocks.push({ lang, code: code.replace(/\n$/, '') });
        return `\u0000CODE${i}\u0000`;
    });
    const lines = html.split('\n');
    const out = [];
    let inList = false, inBlockquote = false, inParagraph = false;
    const closePara = () => { if (inParagraph) { out.push('</p>'); inParagraph = false; } };
    const closeList = () => { if (inList) { out.push(`</${inList}>`); inList = false; } };
    const closeBlockquote = () => { if (inBlockquote) { out.push('</blockquote>'); inBlockquote = false; } };
    for (let i = 0; i < lines.length; i++) {
        const trimmed = lines[i].trim();
        const codeMatch = trimmed.match(/^(\u0000CODE\d+\u0000)$/);
        // Guard: literal NUL+CODEn+NUL in model text has no matching block --
        // fall through to the paragraph path instead of crashing on blk.lang.
        const codeBlk = codeMatch && codeBlocks[+codeMatch[1].match(/\d+/)];
        if (codeBlk) {
            closePara(); closeList(); closeBlockquote();
            out.push(`<pre><code class="language-${escapeHtml(codeBlk.lang || '')}">${codeBlk.code}</code><button class="copy-btn" onclick="copyCode(this)">copy</button></pre>`);
            continue;
        }
        const h = trimmed.match(/^(#{1,6})\s+(.*)$/);
        if (h) { closePara(); closeList(); closeBlockquote(); out.push(`<h${h[1].length}>${mdInline(h[2])}</h${h[1].length}>`); continue; }
        // Input is already HTML-escaped, so markdown "> quote" arrives here
        // as "&gt; quote" -- match the entity, not the raw >.
        const bq = trimmed.match(/^&gt; ?(.*)$/);
        if (bq) { closePara(); closeList(); if (!inBlockquote) { closeBlockquote(); out.push('<blockquote>'); inBlockquote = true; } out.push(mdInline(bq[1])); continue; }
        closeBlockquote();
        const ol = trimmed.match(/^(\d+)\. (.*)$/);
        const ul = trimmed.match(/^[-*+] (.*)$/);
        if (ol || ul) {
            closePara();
            const listType = ol ? 'ol' : 'ul';
            if (inList !== listType) { closeList(); out.push(`<${listType}>`); inList = listType; }
            out.push(`<li>${mdInline((ol ? ol[2] : ul[1]) || '')}</li>`);
            continue;
        }
        closeList();
        if (trimmed === '') { closePara(); closeBlockquote(); continue; }
        if (!inParagraph) { out.push('<p>'); inParagraph = true; } else { out.push(' '); }
        out.push(mdInline(lines[i]));
    }
    closePara(); closeList(); closeBlockquote();
    return out.join('').replace(/\u0000CODE\d+\u0000/g, m => {
        const blk = codeBlocks[+(m.match(/\d+/) || [0])[0]];
        if (!blk) return ''; // literal NUL junk in model text, not ours
        return `<pre><code class="language-${escapeHtml(blk.lang || '')}">${blk.code}</code><button class="copy-btn" onclick="copyCode(this)">copy</button></pre>`;
    });
}

// Apply inline passes (on already-escaped text) to a single LINE of text.
// Called exactly once per line by the block pass in mdEscapeAndRender — never
// run it twice on the same text: a second pass corrupts _ and * inside the
// href/src attributes the first pass generated.
// Code spans and generated tags are pulled into placeholders so the emphasis
// passes can't touch code content (`snake_case`) or URLs (...model_id...).
function mdInline(str) {
    const guarded = [];
    const guard = (html) => { guarded.push(html); return `\u0000G${guarded.length - 1}\u0000`; };
    let s = str;
    s = s.replace(/`([^`\n]+)`/g, (_, c) => guard(`<code>${c}</code>`));
    s = s.replace(/!\[([^\]\n]*)\]\(([^)\n]+)\)/g, (m, a, u) => {
        const url = safeUrl(u);
        return url ? guard(`<img alt="${escapeAttr(a)}" src="${url}">`) : m;
    });
    s = s.replace(/\[([^\]\n]+)\]\(([^)\n]+)\)/g, (m, l, u) => {
        const url = safeUrl(u);
        // Only the opening tag is guarded — emphasis inside link text still works.
        return url ? guard(`<a href="${url}">`) + l + '</a>' : m;
    });
    s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/__([^_\n]+)__/g, '<strong>$1</strong>');
    s = s.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    s = s.replace(/_([^_\n]+)_/g, '<em>$1</em>');
    return s.replace(/\u0000G(\d+)\u0000/g, (m, i) => guarded[+i] !== undefined ? guarded[+i] : '');
}

// escapeHtml (textContent -> innerHTML) escapes & < > but NOT quotes. That is
// fine for text nodes, but attribute values built from model output must have
// quotes escaped too, or `![x" onerror=...](y)` breaks out of the attribute.
function escapeAttr(s) {
    return s.replace(/"/g, '&quot;');
}

// Allow only URL schemes that cannot execute script (plus scheme-less
// relative/fragment URLs). javascript:, data:, vbscript: etc. are rejected;
// the caller then leaves the markdown as plain, already-escaped text.
function safeUrl(u) {
    const url = u.trim();
    const scheme = url.match(/^[a-zA-Z][a-zA-Z0-9+.-]*:/);
    if (scheme && !/^(https?|mailto):$/i.test(scheme[0])) return null;
    return escapeAttr(url);
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
// Exposed for browser-test harnesses (no functional side effects)
window.mdEscapeAndRender = mdEscapeAndRender;
window.renderMarkdown = renderMarkdown;
window.updateStreamBubble = updateStreamBubble;
window.resetStreamState = resetStreamState;

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

    // Show user message. Bypass renderMarkdown: displayHtml is already
    // escaped where needed, and with an attached image it contains a real
    // <img> tag — markdown rendering would escape it and dump the base64
    // src as visible text.
    const userDiv = addMessage('user', '');
    userDiv.innerHTML = displayHtml.replace(/\n/g, '<br>');
    const userMsg = { role: 'user', content: userContent };
    chatHistory.push(userMsg);
    // A send the model never saw (server not ready, network error) must not
    // stay in history: it becomes a stale user turn silently prepended to
    // every later request — two consecutive user messages read as a corrupted
    // transcript to the model (observed on Glimmer: it answers the wrong turn).
    const dropUserMsg = () => {
        const i = chatHistory.lastIndexOf(userMsg);
        if (i >= 0) chatHistory.splice(i, 1);
    };

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
            dropUserMsg();
            const err = await resp.json();
            assistantDiv.innerHTML = `<span style="color:var(--error)">${escapeHtml(err.error?.message || 'Error')}</span>`;
            return;
        }

        // Check if streaming (SSE) or single response
        const contentType = resp.headers.get('content-type') || '';
        if (contentType.includes('text/event-stream')) {
            // Streaming
            let fullText = '';
            resetStreamState();
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
                            updateStreamBubble(assistantDiv, fullText);
                        }
                    } catch {}
                }
            }

            // Re-render with streaming=false to collapse think block
            assistantDiv.innerHTML = renderMarkdown(fullText, false);
            resetStreamState();

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
            // Deliberate cancel: keep the user message — justAnswerMe's retry
            // (and a manual re-ask) still needs it in history.
            assistantDiv.innerHTML += '<br><span style="color:var(--text-dim)">[cancelled]</span>';
        } else {
            dropUserMsg();
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
        // Attach-image-first leaves focus on the attach button (or wherever
        // the paste happened) — keystrokes then go nowhere and the input
        // feels dead. Hand focus to the box the user will type in next.
        input.focus();
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
    // Images copied from web pages often arrive as a text/html flavor with a
    // data: URL and no image/ item — without this, the whole base64 string
    // lands in the input box as text.
    const dataUrl = (e.clipboardData.getData('text/html') + ' ' +
                     e.clipboardData.getData('text/plain'))
        .match(/data:image\/(?:png|jpe?g|gif|webp);base64,[A-Za-z0-9+/=]+/);
    if (dataUrl) {
        e.preventDefault();
        attachedImage = dataUrl[0];
        previewImg.src = attachedImage;
        imagePreview.style.display = 'block';
        input.focus();
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
