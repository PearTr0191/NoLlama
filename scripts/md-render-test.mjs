// Node harness for the self-contained markdown renderer in static/js/app.js.
// Extracts the pure functions straight from the shipped file (no build step)
// and runs the table pass plus the regression cases the renderer is known to
// handle. Run from the repo root:  node scripts/md-render-test.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = readFileSync(join(ROOT, 'static', 'js', 'app.js'), 'utf8');

// Extract one `function name(...) { ... }` definition, honouring strings,
// template literals (with ${...} interpolation), regex literals and comments
// so nested braces don't throw the count off.
function extractFunction(name, source) {
  const start = source.indexOf(`function ${name}(`);
  if (start === -1) throw new Error(`function ${name} not found`);
  const openParen = source.indexOf('(', start);
  let depth = 0, i = openParen;
  for (; i < source.length; i++) {
    const ch = source[i];
    if (ch === '(') depth++;
    else if (ch === ')') { depth--; if (depth === 0) break; }
  }
  let k = i + 1;
  while (source[k] === ' ' || source[k] === '\n' || source[k] === '\t' || source[k] === '\r') k++;
  if (source[k] !== '{') throw new Error(`no body for ${name}`);
  let braceDepth = 0;
  let inStr = null, inTpl = false, inLine = false, inBlock = false;
  let interp = 0;      // depth of `${...}` interpolation inside a template
  let prev = '';       // last significant char, for regex-vs-division
  const isIdent = (c) => /[A-Za-z0-9_$]/.test(c);
  while (k < source.length) {
    const ch = source[k], next = source[k + 1];
    if (inLine) { if (ch === '\n') inLine = false; k++; continue; }
    if (inBlock) { if (ch === '*' && next === '/') inBlock = false; k++; continue; }
    if (inStr) {
      if (ch === '\\') k += 2;
      else { if (ch === inStr) inStr = null; k++; }
      continue;
    }
    if (inTpl) {
      if (ch === '\\') k += 2;
      else if (ch === '`') { inTpl = false; prev = '`'; k++; }
      else if (ch === '$' && next === '{') { inTpl = false; interp = 1; k += 2; }
      else k++;
      continue;
    }
    if (interp > 0) {
      if (ch === '{') interp++;
      else if (ch === '}') { interp--; if (interp === 0) inTpl = true; }
      k++;
      continue;
    }
    if (ch === '"' || ch === "'") { inStr = ch; k++; continue; }
    if (ch === '`') { inTpl = true; k++; continue; }
    if (ch === '/' && next === '/') { inLine = true; k += 2; continue; }
    if (ch === '/' && next === '*') { inBlock = true; k += 2; continue; }
    if (ch === '/') {
      // regex literal unless the previous char suggests a division operand
      const division = prev !== '' && (isIdent(prev) || prev === ')' || prev === ']');
      if (division) { prev = '/'; k++; continue; }
      k++; // past opening '/'
      let inClass = false;
      while (k < source.length) {
        const c = source[k];
        if (c === '\\') { k += 2; continue; }
        if (c === '[') inClass = true;
        else if (c === ']') inClass = false;
        else if (c === '/' && !inClass) { k++; break; }
        k++;
      }
      prev = '/';
      continue;
    }
    if (!/\s/.test(ch)) prev = ch;
    if (ch === '{') braceDepth++;
    else if (ch === '}') { braceDepth--; if (braceDepth === 0) return source.slice(start, k + 1); }
    k++;
  }
  throw new Error(`could not match braces for ${name}`);
}

const names = ['escapeHtml', 'escapeAttr', 'safeUrl', 'splitTableRow', 'isTableSeparator', 'mdInline', 'mdEscapeAndRender'];
const extracted = {};
// Function declarations inside a strict-mode eval are scoped to the eval, so
// hand the names out through a shared object instead of relying on leakage.
const code = names.map((n) => extractFunction(n, src)).join('\n')
  + '\n' + names.map((n) => `extracted['${n}'] = ${n};`).join('\n');

// escapeHtml relies on the browser's textContent -> innerHTML behaviour:
// escapes & < > but NOT quotes. Stub just that contract.
globalThis.document = {
  createElement() {
    const el = { _t: '' };
    Object.defineProperty(el, 'textContent', {
      set(v) { el._t = String(v); el.innerHTML = String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); },
      get() { return el._t; },
    });
    return el;
  },
};

eval(code);
const { escapeHtml, escapeAttr, safeUrl, splitTableRow, isTableSeparator, mdInline, mdEscapeAndRender } = extracted;

let pass = 0, fail = 0;
const has = (name, actual, needle) => actual.includes(needle) ? pass++ : (fail++, console.log('FAIL', name, '\n  got  :', JSON.stringify(actual), '\n  need :', needle));
const notHas = (name, actual, needle) => actual.includes(needle) ? (fail++, console.log('FAIL', name, '\n  got  :', JSON.stringify(actual), '\n  extra:', needle)) : pass++;
const eq = (name, actual, expected) => actual === expected ? pass++ : (fail++, console.log('FAIL', name, '\n  got :', JSON.stringify(actual), '\n  want:', JSON.stringify(expected)));

// --- table pass (new) ---
(() => {
  const t = mdEscapeAndRender('| Model | tok/s |\n|---|---|\n| Qwen3-8B | 12.3 |\n| Qwen3-4B | 18.1 |');
  has('table header', t, '<table><thead><tr><th>Model</th><th>tok/s</th></tr></thead><tbody>');
  has('table rows', t, '<tr><td>Qwen3-8B</td><td>12.3</td></tr><tr><td>Qwen3-4B</td><td>18.1</td></tr>');
  has('table close', t, '</tbody></table>');

  const colon = mdEscapeAndRender('| a | b |\n|:---:|:---:|\n| 1 | 2 |');
  has('alignment colons accepted', colon, '<th>a</th><th>b</th>');
  has('alignment colons body', colon, '<td>1</td><td>2</td>');

  const bold = mdEscapeAndRender('| Name |\n|---|\n| **bold** |');
  has('cell via mdInline once', bold, '<td><strong>bold</strong></td>');

  const underscore = mdEscapeAndRender('| Var |\n|---|\n| snake_case |');
  has('underscore in cell preserved', underscore, '<td>snake_case</td>');

  const link = mdEscapeAndRender('| M |\n|---|\n| [hf](https://huggingface.co/q?id=a_b) |');
  has('link cell safe + url intact', link, '<td><a href="https://huggingface.co/q?id=a_b">hf</a></td>');

  const notTable = mdEscapeAndRender('just | a pipe line');
  notHas('lone pipes stay paragraph', notTable, '<table>');
  eq('lone pipes paragraph', notTable, '<p>just | a pipe line</p>');

  const after = mdEscapeAndRender('| x |\n|---|\n| y |\n\nAfter.');
  has('para after table not swallowed', after, '</table><p>After.</p>');

  const inlinePara = mdEscapeAndRender('Heading text\n| A | B |\n|---|--|\n| 1 | 2 |');
  has('para closed before table', inlinePara, '</p><table><thead><tr><th>A</th><th>B</th></tr></thead><tbody>');
  has('para not nested in table', inlinePara, '<td>1</td><td>2</td>');
})();

// --- regression: the original creator's known-good cases ---
(() => {
  has('bold', mdEscapeAndRender('**bold**'), '<strong>bold</strong>');
  has('italic', mdEscapeAndRender('*italic*'), '<em>italic</em>');
  has('header', mdEscapeAndRender('# h1'), '<h1>h1</h1>');
  has('code span', mdEscapeAndRender('`snake_case`'), '<code>snake_case</code>');
  has('fenced code ignores **', mdEscapeAndRender('```py\nx = **y**\n```'), '<pre><code class="language-py">x = **y**</code>');
  has('ordered list', mdEscapeAndRender('1. a\n2. b'), '<ol><li>a</li><li>b</li></ol>');
  has('unordered list', mdEscapeAndRender('* a\n* b'), '<ul><li>a</li><li>b</li></ul>');
  has('plain link', mdEscapeAndRender('[x](https://y)'), '<a href="https://y">x</a>');
  has('emphasis in link text', mdEscapeAndRender('[**bold**](https://y)'), '<a href="https://y"><strong>bold</strong></a>');
  has('blockquote', mdEscapeAndRender('> quote'), '<blockquote>quote</blockquote>');

  // XSS / injection guards
  const xss = mdEscapeAndRender('![x" onerror="alert(1)](y.png)');
  notHas('attribute XSS neutralised', xss, 'onerror="alert(1)"');
  const jsLink = mdEscapeAndRender('[click](javascript:alert(1))');
  notHas('javascript: link rejected', jsLink, '<a href="javascript:');
  const raw = mdEscapeAndRender('<script>alert(1)</script>');
  notHas('raw HTML escaped', raw, '<script>');
})();

console.log(`\nmd-render-test: pass=${pass} fail=${fail}`);
process.exit(fail ? 1 : 0);
