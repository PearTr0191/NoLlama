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
    if (ch === '/' && next !== '*' && !/\s/.test(next)) {
      // regex literal unless the previous char suggests a division operand,
      // or the slash is followed by whitespace/EOF (can't be a regex body).
      // NB: a regex literal requires a non-whitespace body char after '/',
      // so `/` followed by \n, \r, or space is treated as division. This
      // stops the scanner from eating into inline /** */ doclet comments that
      // follow a `;` (where prev fails the division test) when the doclet's
      // `*` was the next char (handled above) — here we guard the bare `/`.
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

// --- table edge cases ---
(() => {
  // code span in a cell keeps its underscores (mdInline-once + guard)
  const codeCell = mdEscapeAndRender('| Var |\n|---|\n| `snake_case` |');
  has('code span in cell', codeCell, '<td><code>snake_case</code></td>');

  // javascript: link inside a cell is NOT turned into a clickable href
  const xssCell = mdEscapeAndRender('| Bad |\n|---|\n| [click](javascript:alert(1)) |');
  notHas('xss link in cell has no href', xssCell, '<a href="javascript');

  // attribute injection via the IMAGE alt is neutralised
  const attrCell = mdEscapeAndRender('| Bad |\n|---|\n| ![x" onerror="alert(1)](y.png) |');
  notHas('attr-injection img neutralised', attrCell, ' onerror="alert');

  // a header immediately before a table (no blank line) still closes the header
  const afterHeader = mdEscapeAndRender('# Title\n| a |\n|---|');
  has('header then table', afterHeader, '<h1>Title</h1><table>');

  // separator with no body rows still closes <tbody>
  const headerOnly = mdEscapeAndRender('| a |\n|---|');
  has('header-only table closes tbody', headerOnly, '<thead><tr><th>a</th></tr></thead><tbody></tbody></table>');

      // empty cells round-trip (a/b land in thead/tbody; empty cell => <td></td>)
  const emptyCell = mdEscapeAndRender('| a | |\n|---|---|\n| b | |');
  has('empty cells render', emptyCell, '<th>a</th><th></th></tr></thead><tbody><tr><td>b</td><td></td></tr>');

  // alignment colons (left / right / center) are accepted and still render
  const align = mdEscapeAndRender('| a | b |\n|:--|--:|\n| 1 | 2 |');
  has('align colons header', align, '<th>a</th><th>b</th>');
  has('align colons body', align, '<td>1</td><td>2</td>');

  // splitTableRow robustness: a pipe inside a cell's content must NOT be
  // treated as a column delimiter. Otherwise benchmark tables of SQL or
  // filter expressions split on every | and render garbage.
  const sqlCell = mdEscapeAndRender('| sql |\n|---|\n| `select a | b from t` |');
  has('pipe in code span stays in one cell', sqlCell, '<td><code>select a | b from t</code></td>');

  const linkCell = mdEscapeAndRender('| m |\n|---|\n| [x](http://a.com/p|q) |');
  has('pipe in link URL stays in one cell', linkCell, '<td><a href="http://a.com/p|q">x</a></td>');

  // splitTableRow: an unclosed `[` (no matching `)`) must NOT keep inLink set
  // for the rest of the row. The row must still split on the remaining `|` so
  // later cells are not swallowed into one text blob. Before the fix, a `[`
  // with no `)` left inLink=true and ate every subsequent `|`, collapsing the
  // whole row tail into a single cell.
  const unclosedBracket = mdEscapeAndRender('| a | b | c |\n|---|---|---|\n| 1 | [unclosed | 3 |');
  eq('unclosed bracket splits rest of row',
    (unclosedBracket.match(/<tbody><tr><td>1<\/td>/, unclosedBracket) ? 1 : 0), 1);
  has('unclosed bracket row has 3 cells', unclosedBracket, '<td>1</td><td>[unclosed</td><td>3</td>');

  // consecutive tables separated by a blank line render as two tables
  const twoTables = mdEscapeAndRender('| a |\n|---|\n| 1 |\n\n| b |\n|---|\n| 2 |');
  eq('two tables with blank line', (twoTables.match(/<table>/g) || []).length, 2);

  // streaming: a table assembles from partial tokens. The streaming path
  // calls mdEscapeAndRender on the accumulated text, so exercising the
  // block pass on partial inputs covers it.
  notHas('partial header only no table',
    mdEscapeAndRender('| Model | tok/s |'), '<table>');
  has('partial with separator forms table',
    mdEscapeAndRender('| Model | tok/s |\n|---|---|'), '<table>');
  has('full table after body row arrives',
    mdEscapeAndRender('| Model | tok/s |\n|---|---|\n| Qwen | 12 |'),
    '<tr><td>Qwen</td><td>12</td></tr>');

  // blockquote, then blank line, then table
  has('blockquote before table',
    mdEscapeAndRender('> intro\n\n| a |\n|---|\n| 1 |'),
    '<blockquote>intro</blockquote>');

  // img alt attribute injection in a cell: the inner " must be neutralised
  // so the alt attribute remains closed. safeUrl + escapeAttr handle this.
  const imgCell = mdEscapeAndRender('| Bad |\n|---|\n| ![x" onerror="y](a.png) |');
  notHas('img alt injection neutralised', imgCell, ' onerror="y"');
  has('img alt injection safe &quot;', imgCell, '&quot; onerror=&quot;y');

  // ragged table: header has more cells than body row -- browsers handle
  // this, the renderer must not crash or pad cells.
  const ragged = mdEscapeAndRender('| a | b | c |\n|---|---|---|\n| 1 | 2 |');
  eq('ragged 3 th', (ragged.match(/<th>/g) || []).length, 3);
  eq('ragged 2 td', (ragged.match(/<td>/g) || []).length, 2);

  // large table does not crash and produces the expected row count
  let bigBody = '| col |\n|---|\n';
  for (let i = 0; i < 200; i++) bigBody += `| row${i} |\n`;
  const big = mdEscapeAndRender(bigBody);
  eq('large table row count', (big.match(/<tr>/g) || []).length, 201);
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
