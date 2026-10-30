/* concepts tab: registers TABS.concepts = { init, open }. Uses the helpers of core.js.
   Search first: a search box over every concept's name and spellings (D.g.labels), facets as chips (topic, strict, merged,
   size), a fast list of the results beside a reading pane for the open concept. The reading pane shows a concept's
   definitions by paper, what its papers say about it (termOf: every extracted field, merged, with how many papers state each
   item), its family (linked supertypes and subtypes, from relations()), its similar concepts (D.g.cnb/cns), its spellings
   and how they merged, and how alike its mentions are. Everything else (how term merging works, the rule's limits and the
   17 flagged word-order merges, the spread chart, the size chart, the table with every column) sits in expanders below.
   Term merging (the label-match rule, src/mbdef/postprocess/label_match.py) is explained from D.merges, the
   traces of the concepts with two or more spellings: every example and count is read from it when the tab opens. */

GL.spelling = ['Spelling', 'One exact way a label is written in the papers, such as “Fake News” or “fake-news”.', 'Term merging joins the spellings of one term into one concept. A spelling counts every mention written that way.'];
GL.wskey = ['Word-set key', 'The content words of a label, singular, American, sorted: “bias in the media” → “bias media”.', 'Stop words such as “of”, “in”, “the” are left out when they stand between content words (never inside a hyphenated compound), and the words are sorted, so word order does not matter. Spellings with the same word-set key join one concept.'];
GL.spkey = ['Spacing key', 'All the words of a label, singular, American, in order, written together: “click-bait” → “clickbait”.', 'It keeps stop words and word order but ignores spaces and hyphens, so “deep fake”, “deep-fake” and “deepfake” share it. Spellings with the same spacing key join one concept.'];
GL.strictconcept = ['Strict concept', 'A concept with at least one strict mention: one whose score also reaches the stricter best-F1 cut.', ''];
GL.merged = ['Merged concept', 'A concept whose mentions are written in two or more spellings, joined by term merging.', ''];
GL.maintopic = ['Main topic', 'The topic that most of a concept’s mentions fall in; its colour is the topic’s colour on the map.', 'Topics group mentions, not concepts, so the mentions of one concept can fall in several topics. The list colours each concept by the topic of most of its mentions; a tie goes to the topic of its highest-scoring mention.'];

const CO = { q: '', strict: false, merged: false, band: -1, topic: -1, sort: 'n', dir: 1, sel: -1, shown: 0, R: [], allTopics: false };
const BANDS = [[1, 1, '1'], [2, 9, '2–9'], [10, 99, '10–99'], [100, 1e9, '100+']];
const PAGE = 40;
const wide = () => matchMedia('(min-width: 900px)').matches;
const squash = (s) => s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');

/* ------------------------------------------------------------------ the steps of the rule and their colours */
// one entry per kind of change; `cat` picks the colour token --st-<cat> of concepts.css, `no` is the step number
const KIND = {
  unicode: { cat: 'chars', no: '1', name: 'Unicode' }, bracket: { cat: 'bracket', no: '2', name: 'brackets' },
  case: { cat: 'chars', no: '3', name: 'case' }, quotes: { cat: 'chars', no: '3', name: 'quotes' },
  compound: { cat: 'compound', no: '3', name: 'hyphens' }, punct: { cat: 'chars', no: '3', name: 'punctuation' },
  article: { cat: 'article', no: '4', name: 'leading article' }, plural: { cat: 'plural', no: '5', name: 'plural' },
  spelling: { cat: 'spelling', no: '6', name: 'British spelling' }, stop: { cat: 'wordset', no: '7', name: 'stop words' },
  order: { cat: 'wordset', no: '7', name: 'word order' }, spacing: { cat: 'compound', no: '7', name: 'spacing' },
};
const STEP_KINDS = ['unicode', 'bracket', 'case', 'quotes', 'compound', 'punct', 'article', 'plural', 'spelling', 'stop'];
const STOPW = ['the', 'a', 'an', 'of', 'in', 'on', 'for', 'to', 'and', 'by', 'with', 'as', 'at', 'from', 'into', 'within', 'or'];
const cvar = (k) => `--c:var(--st-${KIND[k].cat})`;
const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const wordRe = (w) => new RegExp('(?<![\\p{L}\\p{N}])' + esc(w) + '(?![\\p{L}\\p{N}])', 'giu');
const q = (s) => `“${s}”`;
const plural = (n, one, many) => `${nf(n)} ${n === 1 ? one : many || one + 's'}`;

function kindOf(s) {
  if (s.startsWith('Unicode')) return 'unicode';
  if (s === 'case-folded') return 'case';
  if (s.startsWith('quotes')) return 'quotes';
  if (s.startsWith('hyphen')) return 'compound';
  if (s.startsWith('other punctuation')) return 'punct';
  if (s.startsWith('leading article')) return 'article';
  if (s.startsWith('plural')) return 'plural';
  if (s.startsWith('British')) return 'spelling';
  if (s.startsWith('stop words')) return 'stop';
  return /bracket/.test(s) ? 'bracket' : 'case';
}
function chipText(s) {
  const k = kindOf(s); let m;
  if (k === 'unicode') return 'Unicode normalised';
  if (k === 'case') return 'case-folded';
  if (k === 'quotes') return 'quotes removed';
  if (k === 'compound') return 'hyphen: one compound';
  if (k === 'punct') return 'punctuation removed';
  if (k === 'article' && (m = s.match(/'(.+)'/))) return `${q(m[1])} dropped`;
  if ((k === 'plural' || k === 'spelling') && (m = s.match(/'(.+?)' -> '(.+?)'/))) return `${m[1]} → ${m[2]}`;
  if (k === 'stop') return 'stop words out: ' + [...s.matchAll(/'([^']+)'/g)].map((x) => x[1]).join(', ');
  if ((m = s.match(/^bracket '(.+)' dropped: (.+)$/))) return `${m[1]} dropped: ${m[2].replace(/^an? /, '').replace(/ of '.*'$/, '')}`;
  if ((m = s.match(/^bracket '(.+)' kept as a qualifier/))) return `${m[1]} kept as a qualifier`;
  if ((m = s.match(/^'(.+)' dropped: an acronym of its bracket '(.+)'/))) return `${m[1]} dropped, its bracket ${m[2]} kept`;
  if ((m = s.match(/^prefix in brackets opened: '(.+)'/))) return `${m[1]} opened`;
  return s;
}
function chip(k, text, title) { return H('span', { class: 'co-chip', style: cvar(k), title: title || text }, H('b', { text: KIND[k].no }), text); }
function keyEl(t, key, shared) {
  return H('span', { class: 'co-key' + (t === 's' ? ' sp' : '') + (shared ? ' shared' : ''), title: (t === 's' ? 'spacing key' : 'word-set key') + (shared ? ', shared with another spelling' : ', not shared') },
    H('i', { text: t === 's' ? 'spacing' : 'word set' }), key);
}
// the content words of a label in their written order, read back from its two keys (the spacing key keeps the order,
// the word-set key says which words are content words): "bias media" + "biasinthemedia" -> [bias, media]
function inOrder(ws, sp) {
  const rec = (pos, rem) => {
    if (pos === sp.length) return rem.length ? null : [];
    for (let i = 0; i < rem.length; i++) if (rem[i] && sp.startsWith(rem[i], pos)) { const r = rec(pos + rem[i].length, rem.slice(0, i).concat(rem.slice(i + 1))); if (r) return [rem[i], ...r]; }
    for (const s of STOPW) if (sp.startsWith(s, pos)) { const r = rec(pos + s.length, rem); if (r) return r; }
    return null;
  };
  return rec(0, ws.split(' '));
}
// the word-set joins of a group whose spellings have their content words in different orders: [{key, buckets}]
function orderJoins(v) {
  const out = [];
  for (const [t, key, sps] of v.joined_by) {
    if (t !== 'word-set') continue;
    const B = new Map();
    for (const l of v.labels) if (sps.includes(l[0])) { const o = (inOrder(l[1], l[2]) || [l[1]]).join(' '), b = B.get(o) || { seq: o, n: 0, labels: [] }; b.n += l[4]; b.labels.push(l); B.set(o, b); }
    if (B.size > 1) out.push({ key, buckets: [...B.values()].sort((a, b) => b.n - a.n) });
  }
  return out;
}

/* ------------------------------------------------------------------ counts and worked examples, from D.merges */
function mergeStats() {
  const M = { groups: 0, spellings: 0, k: {}, order: 0, spacing: 0, ex: {}, orderEx: null, spacingEx: null, intro: null };
  for (const k of STEP_KINDS) M.k[k] = { sp: 0, gr: 0 };
  const better = (a, b) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] > b[i]; return false; };
  for (const gk in D.merges) {
    const g = +gk, v = D.merges[gk], L = v.labels;
    M.groups++;
    const ks = L.map((l) => new Set(l[3].map(kindOf))), seen = new Set();
    L.forEach((l, i) => {
      M.spellings++;
      for (const k of ks[i]) {
        M.k[k].sp++; seen.add(k);
        // a good example: another spelling of the concept did not need this step (so the step made the join), few other
        // steps in the way, many mentions
        const partner = L.find((x, j) => j !== i && !ks[j].has(k)), sc = [partner ? 1 : 0, -(ks[i].size - 1), l[4]];
        if (!M.ex[k] || better(sc, M.ex[k].sc)) M.ex[k] = { sc, g, l, partner };
      }
    });
    for (const k of seen) M.k[k].gr++;
    // the intro example: the concept whose spellings needed the most kinds of step, then the most mentions
    const isc = [seen.size, L.reduce((t, l) => t + l[4], 0)];
    if (!M.intro || better(isc, M.intro.sc)) M.intro = { sc: isc, g };
    // joined only through the spacing key: two or more word-set keys in one concept
    const W = new Map(); for (const l of L) { const b = W.get(l[1]) || { n: 0, l }; b.n += l[4]; W.set(l[1], b); }
    if (W.size > 1) {
      M.spacing++;
      // prefer a pair that differs by a space alone ("deep fake" / "deepfake"), then the most mentions in the smaller side
      const b = [...W.values()].sort((a, c) => c.n - a.n), sc = [/[-/–—_]/.test(b[0].l[0] + b[1].l[0]) ? 0 : 1, b[1].n, b[0].n];
      if (!M.spacingEx || better(sc, M.spacingEx.sc)) M.spacingEx = { sc, g, a: b[0].l, b: b[1].l };
    }
    const oj = orderJoins(v);
    if (oj.length) {
      M.order++;
      // prefer a pair where stop words go too ("media bias" / "bias in the media")
      const b = oj[0].buckets, sc = [b[1].labels[0][3].some((x) => kindOf(x) === 'stop') ? 1 : 0, b[1].n, b[0].n];
      if (!M.orderEx || better(sc, M.orderEx.sc)) M.orderEx = { sc, g, a: b[0].labels[0], b: b[1].labels[0], key: oj[0].key };
    }
  }
  return M;
}
// before -> after of one step on one spelling, the changed part marked in the step's colour
function marked(text, ranges, k, cut) {
  const f = H('span', { class: 'bf', style: cvar(k) }); let last = 0;
  for (const [a, b] of ranges) { f.append(text.slice(last, a), H('mark', { class: 'co-mark' + (cut ? ' cut' : ''), text: text.slice(a, b) })); last = b; }
  f.append(text.slice(last)); return f;
}
const rangesOf = (text, re) => [...text.matchAll(re)].filter((m) => m[0]).map((m) => [m.index, m.index + m[0].length]);
function swapped(text, re, to, k) { // text with every match of re replaced by `to`, the replacement marked
  const parts = [], R = rangesOf(text, re); let out = '', last = 0;
  for (const [a, b] of R) { out += text.slice(last, a); parts.push([out.length, out.length + to.length]); out += to; last = b; }
  return marked(out + text.slice(last), parts, k);
}
function beforeAfter(k, l) {
  const [t, ws, sk, steps] = l, s = steps.find((x) => kindOf(x) === k) || '', tidy = (x) => x.replace(/\s+/g, ' ').trim();
  let m;
  if (k === 'case') { const R = rangesOf(t, /\p{Lu}+/gu); return [marked(t, R, k), marked(t.toLowerCase(), t.toLowerCase().length === t.length ? R : [], k)]; }
  if (k === 'quotes') { const re = /["'‘’“”`´]/g; return [marked(t, rangesOf(t, re), k, true), marked(t.replace(re, ''), [], k)]; }
  if (k === 'compound') { const re = /[-/–—_]/g; return [marked(t, rangesOf(t, re), k), swapped(t, re, ' ', k), [H('span', { class: 'ar', text: '→' }), keyEl('s', sk, true)]]; }
  if (k === 'punct') { const re = /[^\p{L}\p{N}\s"'‘’“”`´\-/–—_]/gu; return [marked(t, rangesOf(t, re), k, true), marked(tidy(t.replace(re, ' ')), [], k)]; }
  if (k === 'article' && (m = s.match(/'(.+)'/))) { const re = new RegExp('^\\s*' + esc(m[1]) + '(?=\\s)', 'gi'); return [marked(t, rangesOf(t, re), k, true), marked(tidy(t.replace(re, '')), [], k)]; }
  if ((k === 'plural' || k === 'spelling') && (m = s.match(/'(.+?)' -> '(.+?)'/))) { const re = wordRe(m[1]); return [marked(t, rangesOf(t, re), k), swapped(t, re, m[2], k)]; }
  if (k === 'stop') { const W = [...s.matchAll(/'([^']+)'/g)].map((x) => x[1]), R = W.flatMap((w) => rangesOf(t, wordRe(w))).sort((a, b) => a[0] - b[0]); return [marked(t, R, k, true), marked((inOrder(ws, sk) || [ws]).join(' '), [], k)]; }
  if ((m = s.match(/^bracket '[(\[](.+?)[)\]]' (dropped|kept)/))) {
    const re = new RegExp('\\s*[(\\[]\\s*' + esc(m[1]) + '\\s*[)\\]]', 'g');
    return [marked(t, rangesOf(t, re), k, m[2] === 'dropped'), marked(tidy(t.replace(re, m[2] === 'kept' ? ' ' + m[1] : '')), [], k)];
  }
  if ((m = s.match(/^'(.+)' dropped: an acronym of its bracket '[(\[](.+)[)\]]'/))) return [marked(t, rangesOf(t, new RegExp(esc(m[1]), 'g')).slice(0, 1), k, true), marked(m[2], [], k)];
  if ((m = s.match(/^prefix in brackets opened: '\((.+)\)'/))) { const re = new RegExp('\\(' + esc(m[1]) + '\\)', 'g'); return [marked(t, rangesOf(t, re), k), swapped(t, re, m[1], k)]; }
  return [marked(t, [], k), keyEl('w', ws, false)];
}
const conceptLink = (g, text) => H('button', { class: 'linkbtn', type: 'button', text: text || q(D.g.name[g]), onclick: () => openConcept(g) });
function openHow() { const d = $('#co-how'); if (!d) return; d.open = true; d.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
// a small "▸ title" toggle whose content is built on first open
function lazyMore(title, build) {
  const body = H('div'), d = H('details', { class: 'more co-lm' }, H('summary', { text: title }), body); let done = false;
  d.addEventListener('toggle', () => { if (!d.open || done) return; done = true; try { body.append(build()); } catch (e) { console.error(e); body.append(H('p', { class: 'note', text: 'This could not be drawn: ' + e.message })); } });
  return d;
}

/* ------------------------------------------------------------------ "How term merging works" (an expander) */
function howPanel(M, body) {
  const nG = D.g.n.length;
  body.append(H('p', {}, 'Papers write one term in many ways: ', ...exampleSpellings(M.intro.g), ' Each ', gl('spelling', 'spelling'), ' gets two keys, a ', gl('wskey', 'word-set key'), ' and a ', gl('spkey', 'spacing key'),
    '; spellings that share a key become one concept. The rule reads letters, never meaning: it joins spellings of one term, not synonyms.'));
  body.append(more('How to read the steps', H('p', { text: `Each line is one step, in the order the rule runs, in the colour its changes carry everywhere in this tab; beside it a real spelling the step changed, and how many spellings it changed. The counts cover only the ${nf(M.groups)} concepts with two or more spellings (${nf(M.spellings)} spellings); the other ${nf(nG - M.groups)} concepts have one spelling each, and nothing was traced for them.` })));
  const LINES = [
    { plain: true, src: 'for each spelling s of every label in the corpus:' },
    { k: 'unicode', src: 's = «nfkc»(s)', cm: 'Look-alike Unicode characters get one form: the ligature “ﬁ” becomes “fi”.' },
    { k: 'bracket', src: 's = «drop_bracket»(s)', cm: 'A bracket goes when it is an acronym of the rest, a tag of 1–2 characters, an abbreviation of a word of the label, or another name after “or”, “i.e.”, “aka”. Any other bracket stays as a qualifier, so “affect (negative)” is not “affect”.' },
    { k: 'case', src: 's = «casefold»(s)', cm: 'Upper and lower case no longer differ.' },
    { k: 'quotes', src: 's = «remove_quotes»(s)', cm: 'Quotes and apostrophes go without a space: “liar\'s” becomes “liars”.' },
    { k: 'compound', src: 's = «split_compound»(s)', cm: 'Hyphen, slash, dash and underscore become spaces, but the parts stay marked as one compound.' },
    { k: 'punct', src: 's = «remove_punctuation»(s)', cm: 'Any other punctuation becomes a space.' },
    { plain: true, src: 'words = s.split()' },
    { k: 'article', src: 'words = «drop_leading_article»(words)', cm: 'A leading “the”, “a” or “an” goes, except in “a priori”.' },
    { k: 'plural', src: 'words = [«singular»(w) for w in words]', cm: 'Only when the singular occurs among the corpus labels. News, media, bias and politics never change; irregular plurals (phenomena, criteria) come from a fixed list.' },
    { k: 'spelling', src: 'words = [«american»(w) for w in words]', cm: 'Only when the American form occurs among the labels: -isation, -ise, -our, and judgement, labelling, centre …' },
    { k: 'stop', src: 'content = «drop_stop_words»(words)', cm: 'Stop words (of, in, the, and …) go when they stand between content words, never inside a compound: “in-group bias” keeps “in”.' },
    { k: 'order', src: 'word_set_key = " ".join(«sorted»(content))', cm: 'Sorting makes word order irrelevant.' },
    { k: 'spacing', src: 'spacing_key = «"".join»(words)', cm: 'All words in order, with nothing between them: spaces and hyphens no longer matter.' },
    { k: 'group', src: 'groups = «connected_components»(spellings, linked by a shared word_set_key or spacing_key)', cm: 'Two spellings join when they share either key, directly or through a chain of other spellings.' },
  ];
  const code = H('div', { class: 'co-code', role: 'list', 'aria-label': 'the merging rule, step by step' });
  for (const L of LINES) {
    const src = H('code', { class: 'co-src' });
    L.src.split(/(«[^»]+»)/).forEach((p) => src.append(p.startsWith('«') ? H('span', { class: 'fn', text: p.slice(1, -1) }) : p));
    if (L.cm) src.append(H('span', { class: 'cm', text: L.cm }));
    if (L.plain) { code.append(H('div', { class: 'co-line plain', role: 'listitem' }, H('span', { class: 'co-no', text: '' }), src)); continue; }
    const style = L.k === 'group' ? '--c:var(--accent)' : cvar(L.k);
    code.append(H('div', { class: 'co-line', role: 'listitem', style }, H('span', { class: 'co-no', text: L.k === 'group' ? '⇒' : KIND[L.k].no }), src, lineExample(L.k, M), lineCount(L.k, M)));
  }
  body.append(code);
  // the two keys, drawn for two real concepts
  body.append(H('h4', { class: 'co-sub', text: 'Two keys, one concept' }),
    H('p', { class: 'small' }, 'A key shared by two or more spellings is highlighted; spellings linked through shared keys, directly or in a chain, become one concept. The ', gl('spkey', 'spacing key'), ' (dashed) catches spaces and hyphens; the ', gl('wskey', 'word-set key'), ' (solid) catches stop words and word order.'));
  const two = H('div', { class: 'grid2' });
  if (M.spacingEx) { const e = M.spacingEx; two.append(H('div', { class: 'panel' }, H('div', { class: 'small' }, H('b', { text: 'Joined through the spacing key. ' }), `${q(e.a[0])} and ${q(e.b[0])} have different word-set keys (${q(e.a[1])}, ${q(e.b[1])}) but the same spacing key ${q(e.a[2])}. `, conceptLink(e.g, 'Open this concept')), keyDiagram(e.g, 6))); }
  if (M.orderEx) { const e = M.orderEx; two.append(H('div', { class: 'panel' }, H('div', { class: 'small' }, H('b', { text: 'Joined through the word-set key. ' }), `${q(e.a[0])} and ${q(e.b[0])} have different spacing keys (${q(e.a[2])}, ${q(e.b[2])}) but the same words once stop words go and the words are sorted: ${q(e.key)}. `, conceptLink(e.g, 'Open this concept')), keyDiagram(e.g, 6))); }
  body.append(two, H('p', { class: 'small muted', text: 'Every spelling’s steps above are the ones the label-match rule records when it reads that spelling.' }));
}
function exampleSpellings(g) { // the spellings of one concept, up to five
  const v = D.merges[g], out = [];
  v.labels.slice(0, 5).forEach((l, i, a) => out.push(H('q', { text: l[0] }), i < a.length - 1 ? ', ' : v.labels.length > 5 ? ` and ${nf(v.labels.length - 5)} more.` : '.'));
  return out;
}
function lineExample(k, M) {
  const box = H('div', { class: 'co-ex' });
  if (k === 'order' || k === 'spacing') {
    const e = k === 'order' ? M.orderEx : M.spacingEx; if (!e) return box;
    const key = k === 'order' ? keyEl('w', e.key, true) : keyEl('s', e.a[2], true);
    box.append(H('div', {}, marked(e.a[0], [], k), H('span', { class: 'ar', text: '+' }), marked(e.b[0], [], k), H('span', { class: 'ar', text: '→' }), key),
      H('div', { class: 'small muted' }, 'both in ', conceptLink(e.g), k === 'order' ? ': the same words in another order' : ': written apart in one, together in the other'));
    return box;
  }
  if (k === 'group') {
    const e = M.spacingEx; if (!e) return box;
    const v = D.merges[e.g];
    box.append(H('div', {}, conceptLink(e.g), ` joins ${plural(v.labels.length, 'spelling')} through ${plural(v.joined_by.length, 'shared key')}; see the diagrams below.`));
    return box;
  }
  const e = M.ex[k];
  if (!e) { box.append(H('span', { class: 'muted', text: 'None of the merged spellings needed this step.' })); return box; }
  const [b, a, extra] = beforeAfter(k, e.l);
  box.append(H('div', {}, b, H('span', { class: 'ar', text: '→' }), a, ...(extra || [])),
    H('div', { class: 'small muted' }, 'in ', conceptLink(e.g), e.partner ? `, where it joins ${q(e.partner[0])} (${plural(e.partner[4], 'mention')})` : ''));
  return box;
}
function lineCount(k, M) {
  const box = H('div', { class: 'co-cnt' });
  let n, of, text;
  if (STEP_KINDS.includes(k)) { n = M.k[k].sp; of = M.spellings; text = [H('b', { text: nf(n) }), ` spellings changed, in ${plural(M.k[k].gr, 'concept')}`]; }
  else if (k === 'order') { n = M.order; of = M.groups; text = [H('b', { text: nf(n) }), ' concepts join spellings in another word order']; }
  else if (k === 'spacing') { n = M.spacing; of = M.groups; text = [H('b', { text: nf(n) }), ' concepts would split without the spacing key']; }
  else { n = M.groups; of = M.groups; text = [H('b', { text: nf(n) }), ` concepts with 2+ spellings; ${nf(D.g.n.length - M.groups)} with one`]; }
  box.append(H('span', { class: 'bar', title: `${nf(n)} of ${nf(of)}` }, H('i', { style: `width:${Math.max(n ? 1.5 : 0, (100 * n) / of)}%` })), H('span', {}, ...text));
  return box;
}

/* ------------------------------------------------------------------ what the rule cannot do, and the flagged word-order merges */
// The dataset flags 17 word-order merges whose two orders likely mean different things (its merge flags; in the repository
// data/frozen/dataset/likely-wrong-word-order.json). Each pair is shown only when both spellings are in one concept here.
const WRONG_ORDER = [['bias detection', 'detection bias'], ['Identity groups', 'group identity'], ['event trigger', 'trigger event'], ['online religion', 'religion online'],
  ['First person', 'person-first'], ['community of sense', 'sense of community'], ['Many-to-One', 'One-to-Many'], ['Human-to-LLM Persuasion', 'LLM-to-Human Persuasion'],
  ['Information Science', 'science information'], ['news quality', 'quality news'], ['Information quality', 'quality information'], ['State propaganda', 'propaganda state'],
  ['identity narrative', 'narrative identity'], ['encuadre de estrategia', 'estrategias de encuadre'], ['context information', 'information context'], ['bias and fairness', 'fairness bias'], ['Value and Bias', 'bias value']];
const G_ = (s) => CO.spell.get(s.toLowerCase());
const together = (a, b) => G_(a) != null && G_(a) === G_(b), apart = (a, b) => G_(a) != null && G_(b) != null && G_(a) !== G_(b);
function wrongOrder() { return CO.wrong ||= WRONG_ORDER.filter(([a, b]) => together(a, b)).map(([a, b]) => [a, b, G_(a)]); }
function limits(M) {
  const ul = H('ul', { class: 'co-limits' });
  const lk = (s) => { const g = G_(s); return g == null ? H('q', { text: s }) : H('button', { class: 'linkbtn', type: 'button', text: q(s), onclick: () => openConcept(g) }); };
  const li1 = H('li', {}, H('b', { text: 'It reads spelling, not meaning. ' }));
  if (apart('misinformation', 'false information')) li1.append('Synonyms stay apart: ', lk('misinformation'), ' and ', lk('false information'), ' are two concepts. ');
  // CDA: critical discourse analysis in some papers, a debiasing method (counterfactual data augmentation) in others
  if (apart('CDA', 'critical discourse analysis') && together('critical discourse analysis', 'critical discourse analysis (CDA)')) li1.append('A bare acronym is never expanded: ', lk('CDA'), ' stays apart from ', lk('critical discourse analysis'), ', rightly, since some papers use “CDA” for a debiasing method; “critical discourse analysis (CDA)” joins it only because the bracket is the acronym of the words before it.');
  ul.append(li1);
  const W = wrongOrder(), li2 = H('li', {}, H('b', { text: 'Word order is ignored. ' }));
  if (together('media bias', 'bias in the media')) li2.append('Right for ', lk('media bias'), ' and ', lk('bias in the media'), '; ');
  li2.append(`a reading of all ${nf(M.order)} word-order joins flagged ${nf(W.length)} whose two orders likely mean different things (below). They stay merged, because keeping word order would also split many correct joins such as “positive tone” / “tone positive”.`);
  ul.append(li2);
  const keep = [['news media', 'new media'], ['ethics', 'ethic']].filter(([a, b]) => apart(a, b));
  if (keep.length) { const li = H('li', {}, H('b', { text: 'Some forms stay apart on purpose, ' }), 'because the plural or the -ics word names something else: '); keep.forEach(([a, b], i) => li.append(i ? ' and ' : '', lk(a), ' / ', lk(b))); li.append('.'); ul.append(li); }
  const list = H('ol', { class: 'co-wrong' });
  for (const [a, b, g] of W) list.append(H('li', {}, H('button', { type: 'button', class: 'linkbtn', text: `${a} / ${b}`, onclick: () => openConcept(g) }), H('span', { class: 'muted', text: `  ${plural(D.g.n[g], 'mention')}` })));
  return [ul, H('h4', { class: 'co-sub', text: `The ${nf(W.length)} flagged word-order merges` }), list];
}

/* ------------------------------------------------------------------ spellings -> keys -> group */
function keyDiagram(g, max = 6) {
  const v = D.merges[g], L = v.labels, pick = [], combos = new Set();
  for (const l of L) { const c = l[1] + '|' + l[2]; if (!combos.has(c) && pick.length < max) { combos.add(c); pick.push(l); } }
  for (const l of L) if (pick.length < max && !pick.includes(l)) pick.push(l);
  pick.sort((a, b) => L.indexOf(a) - L.indexOf(b));
  const shared = new Set(v.joined_by.map(([t, key]) => (t === 'word-set' ? 'w' : 's') + key)), keys = [];
  for (const l of pick) for (const tk of [['w', l[1]], ['s', l[2]]]) if (!keys.some(([a, b]) => a === tk[0] && b === tk[1])) keys.push(tk);
  keys.sort((a, b) => (a[0] === b[0] ? 0 : a[0] === 'w' ? -1 : 1));
  const box = H('div', { class: 'mx', role: 'img', 'aria-label': `${plural(L.length, 'spelling')} of ${D.g.name[g]}, their word-set and spacing keys, and the concept they form` });
  const c1 = H('div', { class: 'mx-col' }, H('div', { class: 'mx-h', text: 'spellings' }));
  const spEls = pick.map((l) => { const e = H('div', { class: 'mx-sp' }, l[0], H('small', { text: nf(l[4]) })); c1.append(e); return e; });
  if (L.length > pick.length) c1.append(H('div', { class: 'mx-more', text: `+ ${plural(L.length - pick.length, 'more spelling')}` }));
  const c2 = H('div', { class: 'mx-col k' }, H('div', { class: 'mx-h', text: 'keys' }));
  const kEls = keys.map(([t, key]) => { const e = keyEl(t, key, shared.has(t + key)); c2.append(e); return e; });
  const gEl = H('div', { class: 'mx-g' }, D.g.name[g], H('small', { text: `one concept, ${plural(L.length, 'spelling')}` }));
  const svg = S('svg', { class: 'wires', 'aria-hidden': 'true' });
  box.append(svg, c1, c2, H('div', { class: 'mx-col' }, H('div', { class: 'mx-h', text: 'concept' }), gEl));
  const draw = () => {
    const R = box.getBoundingClientRect(); if (!R.width) return;
    svg.textContent = ''; svg.setAttribute('viewBox', `0 0 ${R.width} ${R.height}`);
    const at = (e) => { const r = e.getBoundingClientRect(); return { l: r.left - R.left, r: r.right - R.left, y: r.top - R.top + r.height / 2, t: r.top - R.top, b: r.bottom - R.top, x: r.left - R.left + r.width / 2 }; };
    const wire = (a, b, cls) => { const mx = (a.r + b.l) / 2; S('path', { d: `M${a.r + 2},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.l - 2},${b.y}`, class: cls }, svg); };
    const K = kEls.map(at), gb = at(gEl), stacked = gb.t > Math.max(...K.map((k) => k.b)); // on a phone the group sits below the keys
    pick.forEach((l, i) => { const a = at(spEls[i]); keys.forEach(([t, key], j) => { if ((t === 'w' ? l[1] : l[2]) === key) wire(a, K[j], (t === 'w' ? 'ws' : 'sp') + (shared.has(t + key) ? '' : ' lone')); }); });
    if (!stacked) { keys.forEach(([t, key], j) => { if (shared.has(t + key)) wire(K[j], gb, 'tog'); }); return; }
    // stacked: each shared key feeds a bus on the right of the keys, which runs down to the group
    const on = K.filter((_, j) => shared.has(keys[j][0] + keys[j][1])), bx = Math.min(R.width - 3, Math.max(...K.map((k) => k.r)) + 8);
    for (const k of on) S('path', { d: `M${k.r + 2},${k.y} H${bx}`, class: 'tog' }, svg);
    const y0 = Math.min(...on.map((k) => k.y)), y1 = Math.max(...K.map((k) => k.b)) + 6, my = (y1 + gb.t) / 2;
    S('path', { d: `M${bx},${y0} V${y1} C${bx},${my} ${gb.x},${my} ${gb.x},${gb.t - 2}`, class: 'tog' }, svg);
  };
  new ResizeObserver(draw).observe(box);
  return box;
}

/* ------------------------------------------------------------------ a concept's spellings: the diagram, then every step */
const nSpell = (g) => (D.merges[g] ? D.merges[g].labels.length : 1);
function spellSection(g) {
  const body = H('div', { class: 'co-body' }), v = D.merges[g], G = D.g, n = G.n[g], nm = G.name[g];
  if (!v) { body.append(H('p', { class: 'small muted', text: n === 1 ? `One spelling: its single mention is written ${q(nm)}, so nothing was merged.` : `One spelling: all ${nf(n)} mentions are written exactly ${q(nm)}, so nothing was merged.` })); return body; }
  const L = v.labels, tot = L.reduce((s, l) => s + l[4], 0), all = v.joined_by.find((j) => j[2].length === L.length);
  body.append(H('p', { class: 'small' }, `${plural(L.length, 'spelling')} with ${plural(tot, 'mention')} became one concept`,
    all ? [`: all share the ${all[0] === 'word-set' ? 'word-set' : 'spacing'} key `, keyEl(all[0] === 'word-set' ? 'w' : 's', all[1], true), '. '] : `: no key is shared by all ${L.length}, so they join through a chain of shared keys. `,
    H('button', { class: 'linkbtn', type: 'button', text: 'How merging works', onclick: openHow })));
  body.append(keyDiagram(g, 6));
  const oj = orderJoins(v);
  if (oj.length) { const b = oj[0].buckets; body.append(H('p', { class: 'note', text: `Joined by the same words in another order: ${q(b[0].labels[0][0])} and ${q(b[1].labels[0][0])}. Usually that is one concept; a few such joins mean different things, so check the definitions.` })); }
  body.append(lazyMore(`Each spelling’s steps and the shared keys`, () => traceRows(g)));
  return body;
}
function traceRows(g) {
  const v = D.merges[g], L = v.labels, oj = orderJoins(v), box = H('div', { class: 'co-body' });
  const shared = new Set(v.joined_by.map(([t, key]) => (t === 'word-set' ? 'w' : 's') + key));
  // which spellings stand in another word order than the most common one of their word-set key
  const odd = new Map();
  for (const j of oj) for (const b of j.buckets.slice(1)) for (const l of b.labels) odd.set(l[0], [b.seq, j.buckets[0].seq, j.key]);
  const rows = H('div', { class: 'co-trace' });
  for (const l of L) {
    const o = odd.get(l[0]);
    const chips = l[3].map((s) => chip(kindOf(s), chipText(s), s));
    if (o) chips.push(chip('order', `other word order: ${o[0]}`, `In order, its content words are “${o[0]}”; the most common spelling has “${o[1]}”. Sorting gives both the word-set key “${o[2]}”.`));
    rows.append(H('div', { class: 'co-sprow' },
      H('div', { class: 'h' }, H('b', { text: l[0] }), H('span', { text: plural(l[4], 'mention') })),
      H('div', { class: 'co-chips' }, chips.length ? chips : H('span', { class: 'co-chip none', text: 'no step changed it' })),
      H('div', { class: 'keys' }, keyEl('w', l[1], shared.has('w' + l[1])), keyEl('s', l[2], shared.has('s' + l[2])))));
  }
  box.append(rows, H('div', { class: 'small', text: 'Joined by:' }), H('ul', { class: 'co-joins' }, v.joined_by.map(([t, key, sps]) => H('li', {}, keyEl(t === 'word-set' ? 'w' : 's', key, true),
    sps.length === L.length ? ` shared by all ${L.length} spellings` : ` shared by ${sps.length}: ${sps.join(' · ')}`))));
  return box;
}

/* ------------------------------------------------------------------ the tab */
function hoverTip(el, text) {
  const on = () => { tipEl.textContent = text; tipEl.hidden = false; place(tipEl, el); }, off = () => (tipEl.hidden = true);
  el.addEventListener('pointerenter', on); el.addEventListener('pointerleave', off); el.addEventListener('focus', on); el.addEventListener('blur', off);
}
const tsw = (t) => H('i', { class: 'co-sw', style: `background:var(--tc${t})`, 'aria-hidden': 'true' });
function topicColours() { const pal = topicPalette(mapFrame()), el = $('#v-concepts'); pal.forEach((c, t) => el.style.setProperty(`--tc${t}`, rgbStr(c))); }

function concepts() {
  const G = D.g, nG = G.n.length, NT = D.meta.topics.length;
  CO.strictIn = Uint32Array.from(D.members, (ms) => ms.reduce((s, i) => s + D.strict[i], 0));
  CO.isM = new Uint8Array(nG); CO.bandOf = new Uint8Array(nG); CO.top = new Uint8Array(nG);
  CO.hay = new Array(nG); CO.sq = new Array(nG); CO.spell = new Map();
  const tc = new Uint32Array(NT);
  for (let g = 0; g < nG; g++) {
    const sp = [G.name[g], ...Object.keys(G.labels[g]).filter((l) => l !== G.name[g])], lo = sp.map((s) => s.toLowerCase());
    CO.hay[g] = '\n' + lo.join('\n') + '\n'; CO.sq[g] = '\n' + sp.map(squash).join('\n') + '\n';
    for (const l of lo) if (!CO.spell.has(l)) CO.spell.set(l, g);
    if (D.merges[g]) CO.isM[g] = 1;
    const n = G.n[g]; CO.bandOf[g] = BANDS.findIndex(([a, b]) => n >= a && n <= b);
    // the main topic: the topic of most of its mentions; a tie goes to the topic of its highest-scoring mention
    const ms = D.members[g];
    if (ms.length === 1) CO.top[g] = D.m.topic[ms[0]];
    else {
      tc.fill(0); let best = -1, bc = 0, bs = -1;
      for (const i of ms) tc[D.m.topic[i]]++;
      for (const i of ms) { const t = D.m.topic[i], c = tc[t], s = D.score.s[i]; if (c > bc || (c === bc && s > bs)) { best = t; bc = c; bs = s; } }
      CO.top[g] = best;
    }
  }
  CO.tN = new Uint32Array(NT); for (let g = 0; g < nG; g++) CO.tN[CO.top[g]]++;
  // topics in the facet: the one led by "media bias", then the others led by a bias concept, then by concepts
  const led = (t) => { const s = D.meta.topics[t].top[0]; return s.toLowerCase() === 'media bias' ? 2 : /bias/i.test(s) ? 1 : 0; };
  CO.torder = [...Array(NT).keys()].sort((a, b) => led(b) - led(a) || CO.tN[b] - CO.tN[a]);
  CO.nBias = CO.torder.filter((t) => led(t)).length;
  topicColours();
  const cnt = { concepts: nG, strict: 0, merged: 0, single: 0 };
  for (let g = 0; g < nG; g++) { if (CO.strictIn[g]) cnt.strict++; if (CO.isM[g]) cnt.merged++; if (G.n[g] === 1) cnt.single++; }
  CO.cnt = cnt;
  // glossary entries with worked examples from the data
  let sx = -1, nx = -1, mx = -1;
  for (let g = 0; g < nG; g++) {
    if (CO.strictIn[g] && CO.strictIn[g] < G.n[g] && (sx < 0 || G.n[g] > G.n[sx])) sx = g;
    if (!CO.strictIn[g] && (nx < 0 || G.n[g] > G.n[nx])) nx = g;
    if (D.merges[g] && (mx < 0 || nSpell(g) > nSpell(mx) || (nSpell(g) === nSpell(mx) && G.n[g] > G.n[mx]))) mx = g;
  }
  GL.strictconcept[2] = `Every mention of the dataset reaches the cut (${D.meta.cut.toFixed(4)}); the ones past the stricter best-F1 cut (${D.meta.best_f1_cut.toFixed(4)}) are marked strict. Example: ${q(G.name[sx])} has ${plural(G.n[sx], 'mention')}, ${nf(CO.strictIn[sx])} of them strict, so it is a strict concept. ${q(G.name[nx])} has ${plural(G.n[nx], 'mention')} but none past the stricter cut, so it is a concept but not a strict one. ${nf(cnt.strict)} of the ${nf(cnt.concepts)} concepts are strict.`;
  GL.merged[2] = `${nf(cnt.merged)} of the ${nf(cnt.concepts)} concepts have two or more spellings. The most: ${q(G.name[mx])}, ${plural(nSpell(mx), 'spelling')}: ${D.merges[mx].labels.map((l) => l[0]).join(' · ')}.`;
  CO.exG = sx; CO.mostSp = mx;
  CO.M = mergeStats();

  // sticky parts (the list, the section bar of a concept) sit just below the header, which takes two rows on a phone
  const hdr = $('header.top'), setTop = () => $('#v-concepts').style.setProperty('--co-top', `${Math.round(hdr.getBoundingClientRect().height) + 6}px`);
  new ResizeObserver(setTop).observe(hdr); setTop();
  hero();
  buildFacets();
  CO.defIO = new IntersectionObserver(loadDefs, { rootMargin: '160px 0px' });
  CO.moreIO = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting) && wide()) renderList(false); }, { rootMargin: '240px 0px' });
  const inp = $('#co-q'); let raf = 0;
  inp.addEventListener('input', () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => { CO.q = inp.value.trim().toLowerCase(); $('#co-split').classList.remove('reading'); run(); }); });
  inp.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && CO.R.length) { e.preventDefault(); openConcept(CO.R[0]); }
    else if (e.key === 'ArrowDown') { const b = $('#co-res .co-row'); if (b) { e.preventDefault(); b.focus(); } }
    else if (e.key === 'Escape' && inp.value) { inp.value = ''; CO.q = ''; run(); }
  });
  addEventListener('keydown', (e) => { if (e.key === '/' && !$('#v-concepts').hidden && !/^(INPUT|TEXTAREA|SELECT)$/.test((document.activeElement || {}).tagName || '') && !e.ctrlKey && !e.metaKey) { e.preventDefault(); inp.focus(); inp.select(); } });
  $('#co-clear').addEventListener('click', clearAll);
  expanders();
  run();
  if (wide()) { const mb = G_('media bias'); if (mb != null) openConcept(mb, true); }
  onTheme(() => { topicColours(); if (CO.spreadOn) spreadChart(); if (CO.sizeEl) sizeChart(); });
  (window.requestIdleCallback || ((f) => setTimeout(f, 400)))(() => nameScan());
}

/* ------------------------------------------------------------------ the head: title, search, headline numbers */
function hero() {
  const G = D.g, nG = G.n.length, c = CO.cnt;
  $('#co-q').placeholder = `Search ${nf(nG)} concepts or any spelling`;
  const lede = $('#co-lede'); lede.textContent = '';
  lede.append(`${nf(nG)} `, gl('concept', 'concepts'), ': every ', gl('mention', 'mention'), ' of one term, in every ', gl('spelling', 'spelling'), ', across papers.');
  const st = $('#co-stats'); st.textContent = '';
  const go = (f) => () => { f(); run(); $('#co-facets').scrollIntoView({ behavior: 'smooth', block: 'start' }); };
  st.append(
    stat({ value: nG, label: 'concepts', sub: `from ${nf(D.m.n)} mentions`, lead: true }),
    stat({ value: c.merged, label: 'merge 2+ spellings', sub: `${pc(c.merged / nG)}, up to ${nSpell(CO.mostSp)} spellings each`, onclick: go(() => { CO.merged = true; }) }),
    stat({ value: c.strict, label: 'strict', sub: `${pc(c.strict / nG)} have a surer mention`, onclick: go(() => { CO.strict = true; }) }),
    stat({ value: c.single, label: 'single-mention', sub: `${pc(c.single / nG)}: one paper defines them`, onclick: go(() => { CO.band = 0; }) }));
  const tr = $('#co-try'); tr.textContent = '';
  tr.append(H('span', { class: 'co-fl', text: 'Try' }));
  for (const s of ['media bias', 'fake news', 'framing', 'echo chambers', 'confirmation bias']) { const g = G_(s); if (g != null) tr.append(H('button', { type: 'button', class: 'co-tryb', text: G.name[g], onclick: () => openConcept(g) })); }
  tr.append(H('button', { type: 'button', class: 'co-tryb', text: 'any concept ↻', title: 'open a concept at random', onclick: () => openConcept(Math.floor(Math.random() * nG)) }));
  tr.after(more('What a concept is', H('p', {}, 'A concept gathers every mention whose label is the same up to spelling (one ', gl('labelgroup', 'label group'), `). ${nf(c.merged)} merge two or more spellings (`, H('button', { class: 'linkbtn', type: 'button', text: 'how term merging works', onclick: openHow }),
    `). ${nf(c.strict)} are `, gl('strictconcept', 'strict concepts'), ': each has at least one ', gl('strict', 'strict'), ' mention, past the stricter best-F1 cut. ', gl('spread', 'Spread'), ' says how alike the mentions of one concept are. Each concept takes the colour of its ', gl('maintopic', 'main topic'), '.')));
}

/* ------------------------------------------------------------------ facets: strict, merged, size, topic, sort */
function facetChip(label, onclick, help, sw) {
  const b = H('button', { type: 'button', class: 'chip', 'aria-pressed': 'false', onclick }, sw || null, H('span', { class: 'co-cl', text: label }), H('span', { class: 'n' }));
  if (help) hoverTip(b, help);
  return b;
}
function buildFacets() {
  const box = $('#co-facets'); box.textContent = '';
  CO.fS = facetChip('strict only', () => { CO.strict = !CO.strict; run(); }, GL.strictconcept[1]);
  CO.fM = facetChip('merged spellings', () => { CO.merged = !CO.merged; run(); }, GL.merged[1]);
  CO.fB = BANDS.map(([, , t], k) => facetChip(t, () => { CO.band = CO.band === k ? -1 : k; run(); }, `Concepts with ${t} mention${t === '1' ? '' : 's'}`));
  const sortEl = H('span');
  CO.dirB = H('button', { type: 'button', class: 'chip co-dir', 'aria-pressed': 'false', 'aria-label': 'reverse the order', title: 'reverse the order', text: '↓', onclick: () => { CO.dir = -CO.dir; run(); } });
  seg(sortEl, [['n', 'mentions'], ['papers', 'papers'], ['az', 'A–Z'], ['spread', 'spread']], CO.sort, (v) => { CO.sort = v; CO.dir = 1; run(); });
  box.append(H('div', { class: 'co-frow' },
    H('span', { class: 'co-fg' }, H('span', { class: 'co-fl', text: 'Show' }), CO.fS, CO.fM),
    H('span', { class: 'co-fg' }, H('span', { class: 'co-fl', text: 'Mentions' }), ...CO.fB),
    H('span', { class: 'co-fg co-sortg' }, H('span', { class: 'co-fl', text: 'Sort' }), sortEl, CO.dirB)));
  CO.fT = CO.torder.map((t) => { const b = facetChip(tshort(t), () => { CO.topic = CO.topic === t ? -1 : t; run(); }, `${D.meta.topics[t].name}: ${plural(CO.tN[t], 'concept')} whose mentions fall mostly in this topic`, tsw(t)); b.dataset.t = t; return b; });
  CO.fTm = H('button', { type: 'button', class: 'chip co-tmore', 'aria-expanded': 'false', onclick: () => { CO.allTopics = !CO.allTopics; updateFacets(); } });
  box.append(H('div', { class: 'co-frow co-topics' }, H('span', { class: 'co-fl' }, gl('maintopic', 'Topic')), ...CO.fT, CO.fTm));
}
function updateFacets() {
  const C = CO.counts; if (!C) return;
  const set = (b, on, n) => { b.setAttribute('aria-pressed', String(on)); $('.n', b).textContent = nf(n); b.classList.toggle('zero', !n && !on); };
  set(CO.fS, CO.strict, C.cS); set(CO.fM, CO.merged, C.cM);
  CO.fB.forEach((b, k) => set(b, CO.band === k, C.cB[k]));
  const SHOW = Math.max(CO.nBias + 6, 9);
  CO.fT.forEach((b, k) => { const t = +b.dataset.t; set(b, CO.topic === t, C.cT[t]); b.hidden = !(CO.allTopics || k < SHOW || CO.topic === t); });
  CO.fTm.textContent = CO.allTopics ? 'fewer topics' : `+ ${nf(CO.fT.length - SHOW)} more topics`;
  CO.fTm.setAttribute('aria-expanded', String(CO.allTopics));
  CO.dirB.textContent = CO.dir > 0 ? '↓' : '↑'; CO.dirB.setAttribute('aria-pressed', String(CO.dir < 0));
  $('#co-clear').hidden = !(CO.q || CO.strict || CO.merged || CO.band >= 0 || CO.topic >= 0);
}
function clearAll() { CO.q = ''; $('#co-q').value = ''; CO.strict = CO.merged = false; CO.band = CO.topic = -1; run(); }

/* ------------------------------------------------------------------ search and the result list */
// every concept in the order of a sort key (mentions, papers, A–Z, spread), computed once per key and direction
function order() {
  const key = CO.sort + CO.dir; CO.ords ||= {}; if (CO.ords[key]) return CO.ords[key];
  const G = D.g, idx = Array.from({ length: G.n.length }, (_, g) => g), sc = (g) => G.mean_s[g];
  let cmp;
  if (CO.sort === 'papers') cmp = (a, b) => G.papers[b] - G.papers[a] || G.n[b] - G.n[a] || sc(b) - sc(a) || a - b;
  else if (CO.sort === 'az') { const k = G.name.map((s) => s.toLowerCase().replace(/^[^\p{L}\p{N}]+/u, '')); cmp = (a, b) => (k[a] < k[b] ? -1 : k[a] > k[b] ? 1 : a - b); }
  else if (CO.sort === 'spread') cmp = (a, b) => (G.pair[b] ?? -1) - (G.pair[a] ?? -1) || G.n[b] - G.n[a] || a - b;
  else cmp = (a, b) => G.n[b] - G.n[a] || G.papers[b] - G.papers[a] || sc(b) - sc(a) || a - b;
  idx.sort(cmp);
  if (CO.dir < 0) { idx.reverse(); if (CO.sort === 'spread') { const k = idx.findIndex((g) => G.pair[g] != null); idx.push(...idx.splice(0, k)); } }
  return (CO.ords[key] = idx);
}
// how well concept g matches the search: 0 exactly (a name or spelling, or the same letters and digits), 1 at the start,
// 2 at the start of a word, 3 anywhere; -1 not at all
function tierOf(g, s, ss) {
  const h = CO.hay[g];
  if (h.includes('\n' + s + '\n') || (ss.length > 2 && CO.sq[g].includes('\n' + ss + '\n'))) return 0;
  if (h.includes('\n' + s)) return 1;
  if (h.includes(' ' + s) || h.includes('-' + s) || h.includes('(' + s)) return 2;
  if (h.includes(s) || (ss.length > 3 && CO.sq[g].includes(ss))) return 3;
  return -1;
}
function run() {
  const s = CO.q, ss = squash(s), ord = order(), NT = D.meta.topics.length, cT = new Uint32Array(NT), cB = [0, 0, 0, 0], tiers = [[], [], [], []];
  let cS = 0, cM = 0;
  for (let k = 0; k < ord.length; k++) {
    const g = ord[k], t = s ? tierOf(g, s, ss) : 0; if (t < 0) continue;
    const st = CO.strictIn[g] > 0, m = CO.isM[g] === 1, b = CO.bandOf[g], tp = CO.top[g];
    const okS = !CO.strict || st, okM = !CO.merged || m, okB = CO.band < 0 || b === CO.band, okT = CO.topic < 0 || tp === CO.topic;
    if (okS && okM && okB) cT[tp]++;
    if (okS && okM && okT) cB[b]++;
    if (okM && okB && okT && st) cS++;
    if (okS && okB && okT && m) cM++;
    if (okS && okM && okB && okT) tiers[t].push(g);
  }
  CO.R = s ? [].concat(...tiers) : tiers[0];
  CO.counts = { cT, cB, cS, cM };
  updateFacets(); renderList(true);
  if (CO.tb && CO.tb.el.isConnected && CO.tb.el.closest('details').open) { CO.tb.page = 0; drawTable(); }
}
function countLine() {
  const el = $('#co-count'), R = CO.R; el.textContent = '';
  const parts = [];
  if (CO.strict) parts.push('strict'); if (CO.merged) parts.push('merged');
  el.append(H('span', {}, H('b', { text: nf(R.length) }), ` ${parts.length ? parts.join(', ') + ' ' : ''}concept${R.length === 1 ? '' : 's'}`,
    CO.band >= 0 ? ` with ${BANDS[CO.band][2]} mention${CO.band ? 's' : ''}` : '', CO.topic >= 0 ? [' in ', tsw(CO.topic), tshort(CO.topic)] : '', CO.q ? ` matching ${q(CO.q)}` : ''));
}
function renderList(reset) {
  const ol = $('#co-res');
  if (reset) { ol.textContent = ''; CO.shown = 0; ol.scrollTop = 0; CO.defIO.disconnect(); CO.moreIO.disconnect(); countLine(); }
  const old = $('.co-moreli', ol); if (old) { CO.moreIO.unobserve(old); old.remove(); }
  const add = CO.R.slice(CO.shown, CO.shown + PAGE), frag = document.createDocumentFragment();
  for (const g of add) frag.append(rowEl(g));
  ol.append(frag); CO.shown += add.length;
  if (!CO.R.length) {
    const li = H('li', { class: 'co-empty' }, H('p', {}, 'Nothing matches', CO.q ? ` ${q(CO.q)}` : '', '.'));
    const filt = CO.strict || CO.merged || CO.band >= 0 || CO.topic >= 0;
    li.append(H('p', { class: 'small muted' }, filt ? ['Some filters are on: ', H('button', { type: 'button', class: 'linkbtn', text: 'clear them', onclick: () => { const v = CO.q; clearAll(); $('#co-q').value = v; CO.q = v; run(); } }), '.'] : 'Try a shorter part of the name, or another spelling.'));
    ol.append(li);
  } else if (CO.shown < CO.R.length) {
    const left = CO.R.length - CO.shown, li = H('li', { class: 'co-moreli' }, H('button', { type: 'button', class: 'btn', text: `Show ${nf(Math.min(PAGE, left))} more · ${nf(left)} left`, onclick: () => renderList(false) }));
    ol.append(li); CO.moreIO.observe(li);
  }
}
const metricOf = (g) => {
  const G = D.g;
  if (CO.sort === 'papers') return [nf(G.papers[g]), G.papers[g] === 1 ? 'paper' : 'papers', Math.log1p(G.papers[g]) / Math.log1p(CO.maxP ||= Math.max(...G.papers))];
  if (CO.sort === 'spread') return G.pair[g] == null ? ['–', 'one mention', 0] : [f2(G.pair[g] / 1000), 'spread', G.pair[g] / 1000];
  return [nf(G.n[g]), G.n[g] === 1 ? 'mention' : 'mentions', Math.log1p(G.n[g]) / Math.log1p(CO.maxN ||= Math.max(...G.n))];
};
function matchedSpelling(g) { // the spelling the search found, when it is not the concept's name
  if (!CO.q) return null;
  const s = CO.q, ss = squash(s), hit = (x) => x.toLowerCase().includes(s) || (ss.length > 3 && squash(x).includes(ss));
  if (hit(D.g.name[g])) return null;
  return Object.keys(D.g.labels[g]).find(hit) || null;
}
function rowEl(g) {
  const G = D.g, tp = CO.top[g], as = matchedSpelling(g), [v, unit, w] = metricOf(g), sub = CO.sort === 'papers' ? plural(G.n[g], 'mention') : plural(G.papers[g], 'paper');
  const b = H('button', { type: 'button', class: 'co-row', 'data-g': g, 'aria-current': CO.sel === g ? 'true' : null, onclick: () => openConcept(g), onkeydown: rowKey },
    H('span', { class: 'co-rsw', style: `background:var(--tc${tp})`, title: `main topic: ${tshort(tp)}` }),
    H('span', { class: 'co-rmain' },
      H('span', { class: 'co-rt' }, H('b', { text: G.name[g] }), as ? H('span', { class: 'co-ras', text: `as ${q(as)}` }) : null, CO.strictIn[g] ? H('span', { class: 'co-rst', title: GL.strictconcept[1], text: 'strict' }) : null),
      H('span', { class: 'co-rd' })),
    H('span', { class: 'co-rnum' }, H('span', { class: 'co-rv' }, H('b', { text: v }), ` ${unit}`), H('span', { class: 'co-rbar', 'aria-hidden': 'true' }, H('i', { style: `width:${Math.max(4, 100 * w).toFixed(1)}%` })), H('span', { class: 'co-rs', text: sub })));
  CO.defIO.observe(b);
  return H('li', {}, b);
}
function rowKey(e) {
  if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
  e.preventDefault();
  const rows = $$('#co-res .co-row'), k = rows.indexOf(e.currentTarget) + (e.key === 'ArrowDown' ? 1 : -1);
  if (k < 0) $('#co-q').focus(); else if (rows[k]) rows[k].focus();
}
// one definition per visible row, from the mention whose shard of definitions is already loaded (then the highest score)
function pickMention(g) {
  const ms = D.members[g]; if (ms.length === 1) return ms[0];
  let best = ms[0], bs = -1;
  for (const i of ms) { const sc = (shardCache[Math.floor(i / D.meta.shard)] ? 2 : 0) + D.score.s[i]; if (sc > bs) { bs = sc; best = i; } }
  return best;
}
function loadDefs(entries) {
  const vis = entries.filter((e) => e.isIntersecting).map((e) => e.target);
  vis.sort((a, b) => D.members[+a.dataset.g].length - D.members[+b.dataset.g].length); // single mentions first: they fix their shard
  for (const el of vis) {
    CO.defIO.unobserve(el);
    const g = +el.dataset.g;
    defsOf(pickMention(g)).then((d) => { const t = (d && d[0]) || ''; $('.co-rd', el).textContent = t.length > 220 ? t.slice(0, 219) + '…' : t; }).catch(() => {});
  }
}

/* ------------------------------------------------------------------ the open concept */
function kfig(v, label) { return H('div', { class: 'co-k' }, H('b', { text: v }), H('span', {}, label)); }
function backToList() {
  $('#co-split').classList.remove('reading');
  const r = $(`#co-res .co-row[data-g="${CO.sel}"]`);
  if (r) { r.scrollIntoView({ block: 'center' }); r.focus({ preventScroll: true }); } else $('#co-facets').scrollIntoView({ block: 'start' });
}
function alsoLine(g) {
  const G = D.g, L = Object.entries(G.labels[g]).filter(([l]) => l !== G.name[g]).sort((a, b) => b[1] - a[1]);
  if (!L.length) return null;
  const p = H('p', { class: 'co-also' }, 'also written ');
  L.slice(0, 4).forEach(([l, n], k) => p.append(k ? ' · ' : '', H('q', { text: l }), H('small', { text: ` ${nf(n)}` })));
  if (L.length > 4) p.append(` · +${nf(L.length - 4)} more`);
  return p;
}
function topicBar(g) {
  const ms = D.members[g]; if (ms.length < 2) return null;
  const C = new Map(); for (const i of ms) C.set(D.m.topic[i], (C.get(D.m.topic[i]) || 0) + 1);
  const T = [...C].sort((a, b) => b[1] - a[1]), n = ms.length;
  const bar = H('div', { class: 'co-tbar', role: 'img', 'aria-label': `its ${nf(n)} mentions by topic: ` + T.slice(0, 4).map(([t, c]) => `${tshort(t)} ${nf(c)}`).join(', ') });
  for (const [t, c] of T) { const s = H('span', { style: `flex:${c};background:var(--tc${t})` }); hoverTip(s, `${tshort(t)}: ${plural(c, 'mention')} (${pc(c / n)})`); bar.append(s); }
  const lg = H('div', { class: 'co-tlg' });
  T.slice(0, 3).forEach(([t, c]) => lg.append(H('button', { type: 'button', class: 'co-tlb', title: 'list the concepts of this topic', onclick: () => { CO.topic = t; run(); backToList(); } }, tsw(t), tshort(t), H('small', { text: ` ${nf(c)}` }))));
  if (T.length > 3) lg.append(H('span', { class: 'muted', text: `+ ${plural(T.length - 3, 'topic')}` }));
  if (T.length === 1) return H('p', { class: 'co-tlg' }, tsw(T[0][0]), `All ${nf(n)} of its mentions fall in this topic.`);
  return H('div', { class: 'co-topics1' }, H('span', { class: 'co-fl' }, `its mentions in ${nf(T.length)} topics`), bar, lg);
}
function section(id, title, meta) { const m = H('span', { class: 'co-sm', text: meta || '' }), s = H('section', { class: 'co-sec', id: 'co-s-' + id }, H('div', { class: 'co-sh' }, H('h3', {}, title), m)); s.meta = m; return s; }
function openConcept(g, auto) {
  if (g == null || !(g >= 0 && g < D.g.n.length)) return;
  CO.sel = g;
  const G = D.g, n = G.n[g], st = CO.strictIn[g], nsp = nSpell(g), tp = CO.top[g], box = $('#co-detail');
  box.textContent = '';
  const head = H('header', { class: 'co-dh' },
    H('div', { class: 'co-dhrow' }, H('button', { type: 'button', class: 'btn co-back', text: '← Results', onclick: backToList }),
      H('button', { type: 'button', class: 'co-tp', title: 'list the concepts of this topic', onclick: () => { CO.topic = tp; run(); backToList(); } }, tsw(tp), tshort(tp))),
    H('h2', { class: 'co-dname', text: G.name[g] }), alsoLine(g),
    H('div', { class: 'co-pills' }, st ? H('span', { class: 'pill strict', text: 'strict concept' }) : H('span', { class: 'pill plain', text: 'not strict' }),
      nsp > 1 ? H('span', { class: 'pill plain', text: `${nsp} spellings merged` }) : null, n === 1 ? H('span', { class: 'pill plain', text: 'one mention, one paper' }) : null));
  const kn = H('div', { class: 'co-kn' }, kfig(nf(n), gl('mention', n === 1 ? 'mention' : 'mentions')), kfig(nf(G.papers[g]), G.papers[g] === 1 ? 'paper' : 'papers'),
    kfig(nf(G.ndef[g]), gl('definition', G.ndef[g] === 1 ? 'definition' : 'definitions')), kfig(nf(st), gl('strict', 'strict')),
    kfig(f2(G.mean_s[g] / 10000), gl('ensemble', 'mean score')), G.pair[g] != null ? kfig(f2(G.pair[g] / 1000), gl('spread', 'spread')) : null);
  const acts = H('div', { class: 'co-acts' },
    H('button', { type: 'button', class: 'btn primary', text: 'Show on the map', onclick: () => { MAPSEL.add(g); show('map'); TABS.map && TABS.map.showConcepts && TABS.map.showConcepts(); } }));
  const S1 = section('defs', 'Definitions', `${nf(G.ndef[g])} from ${plural(G.papers[g], 'paper')}`), S2 = section('says', 'What its papers say', '…'),
    S3 = section('fam', 'Family', '…'), S4 = section('sim', 'Similar concepts', 'closest in meaning'), S5 = section('spell', 'Spellings', nsp > 1 ? `${nsp} → one concept` : 'one'),
    S6 = n > 1 ? section('spread', 'How alike its mentions are', `spread ${f3(G.pair[g] / 1000)}`) : null;
  const secs = [S1, S2, S3, S4, S5, S6].filter(Boolean);
  const toc = H('nav', { class: 'co-toc', 'aria-label': 'Sections of this concept' }, secs.map((s) => H('button', { type: 'button', class: 'co-tocb', text: $('h3', s).textContent, onclick: () => s.scrollIntoView({ behavior: 'smooth', block: 'start' }) })));
  box.append(...[head, kn, topicBar(g), acts, toc, ...secs].filter(Boolean));
  defsSection(g, S1); saysSection(g, S2); familySection(g, S3); S4.append(similarSection(g)); S5.append(spellSection(g)); if (S6) S6.append(spreadBlock(g));
  for (const r of $$('#co-res .co-row')) { if (+r.dataset.g === g) r.setAttribute('aria-current', 'true'); else r.removeAttribute('aria-current'); }
  markSel();
  if (CO.tb && CO.tb.el.isConnected) for (const r of $$('tr.row', CO.tb.el)) r.classList.toggle('sel', +r.dataset.g === g);
  if (auto) return;
  $('#co-split').classList.add('reading');
  requestAnimationFrame(() => { const t = box.getBoundingClientRect().top; if (!wide() || t < 50 || t > innerHeight * 0.55) box.scrollIntoView({ behavior: wide() ? 'smooth' : 'auto', block: 'start' }); });
}

/* definitions, grouped by paper: the surest mention's paper first; a paper that writes the term twice has two mentions */
function defsSection(g, sec) {
  const byP = new Map();
  for (const i of D.members[g]) { const p = D.m.paper[i]; (byP.get(p) || byP.set(p, []).get(p)).push(i); }
  const P = [...byP].map(([p, is]) => [p, is.sort((a, b) => D.score.s[b] - D.score.s[a])]).sort((a, b) => D.score.s[b[1][0]] - D.score.s[a[1][0]] || a[0] - b[0]);
  const shared = H('div', { class: 'co-shareddefs' }), ol = H('ol', { class: 'co-papers' }), moreB = H('button', { type: 'button', class: 'btn co-morep' });
  let shown = 0;
  const step = () => { const k = shown ? 15 : 5; for (const [p, is] of P.slice(shown, shown + k)) ol.append(paperRow(g, p, is)); shown = Math.min(P.length, shown + k); const left = P.length - shown; moreB.hidden = !left; moreB.textContent = `Show ${nf(Math.min(15, left))} more papers · ${nf(left)} left`; };
  moreB.addEventListener('click', step); step();
  sec.append(shared, ol, moreB);
  if (P.length > 1) sec.append(H('p', { class: 'small muted', text: 'Papers are listed surest first: by the highest ensemble score of their mentions of it.' }));
  termOf(g).then((tv) => {
    if (!tv || !tv.df.length || !shared.isConnected) return;
    const max = tv.df[0][1], list = H('ul', { class: 'co-items' });
    tv.df.slice(0, 3).forEach((it) => list.append(H('li', { class: 'co-it sent' }, H('div', { class: 'co-itr' }, H('span', { class: 't', text: it[0] }), H('span', { class: 'co-bar' }, H('i', { style: `width:${Math.max(3, (100 * it[1]) / max)}%` })), H('b', { class: 'co-n', text: `×${nf(it[1])}` })))));
    shared.append(H('div', { class: 'co-fldh' }, H('h5', {}, gl('agree', 'Stated by several papers')), H('span', { class: 'small muted', text: `${plural(tv.df.length, 'definition')} that two or more papers give word for word` })), list,
      ...(tv.df.length > 3 ? [lazyMore(`All ${nf(tv.df.length)} shared definitions, with their papers`, () => itemList(g, 'df', tv.df))] : []));
  }).catch(() => {});
}
function paperRow(g, p, is) {
  const li = H('li', { class: 'co-pr' });
  li.append(H('div', { class: 'co-prh' }, H('button', { type: 'button', class: 'linkbtn co-pt', title: 'open this paper', text: clip(D.p.title[p] || D.p.id[p], 160), onclick: () => { show('papers'); TABS.papers && TABS.papers.open && TABS.papers.open(p); } }),
    H('span', { class: 'co-pm', text: D.p.corpus[p] === 'seeds' ? 'seed paper' : 'cited paper' })));
  for (const i of is) {
    const defs = H('ol', { class: 'co-defs' }, H('li', { class: 'muted', text: '…' }));
    const meta = H('div', { class: 'co-pdm' }, D.m.label[i] !== D.g.name[g] ? H('span', { text: `written ${q(D.m.label[i])}` }) : null, D.strict[i] ? H('span', { class: 'co-rst', text: 'strict' }) : null,
      H('span', { class: 'mono', title: 'ensemble score', text: f3(D.score.s[i]) }), H('button', { type: 'button', class: 'linkbtn', text: 'scores and neighbours', onclick: () => openMention(i) }));
    li.append(H('div', { class: 'co-pd' }, defs, meta));
    defsOf(i).then((d) => {
      defs.textContent = ''; d = d || [];
      d.forEach((t, k) => defs.append(H('li', { text: t, hidden: k ? '' : null })));
      if (d.length > 1) {
        const tb = H('button', { type: 'button', class: 'linkbtn co-tg', 'aria-expanded': 'false', text: `${d.length - 1} more definition${d.length > 2 ? 's' : ''} in this paper` });
        tb.addEventListener('click', () => { const o = tb.getAttribute('aria-expanded') !== 'true'; tb.setAttribute('aria-expanded', String(o)); $$('li', defs).slice(1).forEach((x) => (x.hidden = !o)); defs.classList.toggle('many', o); });
        meta.prepend(tb);
      }
    }).catch((e) => { defs.textContent = 'The definitions could not be loaded: ' + e.message; });
  }
  return li;
}

/* what its papers say: compact chips per field, then everything with its forms and papers one click away */
const SAYLAB = { sp: 'A kind of', sb: 'Its kinds', sy: 'Other names', rt: 'Related terms', de: 'Described as', ex: 'Examples' };
function itemChip(g, it) {
  const [text, n, link] = it, lk = link >= 0 && link !== g, kids = [H('span', { text: clip(text, 64) }), n > 1 ? H('b', { text: `×${nf(n)}` }) : null];
  return lk ? H('button', { type: 'button', class: 'fchip co-fc lk', title: `${plural(n, 'paper')} · open the concept “${D.g.name[link]}”`, onclick: () => openConcept(link) }, ...kids)
    : H('span', { class: 'fchip co-fc', title: `${plural(n, 'paper')} state${n === 1 ? 's' : ''} this` }, ...kids);
}
function chipRow(g, c, items) {
  const chips = H('div', { class: 'co-crc' }), K = 10, MAX = 150;
  items.slice(0, K).forEach((it) => chips.append(itemChip(g, it)));
  if (items.length > K) {
    const b = H('button', { type: 'button', class: 'chip co-cmore', text: `+ ${nf(items.length - K)} more` });
    b.addEventListener('click', () => { items.slice(K, MAX).forEach((it) => chips.insertBefore(itemChip(g, it), b)); if (items.length > MAX) { b.textContent = `${nf(items.length - MAX)} more in the full list below`; b.disabled = true; } else b.remove(); }, { once: true });
    chips.append(b);
  }
  return H('div', { class: 'co-cr', style: `--fc:var(${FIELD[c][3]})` }, H('span', { class: 'co-crl' }, gl(FIELD[c][2].replace(' ', ''), SAYLAB[c]), H('span', { class: 'n', text: nf(items.length) })), chips);
}
function textRow(g, c, items, full) {
  const ul = H('ul', { class: 'co-txt' });
  items.slice(0, 2).forEach((it) => ul.append(H('li', {}, H('span', { text: clip(it[0], 200) }), it[1] > 1 ? H('b', { class: 'co-n', text: ` ×${nf(it[1])}` }) : null)));
  if (items.length > 2) ul.append(H('li', { class: 'co-txtm' }, H('button', { type: 'button', class: 'linkbtn', text: `${nf(items.length - 2)} more`, onclick: () => { full.open = true; full.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); } })));
  return H('div', { class: 'co-cr', style: `--fc:var(${FIELD[c][3]})` }, H('span', { class: 'co-crl' }, gl(FIELD[c][2].replace(' ', ''), SAYLAB[c]), H('span', { class: 'n', text: nf(items.length) })), ul);
}
function saysSection(g, sec) {
  const box = H('div', { class: 'co-says' }, H('p', { class: 'small muted', text: 'Loading what its papers say…' })), name = D.g.name[g];
  sec.append(box);
  termOf(g).then((tv) => {
    if (!box.isConnected) return; box.textContent = '';
    if (!tv) { sec.meta.textContent = 'nothing'; box.append(H('p', { class: 'small', text: `Nothing was merged for ${q(name)}.` })); return; }
    const other = FIELDS.reduce((s, f) => s + tv[f[0]].length, 0), P = D.g.papers[g];
    sec.meta.textContent = other ? `${plural(other, 'item')} from ${plural(P, 'paper')}` : 'only definitions';
    if (!other) { box.append(H('p', { class: 'small muted', text: `${P === 1 ? 'Its paper states' : 'Its papers state'} nothing else about it: no broader or narrower terms, other names, related terms, descriptions or examples.` })); return; }
    const full = lazyMore('Every item, with the forms written and its papers', () => fullSays(g, tv));
    for (const c of ['sp', 'sb', 'sy', 'rt']) if (tv[c].length) box.append(chipRow(g, c, tv[c]));
    for (const c of ['de', 'ex']) if (tv[c].length) box.append(textRow(g, c, tv[c], full));
    box.append(H('p', { class: 'small muted' }, gl('agree', '×n'), ': how many papers state it; underlined names open that concept.'), full);
  }).catch((e) => { console.error(e); box.textContent = 'What the papers say could not be loaded: ' + e.message; });
}

/* ------------------------------------------------------------------ the full list of what the papers say (one click away)
   termOf(g) is the merged view of everything the papers of a concept's mentions state about it; relations() links
   the concepts that name each other. Every count and example below is read from them when a concept opens. */
GL.agree = ['Papers that agree (×n)', 'How many papers state the same item for this concept. The bar compares it with the most stated item of its field.', 'Open a concept to see an example from its papers.'];
GL.mergeditem = ['Merged item', 'Items that read the same count as one; every written form is listed, with how many papers wrote it.', 'Open a concept to see an example from its papers.'];
GL.similarc = ['Similar concepts', 'The 8 concepts of the dataset whose definitions are closest in meaning, by the cosine similarity of their vectors.', 'Open a concept to see an example.'];
GL.namelike = ['Name differs only slightly', 'The two names differ only in spelling, by a typo or by a bracket, so they are probably one concept. A reading aid only: nothing is merged.', ''];
GL.sharedlinks = ['Shared links', 'Two concepts share a link when both are linked to the same third concept, in any name field and either direction.', 'Open a concept to see an example.'];
GL.family = ['Family', 'The concept with the concepts linked to it as supertypes and subtypes, by its own papers or by theirs.', 'Open a concept to see an example.'];

const lexGt = (a, b) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] > b[i]; return false; };
const andList = (a) => (a.length < 3 ? a.join(' and ') : a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1]);
const clip = (s, n = 90) => (s.length > n ? s.slice(0, n - 1).trimEnd() + '…' : s);
const SAY_ORDER = ['sp', 'sb', 'df', 'sy', 'rt', 'de', 'ex'];
const KINDN = ['supertype', 'subtype', 'synonym', 'related term'];
// one sentence per field: how n papers state item x of concept c
const SAYS = {
  sp: (n, c, x) => `${plural(n, 'paper')} call ${q(c)} a kind of ${q(x)}`, sb: (n, c, x) => `${plural(n, 'paper')} name ${q(x)} as a kind of ${q(c)}`,
  sy: (n, c, x) => `${plural(n, 'paper')} give ${q(x)} as another name for ${q(c)}`, rt: (n, c, x) => `${plural(n, 'paper')} relate ${q(c)} to ${q(x)}`,
  df: (n, c, x) => `${plural(n, 'paper')} define ${q(c)} as ${q(x)}`, de: (n, c, x) => `${plural(n, 'paper')} describe ${q(c)} with ${q(x)}`,
  ex: (n, c, x) => `${plural(n, 'paper')} give ${q(x)} as an example of ${q(c)}`,
};
function toggle(label, build, into) { // a small "▸ label" button that builds its content on first use and shows or hides it
  let el = null; const b = H('button', { type: 'button', class: 'linkbtn co-tg', 'aria-expanded': 'false', text: label });
  b.addEventListener('click', () => { if (!el) { el = build(); into.append(el); } const open = b.getAttribute('aria-expanded') !== 'true'; b.setAttribute('aria-expanded', String(open)); el.hidden = !open; });
  return b;
}
function paperList(g, n, ms) {
  const box = H('div', { class: 'co-plist' });
  if (ms.length > n) box.append(H('p', { class: 'small muted', text: `${nf(ms.length)} mentions in ${plural(n, 'paper')}: a paper that writes the term in two ways has two mentions of it, and counts once.` }));
  const ul = H('ul');
  for (const i of ms) {
    const pi = D.m.paper[i];
    ul.append(H('li', {}, H('button', { type: 'button', class: 'linkbtn', title: 'open this mention', text: clip(D.p.title[pi] || D.p.id[pi], 110), onclick: () => openMention(i) }),
      D.m.label[i] !== D.g.name[g] ? H('span', { class: 'muted', text: ` as ${q(D.m.label[i])}` }) : null));
  }
  box.append(ul); return box;
}
function itemRow(g, c, it, max) {
  const [text, n, link, vars, ms] = it, li = H('li', { class: 'co-it' + (c === 'df' || c === 'de' || text.length > 70 ? ' sent' : '') });
  const t = H('span', { class: 't' }, text);
  if (link === g) t.append(H('span', { class: 'small muted', text: '  (names this concept itself)' }));
  else if (link >= 0) t.append(H('button', { type: 'button', class: 'linkbtn co-go', title: `open the concept “${D.g.name[link]}”`, 'aria-label': `open the concept ${D.g.name[link]}`, text: '→', onclick: () => openConcept(link) }));
  li.append(H('div', { class: 'co-itr' }, t, H('span', { class: 'co-bar', title: `${plural(n, 'paper')} state this` }, H('i', { style: `width:${Math.max(3, (100 * n) / max)}%` })), H('b', { class: 'co-n', text: `×${nf(n)}` })));
  const tg = H('div', { class: 'co-itt' }), more = H('div', { class: 'co-itx' });
  if (vars.length > 1) tg.append(toggle(`merged from ${vars.length} forms`, () => H('ul', { class: 'co-vars' }, vars.map(([v, k]) => H('li', {}, H('q', { text: v }), H('span', { class: 'muted', text: `  ${plural(k, 'paper')}` })))), more));
  tg.append(toggle(n === 1 ? 'show the paper' : `show the ${nf(n)} papers`, () => paperList(g, n, ms), more));
  li.append(tg, more); return li;
}
function fieldHead(c, label, gk, note) {
  return H('div', { class: 'co-fldh' }, H('h5', {}, gl(gk, label)), H('span', { class: 'small muted', text: note }));
}
function itemList(g, c, items) {
  const ul = H('ul', { class: 'co-items' }), max = items.length ? items[0][1] : 1, box = H('div', { class: 'co-fldl' }, ul);
  items.slice(0, 8).forEach((it) => ul.append(itemRow(g, c, it, max)));
  if (items.length > 8) {
    const b = H('button', { class: 'btn co-all', type: 'button', text: `show all ${nf(items.length)}` });
    b.addEventListener('click', () => { b.remove(); items.slice(8).forEach((it) => ul.append(itemRow(g, c, it, max))); ul.classList.add('long'); });
    box.append(b);
  }
  return box;
}
function fieldBlock(g, c, items) {
  const multi = items.filter((it) => it[1] > 1).length, linked = NAMES.includes(c) ? items.filter((it) => it[2] >= 0).length : 0;
  const note = `${plural(items.length, 'item')}${multi ? `, ${nf(multi)} stated by 2+ papers` : items.length > 1 ? ', each stated by one paper' : ', stated by one paper'}${linked ? `, ${nf(linked)} linked to a concept` : ''}`;
  return H('div', { class: 'co-fld', style: `--fc:var(${FIELD[c][3]})` }, fieldHead(c, FIELD[c][1], FIELD[c][2].replace(' ', ''), note), itemList(g, c, items));
}
function defBlock(g, tv) {
  const kd = D.g.ndef[g], one = tv.nd - tv.df.length, k = D.g.n[g];
  const sec = H('div', { class: 'co-fld', style: '--fc:var(--accent)' }, fieldHead('df', 'Definitions', 'definition', `${plural(tv.nd, 'merged definition')}`));
  const p = H('p', { class: 'small' }, `${k === 1 ? 'Its one mention gives' : `Its ${nf(k)} mentions give`} ${kd === 1 ? 'one definition' : plural(kd, 'definition')}${kd > tv.nd ? `; ${nf(kd - tv.nd)} read the same as another once case, punctuation and spacing are set aside, which leaves ` : kd > 1 ? ', none of them the same as another: ' : ': '}`, H('b', { text: plural(tv.nd, 'merged definition') }), '. ');
  if (tv.df.length) p.append(`${tv.df.length === 1 ? 'One is' : nf(tv.df.length) + ' are'} stated by two or more papers (below, most stated first); the other ${plural(one, 'definition')} ${one === 1 ? 'is' : 'are each'} stated by one paper only.`);
  else if (D.g.papers[g] === 1) p.append(tv.nd === 1 ? 'It comes from one paper, so no other paper can agree with it yet.' : 'They all come from one paper, so no two papers can agree on one yet.');
  else p.append('Each is stated by one paper only: no two papers give the same definition.');
  sec.append(p);
  if (tv.df.length) sec.append(itemList(g, 'df', tv.df));
  return sec;
}
function sayIntro(g, tv, P) {
  const name = D.g.name[g], box = H('div', { class: 'co-intro' });
  box.append(H('p', { class: 'small' }, P === 1 ? 'With one paper every item shows ×1; across several papers, items that read the same become one ' : `Merged across its ${plural(P, 'paper')}: items that read the same become one `, gl('mergeditem', 'merged item'),
    '. Names merge by the ', H('button', { class: 'linkbtn', type: 'button', text: 'term-merging rule', onclick: openHow }),
    '; sentences when they read the same once case, punctuation and spacing are set aside. ', gl('agree', '×n'), ' counts papers; → opens the concept a name stands for.'));
  // the worked example: a merged name stated by many papers, then a merged sentence
  let ex = null, sx = null;
  for (const c of ['sp', 'sb', 'sy', 'rt']) for (const it of tv[c]) { const sc = [it[3].length > 1 ? 1 : 0, it[2] >= 0 && it[2] !== g ? 1 : 0, it[1]]; if (it[1] > 1 && (!ex || lexGt(sc, ex.sc))) ex = { c, it, sc }; }
  for (const c of ['df', 'de', 'ex']) for (const it of tv[c]) { const sc = [it[3].length > 1 ? 1 : 0, c === 'df' ? 1 : 0, it[1]]; if (it[1] > 1 && (!sx || lexGt(sc, sx.sc))) sx = { c, it, sc }; }
  const parts = [];
  if (ex) {
    const [x, n, link, vars] = ex.it;
    parts.push(SAYS[ex.c](n, name, x) + (vars.length > 1 ? `. They write it ${andList(vars.map(([v, c]) => `${q(v)} (${plural(c, 'paper')})`))}: one merged item, ×${nf(n)}` : `, all in the same form: one item, ×${nf(n)}`)
      + (link >= 0 && link !== g ? `, and → opens the concept ${q(D.g.name[link])}.` : '.'));
  }
  if (sx) {
    const [x, n, , vars] = sx.it;
    parts.push((ex ? 'Sentences merge the same way: ' : '') + SAYS[sx.c](n, name, clip(x, 100)) + (vars.length > 1 ? `, written in ${vars.length} forms that differ only in case, punctuation or spacing.` : ', word for word.'));
  }
  const other = FIELDS.reduce((s, f) => s + tv[f[0]].length, 0);
  const text = parts.length ? 'Example: ' + parts.join(' ') : `Example: ${P === 1 ? `${q(name)} has one paper` : `no item of ${q(name)} is stated by two papers`}, so nothing was merged${other ? ' and every item shows ×1' : ''}.`;
  if (parts.length || other) box.append(H('p', { class: 'note', text }));
  GL.agree[2] = `×n counts papers, not mentions: a paper that writes the term twice counts once. ${text}`;
  GL.mergeditem[2] = `Merging only groups items; it removes nothing. Names (supertypes, subtypes, synonyms, related terms) merge by the rule that merges the terms; sentences when they read the same once case, punctuation and spacing are set aside. ${text}`;
  return box;
}
function fullSays(g, tv) {
  const box = H('div', { class: 'co-body co-say' }), P = D.g.papers[g], absent = FIELDS.map((f) => f[0]).filter((c) => !tv[c].length);
  box.append(sayIntro(g, tv, P));
  for (const c of SAY_ORDER) { if (c === 'df') box.append(defBlock(g, tv)); else if (tv[c].length) box.append(fieldBlock(g, c, tv[c])); }
  if (absent.length && absent.length < FIELDS.length) box.append(H('p', { class: 'small muted', text: `Nothing extracted for: ${absent.map((c) => FIELD[c][1].toLowerCase()).join(', ')}.` }));
  return box;
}

/* names that differ only slightly: the same letters once case, accents, spaces, hyphens and punctuation are set aside;
   one letter more or less inside one long word; or the same name plus a bracket that stands as a word of its own */
const foldN = (s) => s.normalize('NFKD').replace(/\p{M}/gu, '').toLowerCase();
const onlyLN = (s) => s.replace(/[^\p{L}\p{N}]/gu, '');
const BRK = /(^|\s)[([][^)\]]*[)\]](?=\s|$)/u, BRKG = new RegExp(BRK.source, 'gu');
function nameKeys(g) {
  CO.nk ||= new Map(); let k = CO.nk.get(g); if (k) return k;
  const s = D.g.name[g], f = foldN(s);
  k = [onlyLN(f), BRK.test(s) ? onlyLN(foldN(s.replace(BRKG, ' '))) : null, f.match(/[\p{L}\p{N}]+/gu) || []];
  CO.nk.set(g, k); return k;
}
function indel1(p, r) { if (p.length > r.length) [p, r] = [r, p]; if (r.length - p.length !== 1) return -1; let i = 0; while (i < p.length && p[i] === r[i]) i++; return r.slice(i + 1) === p.slice(i) ? i : -1; }
function nameLike(a, b) { // 'same' | 'typo' | 'bracket' | null
  const A = nameKeys(a), B = nameKeys(b);
  if (A[0] && A[0] === B[0]) return 'same';
  if ((A[1] == null) !== (B[1] == null)) { const x = A[1] ?? A[0], y = B[1] ?? B[0]; if (x && x === y) return 'bracket'; }
  const wa = A[2], wb = B[2];
  if (wa.length === wb.length) {
    let d = -1, nd = 0; for (let j = 0; j < wa.length; j++) if (wa[j] !== wb[j]) { nd++; d = j; }
    if (nd === 1) { const i = indel1(wa[d], wb[d]), L = Math.max(wa[d].length, wb[d].length); if (i >= 2 && L >= 7 && i <= L - 3) return 'typo'; }
  }
  return null;
}
const LIKE = { // [mark, what the marked names are]
  same: ['probably the same, written differently', 'written with the same letters once case, accents, spaces, hyphens and punctuation are set aside'],
  typo: ['probably the same, one letter apart', 'one letter longer or shorter in one long word, as a typo or another spelling would be'],
  bracket: ['the same name plus a bracket', 'the same name with a bracket added, which is often the same concept and sometimes a narrower one'],
};
function nameScan() { // every marked pair among the 8 most similar concepts of every concept, once per pair
  if (CO.ns) return CO.ns;
  const G = D.g, ns = { same: [0, null], typo: [0, null], bracket: [0, null], total: 0, top1: 0, nC: 0 }, seen = new Set();
  for (let x = 0; x < G.n.length; x++) {
    if (G.cnb[x * 8] < 0) continue;
    ns.top1 += G.cns[x * 8] / 1000; ns.nC++;
    for (let j = 0; j < 8; j++) {
      const y = G.cnb[x * 8 + j]; if (y < 0) continue;
      const key = Math.min(x, y) * 65536 + Math.max(x, y); if (seen.has(key)) continue; seen.add(key);
      const t = nameLike(x, y); if (!t) continue;
      ns[t][0]++; ns.total++;
      const sc = [G.cns[x * 8 + j], Math.min(G.n[x], G.n[y])]; // the example: the most similar pair, then the most mentions
      if (!ns[t][1] || lexGt(sc, ns[t][1].sc)) ns[t][1] = { sc, a: x, b: y };
    }
  }
  ns.top1 /= Math.max(1, ns.nC);
  const eg = (t) => (ns[t][1] ? ` (${q(G.name[ns[t][1].a])} and ${q(G.name[ns[t][1].b])})` : '');
  GL.namelike[2] = `Three kinds of slight difference are marked: the same letters once case, accents, spaces, hyphens and punctuation are set aside${eg('same')}; one letter more or less inside one long word, usually a typo or another spelling${eg('typo')}; and the same name plus a bracket${eg('bracket')}, often the same concept and sometimes a narrower one. Among the 8 most similar concepts of every concept, ${nf(ns.total)} pairs are marked: ${nf(ns.same[0])} written differently, ${nf(ns.typo[0])} one letter apart and ${nf(ns.bracket[0])} with a bracket added. One letter can also make another word, and a bracket can narrow a concept, so a mark is a hint only: term merging did not join these names, and the dataset keeps them as separate concepts.`;
  return (CO.ns = ns);
}

/* links between concepts, from relations(): edges [from, to, field, papers] */
function linksOf(rel, g) { // h -> [sup, sub, syn, rel stated by g's papers, the same four stated by h's papers], from g's side
  const fi = (c) => rel.fields.indexOf(c), sp = fi('sp'), sb = fi('sb'), sy = fi('sy'), M = new Map();
  const add = (h, k, n, own) => { if (h === g) return; let v = M.get(h); if (!v) M.set(h, (v = [0, 0, 0, 0, 0, 0, 0, 0])); v[k + (own ? 0 : 4)] += n; };
  for (const e of rel.out.get(g) || []) { const [, b, f, n] = rel.edges[e]; add(b, f === sp ? 0 : f === sb ? 1 : f === sy ? 2 : 3, n, true); }
  for (const e of rel.in.get(g) || []) { const [a, , f, n] = rel.edges[e]; add(a, f === sp ? 1 : f === sb ? 0 : f === sy ? 2 : 3, n, false); }
  return M;
}
function adjOf(rel) { // every concept -> the set of concepts it is linked to, in any field and either direction
  if (CO.adj) return CO.adj;
  const A = new Map(), add = (a, b) => (A.get(a) || A.set(a, new Set()).get(a)).add(b);
  for (const [a, b] of rel.edges) if (a !== b) { add(a, b); add(b, a); }
  return (CO.adj = A);
}
/* how many DIFFERENT papers state a link, from either side. A paper that defines both concepts can state one link twice,
   once in each concept's entry, so adding the two sides would count it twice. One side alone: its edge already counts
   different papers (one item per edge). Both sides: the union of the papers behind both concepts' items, from termOf,
   loaded only for the concepts with a two-sided link. */
const SIDES = [['sp', 'sb'], ['sb', 'sp'], ['sy', 'sy'], ['rt', 'rt']]; // kind k: the field where g's papers name h, where h's papers name g
const tidx = new WeakMap();
function paperIdx(tv) { // a merged view -> {field: Map(linked concept -> Set of papers)}
  let x = tidx.get(tv); if (x) return x; x = {};
  for (const c of NAMES) { const M = (x[c] = new Map()); for (const it of tv[c]) if (it[2] >= 0) { let ps = M.get(it[2]); if (!ps) M.set(it[2], (ps = new Set())); for (const i of it[4]) ps.add(D.m.paper[i]); } }
  tidx.set(tv, x); return x;
}
async function distinctLinks(g, L, hs, kinds = [0, 1, 2, 3]) { // Map h -> [supertype, subtype, synonym, related term] papers (null: not asked)
  const two = (v) => kinds.some((k) => v[k] && v[k + 4]), need = hs.filter((h) => L.has(h) && two(L.get(h)));
  const [tg, ...tn] = await Promise.all([termOf(g), ...need.map((h) => termOf(h))]);
  const T = new Map(need.map((h, j) => [h, tn[j]])), ig = paperIdx(tg), out = new Map();
  for (const h of hs) {
    const v = L.get(h); if (!v) continue;
    out.set(h, [0, 1, 2, 3].map((k) => {
      if (!kinds.includes(k)) return null;
      const A = ig[SIDES[k][0]].get(h);
      if (!v[k + 4]) return A ? A.size : v[k];
      if (!v[k]) return v[k + 4];
      const B = paperIdx(T.get(h))[SIDES[k][1]].get(g) || new Set(); let n = A ? A.size : 0;
      for (const pp of B) if (!A || !A.has(pp)) n++;
      return n;
    }));
  }
  return out;
}
async function familyOf(g, rel) { // supertypes and subtypes by different papers; a concept named both ways is listed on both sides
  const L = linksOf(rel, g), hs = [...L.keys()].filter((h) => { const v = L.get(h); return v[0] + v[1] + v[4] + v[5] > 0; });
  const P = await distinctLinks(g, L, hs, [0, 1]), sup = [], sub = [];
  for (const h of hs) { const [a, b] = P.get(h); if (a) sup.push([h, a]); if (b) sub.push([h, b]); }
  const by = (x, y) => y[1] - x[1] || x[0] - y[0]; sup.sort(by); sub.sort(by);
  const both = sup.filter(([h]) => P.get(h)[1] > 0).map(([h]) => h), size = hs.length;
  const pick = []; // the map has 12 colours: the concept and up to 11, the supertypes and subtypes stated by the most papers, in turn
  for (let i = 0; pick.length < 11 && (i < sup.length || i < sub.length); i++) for (const lst of [sup, sub]) if (i < lst.length && pick.length < 11 && !pick.includes(lst[i][0])) pick.push(lst[i][0]);
  return { sup, sub, both, pick, P, size };
}
function famRow(label, gk, list) {
  const chips = H('div', { class: 'co-crc' }), K = 12;
  const put = (a) => a.forEach(([h, n]) => chips.insertBefore(H('button', { type: 'button', class: 'fchip co-fc lk', title: `${plural(n, 'paper')} state this link · open “${D.g.name[h]}”`, onclick: () => openConcept(h) }, H('span', { text: clip(D.g.name[h], 60) }), n > 1 ? H('b', { text: `×${nf(n)}` }) : null), mb));
  const mb = list.length > K ? H('button', { type: 'button', class: 'chip co-cmore', text: `+ ${nf(list.length - K)} more`, onclick: () => { put(list.slice(K)); mb.remove(); } }) : null;
  if (mb) chips.append(mb);
  put(list.slice(0, K));
  if (!list.length) chips.append(H('span', { class: 'small muted', text: 'none' }));
  return H('div', { class: 'co-cr', style: `--fc:var(${gk === 'supertype' ? '--f-sp' : '--f-sb'})` }, H('span', { class: 'co-crl' }, gl(gk, label), H('span', { class: 'n', text: nf(list.length) })), chips);
}
function familySection(g, sec) {
  const name = D.g.name[g], box = H('div', { class: 'co-says' }, H('p', { class: 'small muted', text: 'Looking up its family…' }));
  sec.append(box);
  relations().then((rel) => familyOf(g, rel)).then((F) => {
    if (!box.isConnected) return;
    const { sup, sub, both, pick, P, size } = F; box.textContent = '';
    sec.meta.textContent = `${nf(sup.length)} broader · ${nf(sub.length)} narrower`;
    if (!pick.length) { sec.meta.textContent = 'none linked'; box.append(H('p', { class: 'small muted', text: `No broader or narrower concept is linked to it: no kind its papers name is a concept of the dataset, and no concept names it as one.` })); GL.family[2] = `For example, ${q(name)} has no family: no linked supertype or subtype.`; return; }
    const top = (a) => a.slice(0, 3).map(([h, n]) => `${q(D.g.name[h])} ×${nf(n)}`).join(', ') + (a.length > 3 ? ', …' : '');
    const vp = (n, v) => `${nf(n)} ${n === 1 ? 'paper ' + v + 's' : 'papers ' + v}`;
    let bothTxt = '';
    if (both.length) {
      const h = both.reduce((x, y) => (P.get(y)[0] + P.get(y)[1] > P.get(x)[0] + P.get(x)[1] ? y : x)), [a, b] = P.get(h);
      bothTxt = ` ${both.length === 1 ? 'One concept is' : `${nf(both.length)} concepts are`} named both ways and listed on both sides: ${vp(a, 'call')} ${q(D.g.name[h])} broader than ${q(name)}, ${vp(b, 'call')} it narrower.`;
    }
    const text = `${q(name)} has ${plural(sup.length, 'linked supertype')}${sup.length ? ` (${top(sup)})` : ''} and ${plural(sub.length, 'linked subtype')}${sub.length ? ` (${top(sub)})` : ''}, counting what its papers say and what the papers of those concepts say about it. ×n is how many different papers state the link, from either side; a paper that states it from both sides counts once.${bothTxt}`;
    GL.family[2] = `A concept that some papers call broader and others narrower is listed on both sides. Example: ${text}`;
    const col = H('button', { type: 'button', class: 'btn', text: `Colour the family on the map (${nf(pick.length + 1)})`, onclick: () => { for (const h of pick) MAPSEL.add(h); MAPSEL.delete(g); MAPSEL.add(g); show('map'); TABS.map && TABS.map.showConcepts && TABS.map.showConcepts(); } });
    box.append(famRow('Broader', 'supertype', sup), famRow('Narrower', 'subtype', sub),
      H('div', { class: 'co-acts' }, col),
      more('How the family is counted', H('p', { text }), pick.length < size ? H('p', { text: `The map has 12 colours, so the button colours it with the ${nf(pick.length)} stated by the most papers, broader and narrower in turn.` }) : null));
  }).catch((e) => { console.error(e); box.textContent = 'The links between concepts could not be loaded: ' + e.message; });
}

function similarSection(g) {
  const box = H('div', { class: 'co-body co-sim' }), G = D.g, name = G.name[g], rows = [];
  for (let j = 0; j < 8; j++) { const h = G.cnb[g * 8 + j]; if (h >= 0) rows.push([h, G.cns[g * 8 + j] / 1000, nameLike(g, h)]); }
  if (!rows.length) { box.append(H('p', { class: 'small muted', text: 'No similar concepts are listed for it.' })); return box; }
  const ns = CO.ns;
  GL.similarc[2] = `Each concept has one vector: the mean of its mentions’ vectors, where a mention’s vector is the mean of its definitions’ vectors from the Qwen3-Embedding-8B text-embedding model. Cosine similarity runs from about 0 (unrelated) to 1 (the same meaning)${ns ? `; a concept and its most similar one average ${f3(ns.top1)} over the ${nf(ns.nC)} concepts` : ''}. Example: the most similar concept to ${q(name)} is ${q(G.name[rows[0][0]])}, at ${f3(rows[0][1])}.`;
  box.append(H('p', { class: 'small muted' }, 'By ', gl('similarc', 'cosine similarity'), ' of their definitions (1 = the same meaning); bars from 0.5 to 1.'));
  const ol = H('ol', { class: 'co-simlist' }), badges = new Map();
  for (const [h, s, like] of rows) {
    const meta = H('span', { class: 'meta' }, H('span', { text: plural(G.n[h], 'mention') }), like ? H('span', { class: 'co-like' }, gl('namelike', LIKE[like][0])) : null);
    badges.set(h, meta);
    ol.append(H('li', {}, H('button', { type: 'button', class: 'linkbtn co-simn', text: G.name[h], title: `open the concept “${G.name[h]}”`, onclick: () => openConcept(h) }),
      H('span', { class: 'co-bar', title: `cosine ${f3(s)}`, style: '--fc:var(--accent)' }, H('i', { style: `width:${Math.max(3, Math.min(100, 200 * (s - 0.5)))}%` })), H('b', { class: 'co-n', text: f3(s) }), meta));
  }
  box.append(ol);
  const marked = rows.filter((r) => r[2]);
  if (marked.length) {
    const byKind = ['same', 'typo', 'bracket'].map((t) => [t, marked.filter((r) => r[2] === t).map(([h]) => q(G.name[h]))]).filter(([, a]) => a.length);
    box.append(H('p', { class: 'small' }, gl('namelike', 'Marked names'), `: ${byKind.map(([t, a]) => `${andList(a)} ${a.length === 1 ? 'is' : 'are'} ${LIKE[t][1]}`).join('; ')}. Term merging did not join them; the dataset keeps them apart.`));
  }
  // "linked: its related term ×n": the strongest direct link, in different papers from either side
  relations().then((rel) => {
    if (!box.isConnected) return;
    const L = linksOf(rel, g), hs = [...badges.keys()].filter((h) => L.has(h));
    if (hs.length) return distinctLinks(g, L, hs).then((P) => {
      if (!box.isConnected) return;
      for (const h of hs) {
        const n = P.get(h); if (!n) continue;
        let k = 0; for (let j = 1; j < 4; j++) if (n[j] > n[k]) k = j;
        badges.get(h).append(H('span', { class: 'co-lk', title: `different papers stating each link, from either side: ${KINDN.map((t, j) => (n[j] ? `${t} ×${n[j]}` : '')).filter(Boolean).join(', ')}` }, `linked: its ${KINDN[k]} ×${nf(n[k])}`));
      }
    });
  }).catch((e) => console.error(e));
  const shBox = H('div', { class: 'co-shared' }, H('p', { class: 'small muted', text: 'Loading the links between concepts…' }));
  box.append(lazyMore('Concepts that share many links with it', () => { sharedLinks(g, shBox); return shBox; }));
  return box;
}
function sharedLinks(g, sh) {
  const G = D.g, name = G.name[g];
  relations().then((rel) => {
    const L = linksOf(rel, g), A = adjOf(rel), mine = A.get(g) || new Set();
    sh.textContent = '';
    if (!mine.size) { sh.append(H('p', { class: 'small', text: `No concept shares a link with ${q(name)}: none of its names (supertypes, subtypes, synonyms, related terms) is a concept of the dataset, and no concept names it.` })); GL.sharedlinks[2] = `Example: ${q(name)} has no links, so it shares none.`; return; }
    const cnt = new Map();
    for (const x of mine) for (const h of A.get(x)) if (h !== g) cnt.set(h, (cnt.get(h) || 0) + 1);
    const jac = (h, c) => c / (mine.size + A.get(h).size - c);
    const top = [...cnt].filter(([, c]) => c >= 2).sort((a, b) => b[1] - a[1] || jac(b[0], b[1]) - jac(a[0], a[1]) || a[0] - b[0]).slice(0, 8);
    const w = (x) => { const v = L.get(x); return v ? Math.max(...v) : 0; }; // orders the examples only: the larger side's papers
    const egs = (h) => [...mine].filter((x) => A.get(h).has(x)).sort((a, b) => w(b) - w(a) || a - b).slice(0, 3).map((x) => q(G.name[x]));
    if (!top.length) { sh.append(H('p', { class: 'small', text: `${q(name)} is linked to ${plural(mine.size, 'concept')}, but no other concept is linked to two of the same ones.` })); GL.sharedlinks[2] = `Example: ${q(name)} is linked to ${plural(mine.size, 'concept')}, and no other concept shares two of them.`; return; }
    const [h0, c0] = top[0], ex0 = egs(h0);
    const text = `${q(name)} is linked to ${plural(mine.size, 'concept')}. ${q(G.name[h0])} is linked to ${nf(c0)} of them as well, such as ${andList(ex0)}.`;
    GL.sharedlinks[2] = `Concepts that share many links play a similar part in the papers, even when their definitions differ. Example: ${text}`;
    sh.append(H('p', { class: 'small' }, gl('sharedlinks', 'Two concepts share a link'), ' when both are linked to the same third concept. ' + text + ' Such concepts play a similar part in the papers, even when their definitions differ.'));
    const ul = H('ol', { class: 'co-simlist' });
    for (const [h, c] of top) {
      const meta = H('span', { class: 'meta' }, H('span', { text: `both linked to ${c > 3 ? `${egs(h).join(', ')} and ${nf(c - 3)} more` : andList(egs(h))}` }));
      ul.append(H('li', {}, H('button', { type: 'button', class: 'linkbtn co-simn', text: G.name[h], title: `open the concept “${G.name[h]}”`, onclick: () => openConcept(h) }),
        H('span', { class: 'co-bar', title: `${nf(c)} shared of its ${nf(mine.size)} links`, style: '--fc:var(--f-rt)' }, H('i', { style: `width:${Math.max(3, (100 * c) / top[0][1])}%` })), H('b', { class: 'co-n', title: 'shared links', text: nf(c) }), meta));
    }
    sh.append(ul, H('p', { class: 'small muted', text: 'The number is how many linked concepts the two share; the bar compares it with the first row.' }));
  }).catch((e) => { console.error(e); sh.textContent = 'The links between concepts could not be loaded: ' + e.message; });
}

/* how alike a concept's mentions are: its spread on a 0–1 line beside two reference similarities */
function spreadBlock(g) {
  const G = D.g, bl = D.meta.baseline, v = G.pair[g] / 1000, far = G.far[g], pos = (x) => `${(100 * Math.max(0, Math.min(1, x))).toFixed(1)}%`;
  const ref = (x, t, row2) => H('span', { class: 'co-gref' + (x > 0.6 ? ' r' : '') + (row2 ? ' row2' : ''), style: `left:${pos(x)}` }, H('span', { text: t }));
  const gauge = H('div', { class: 'co-gauge', role: 'img', 'aria-label': `spread ${f3(v)}, against two random mentions ${f2(bl.random_pair_cos_mean)} and a mention with its nearest neighbour ${f2(bl.nearest_neighbour_cos_mean)}` },
    H('div', { class: 'co-gtrack' }, H('i', { class: 'co-gfill', style: `width:${pos(v)}` }),
      ref(bl.random_pair_cos_mean, `two random mentions ${f2(bl.random_pair_cos_mean)}`), ref(bl.nearest_neighbour_cos_mean, `a mention and its nearest neighbour ${f2(bl.nearest_neighbour_cos_mean)}`, true),
      H('span', { class: 'co-gdot', style: `left:${pos(v)}` }, H('b', { text: f3(v) }))));
  return H('div', { class: 'co-body' }, gauge,
    H('p', { class: 'small' }, 'Two of its mentions have a ', gl('spread', 'cosine similarity'), ` of ${f3(v)} on average; to the concept’s centre ${f3(G.to_c[g] / 1000)} on average, ${f3(G.min_c[g] / 1000)} at the least. Farthest from the centre: `,
      H('button', { class: 'linkbtn', type: 'button', text: `${D.m.label[far]} in “${clip(D.p.title[D.m.paper[far]] || '', 70)}”`, onclick: () => openMention(far) }), '.'));
}

/* ------------------------------------------------------------------ the expanders below the list */
function expanders() {
  const box = $('#co-xps'), M = CO.M, W = wrongOrder(), nG = D.g.n.length;
  const pts = Array.from({ length: nG }, (_, g) => g).filter((g) => D.g.pair[g] != null);
  CO.pts = pts;
  box.append(
    expander({ id: 'co-how', title: 'How spellings merge into one concept', meta: H('span', {}, H('b', { text: nf(M.groups) }), ' concepts merge 2+ spellings'), build: (b) => howPanel(M, b) }),
    expander({ title: 'What the merging rule cannot do', meta: H('span', {}, H('b', { text: nf(W.length) }), ' word-order merges flagged'), build: (b) => b.append(...limits(M)) }),
    expander({ title: 'Are the mentions of one concept alike?', meta: H('span', {}, H('b', { text: nf(pts.length) }), ' concepts with 2+ mentions'), build: (b) => {
      CO.spLeg = H('div', { class: 'co-legend' }); CO.spEl = H('div', { class: 'chart co-chart', id: 'ch-spread' }); CO.spCap = H('p', { class: 'cap' });
      b.append(CO.spLeg, CO.spEl, CO.spCap); CO.spreadOn = true; spreadChart();
      let lastW = CO.spEl.clientWidth;
      new ResizeObserver(() => { const w = CO.spEl.clientWidth; if (w && Math.abs(w - lastW) > 8) { lastW = w; spreadChart(); } }).observe(CO.spEl);
    } }),
    expander({ title: 'How many mentions a concept has', meta: H('span', {}, H('b', { text: pc(CO.cnt.single / nG) }), ' single-mention'), build: (b) => {
      CO.sizeEl = H('div', { class: 'chart' }); b.append(CO.sizeEl, H('p', { class: 'cap', id: 'co-sizecap' })); sizeChart();
    } }),
    expander({ id: 'co-tablexp', title: 'The results as a table, with every column', meta: H('span', {}, H('b', { text: '8' }), ' sortable columns'), build: tableBuild }));
}

/* the size chart: concepts by their number of mentions; a bar sets the list's size filter */
const SIZE_BINS = [[1, 1], [2, 2], [3, 4], [5, 9], [10, 19], [20, 49], [50, 99], [100, 199], [200, 1e9]];
function sizeChart() {
  const el = CO.sizeEl, G = D.g, vals = bandCounts(Array.from(G.n), SIZE_BINS), nG = G.n.length;
  const bandFor = (a) => BANDS.findIndex(([x, y]) => a >= x && a <= y);
  bars(el, SIZE_BINS.map(bandLabel), vals, { log: true, h: 230, xlabel: 'mentions of a concept (bars on a log scale)', onclick: (i) => { CO.band = bandFor(SIZE_BINS[i][0]); run(); $('#co-facets').scrollIntoView({ behavior: 'smooth', block: 'start' }); } });
  const big = G.n.reduce((b, v, g) => (v > G.n[b] ? g : b), 0), ten = G.n.filter((v) => v >= 10).length;
  const cap = $('#co-sizecap'); if (!cap) return; cap.textContent = '';
  cap.append(`${nf(vals[0])} of ${nf(nG)} concepts (${pc(vals[0] / nG)}) have one mention: one paper defines them. ${nf(ten)} have 10 or more; the most is `, conceptLink(big), ` with ${nf(G.n[big])}. Click a bar to list its concepts.`);
}

/* the table: the list's results with every column, sortable, an optional links column, and what each column means */
// [key, header, sort value, glossary key, what the column means]
const CO_COLS = [
  ['name', 'Concept', (g) => D.g.name[g].toLowerCase(), 'concept', 'The most common spelling names the concept; the grey line lists the other spellings merged into it.'],
  ['nsp', 'Spellings', nSpell, 'spelling', 'How many different spellings were merged. 1: every mention is written the same way.'],
  ['n', 'Mentions', (g) => D.g.n[g], 'mention', 'Its term mentions: each is the term as defined in one paper, with all that paper’s definitions of it.'],
  ['strict', 'Strict', (g) => CO.strictIn[g], 'strict', 'Mentions that also reach the stricter best-F1 cut. One or more makes a strict concept.'],
  ['papers', 'Papers', (g) => D.g.papers[g], null, 'Papers with at least one of its mentions.'],
  ['ndef', 'Definitions', (g) => D.g.ndef[g], 'definition', 'Definition passages over all its mentions.'],
  ['mean_s', 'Mean score', (g) => D.g.mean_s[g], 'ensemble', 'The mean ensemble score of its mentions. Every mention of the dataset reaches the cut, so no mean is below it.'],
  ['pair', 'Spread', (g) => (D.g.pair[g] == null ? -9 : D.g.pair[g]), 'spread', 'The mean cosine similarity of every pair of its mentions; – with a single mention.'],
];
function columnsHelp() {
  const g = CO.exG, G = D.g, dl = H('dl');
  const others = D.merges[g] ? D.merges[g].labels.map((l) => l[0]).filter((l) => l !== G.name[g]) : [];
  const ex = { name: others.length ? `named ${q(G.name[g])}, also written ${others.map(q).join(', ')}` : `written ${q(G.name[g])} only`, nsp: nf(nSpell(g)), strict: nf(CO.strictIn[g]), n: nf(G.n[g]), papers: nf(G.papers[g]), ndef: nf(G.ndef[g]), mean_s: f3(G.mean_s[g] / 10000), pair: G.pair[g] == null ? '–' : f3(G.pair[g] / 1000) };
  for (const [k, nm, , gk, help] of CO_COLS) dl.append(H('dt', {}, gk ? gl(gk, nm) : nm), H('dd', {}, help + ' ', H('span', { class: 'muted', text: k === 'name' ? `Example: ${ex[k]}.` : `Example, ${q(G.name[g])}: ${ex[k]}.` })));
  return [H('p', { class: 'cap', text: `The example is ${q(G.name[g])}, the concept with the most mentions that has both strict and non-strict ones.` }), dl];
}
function tableBuild(body) {
  const sel = H('select', { id: 'co-links' }, H('option', { value: '', text: 'none' }), H('option', { value: 'sp', text: 'supertypes' }), H('option', { value: 'sb', text: 'subtypes' }), H('option', { value: 'all', text: 'all linked concepts' }));
  CO.tb = { sort: 'n', dir: -1, page: 0, help: H('p', { class: 'small muted', role: 'status' }), count: H('p', { class: 'small muted', role: 'status' }), el: H('div', { class: 'tblwrap co-tbl' }), pager: H('div', { class: 'pager' }), ex: H('p', { class: 'co-rowex' }) };
  body.append(H('p', { class: 'small', text: 'The same concepts as the list, with the same search and filters.' }),
    H('div', { class: 'ctl' }, H('label', { class: 'small', for: 'co-links', text: 'Add a links column, most first:' }), sel), CO.tb.help,
    more('What the columns mean', ...columnsHelp()), CO.tb.count, CO.tb.el, CO.tb.pager, CO.tb.ex);
  linkControl(sel);
  drawTable();
}
function coCols() {
  if (!CO.linkCol) return CO_COLS;
  const nm = { sp: 'Supertypes', sb: 'Subtypes', all: 'Links' }[CO.linkCol];
  return [CO_COLS[0], ['links', nm, (g) => CO.lc[CO.linkCol][g], 'link', { sp: 'Concepts linked to it as a supertype, by its papers or by theirs.', sb: 'Concepts linked to it as a subtype, by its papers or by theirs.', all: 'Concepts linked to it in any name field, either direction.' }[CO.linkCol]], ...CO_COLS.slice(1)];
}
function drawTable() {
  const tb = CO.tb; if (!tb) return;
  const G = D.g, R = CO.R.slice(), COLS_ = coCols(), key = (COLS_.find((c) => c[0] === tb.sort) || COLS_.find((c) => c[0] === 'n'))[2];
  if (tb.sort === 'name') { const K = new Map(R.map((g) => [g, key(g)])); R.sort((a, b) => { const x = K.get(a), y = K.get(b); return (x < y ? -1 : x > y ? 1 : a - b) * tb.dir; }); }
  else R.sort((a, b) => { const x = key(a), y = key(b); return (x < y ? -1 : x > y ? 1 : a - b) * tb.dir; });
  tb.count.textContent = `${nf(R.length)} concepts. Click a row to open it.`;
  const PG = 50, pages = Math.max(1, Math.ceil(R.length / PG)); tb.page = Math.min(tb.page, pages - 1);
  const thead = H('tr', {}, COLS_.map(([k, nm, , gk, help], ci) => {
    const th = H('th', { 'aria-sort': tb.sort === k ? (tb.dir > 0 ? 'ascending' : 'descending') : 'none', style: ci ? 'text-align:right' : null });
    const sb = H('button', { type: 'button', text: nm, 'aria-label': `sort by ${nm}`, onclick: () => { tb.dir = tb.sort === k ? -tb.dir : (k === 'name' ? 1 : -1); tb.sort = k; drawTable(); } });
    hoverTip(sb, help);
    const gb = gk ? gl(gk, '?') : null; if (gb) gb.setAttribute('aria-label', `${GL[gk][0]}: what this means`);
    th.append(H('span', { class: 'thh' }, sb, gb)); return th;
  }));
  const tbody = H('tbody'), rows = R.slice(tb.page * PG, (tb.page + 1) * PG);
  for (const g of rows) {
    const st = CO.strictIn[g], sp = D.merges[g] ? D.merges[g].labels.map((l) => l[0]).filter((l) => l !== G.name[g]) : [];
    tbody.append(H('tr', { class: 'row' + (CO.sel === g ? ' sel' : ''), 'data-g': g, tabindex: '0', onclick: () => openConcept(g), onkeydown: (e) => { if (e.key === 'Enter') openConcept(g); } },
      H('td', { class: 'wrapc' }, H('b', { text: G.name[g] }), sp.length ? H('div', { class: 'also', text: 'also: ' + sp.slice(0, 3).join(' · ') + (sp.length > 3 ? ` · +${sp.length - 3} more` : '') }) : null),
      CO.linkCol ? H('td', { class: 'n co-lc', text: nf(CO.lc[CO.linkCol][g]) }) : null,
      H('td', { class: 'n', text: nf(nSpell(g)) }), H('td', { class: 'n', text: nf(G.n[g]) }),
      H('td', { class: 'n strict' + (st ? '' : ' zero'), text: nf(st) }),
      H('td', { class: 'n', text: nf(G.papers[g]) }), H('td', { class: 'n', text: nf(G.ndef[g]) }),
      H('td', { class: 'n', text: f3(G.mean_s[g] / 10000) }), H('td', { class: 'n', text: G.pair[g] == null ? '–' : f3(G.pair[g] / 1000) })));
  }
  if (!rows.length) tbody.append(H('tr', {}, H('td', { colspan: String(COLS_.length), class: 'muted', text: 'Nothing matches the search and filters above.' })));
  tb.el.textContent = ''; tb.el.append(H('table', {}, H('thead', {}, thead), tbody));
  pager(tb.pager, tb, pages, drawTable);
  tb.ex.textContent = '';
  if (rows.length) {
    const g = rows[0];
    tb.ex.append('Reading the first row: ', conceptLink(g), ` is written in ${plural(nSpell(g), 'spelling')} and has ${plural(G.n[g], 'mention')}, ${nf(CO.strictIn[g])} of them strict, from ${plural(G.papers[g], 'paper')} with ${plural(G.ndef[g], 'definition')}. Its mentions score ${f3(G.mean_s[g] / 10000)} on average (the cut is ${D.meta.cut.toFixed(3)})${G.pair[g] == null ? '.' : `, and two of them have a cosine similarity of ${f3(G.pair[g] / 1000)} on average (two random mentions: ${f2(D.meta.baseline.random_pair_cos_mean)}).`}`);
    if (CO.linkCol) { const c = CO.lc[CO.linkCol][g]; tb.ex.append(` It is linked to ${plural(c, 'concept')}${CO.linkCol === 'all' ? '' : ` as ${c === 1 ? 'a ' : ''}${CO.linkCol === 'sp' ? 'supertype' : 'subtype'}${c === 1 ? '' : 's'}`}.`); }
  }
}
function linkCounts(rel) { // per concept: different concepts linked as supertype, as subtype, in any way; named both ways counts on both sides
  const nG = D.g.n.length, fi = (c) => rel.fields.indexOf(c), sp = fi('sp'), sb = fi('sb'), S_ = { sp: new Map(), sb: new Map(), all: new Map() };
  const add = (M, a, b) => (M.get(a) || M.set(a, new Set()).get(a)).add(b);
  for (const [a, b, f] of rel.edges) {
    if (a === b) continue;
    add(S_.all, a, b); add(S_.all, b, a);
    if (f === sp) { add(S_.sp, a, b); add(S_.sb, b, a); } else if (f === sb) { add(S_.sb, a, b); add(S_.sp, b, a); }
  }
  const out = {};
  for (const k in S_) { out[k] = new Uint32Array(nG); for (const [g, set] of S_[k]) out[k][g] = set.size; }
  return out;
}
function linkControl(sel) {
  const help = CO.tb.help;
  sel.addEventListener('change', () => {
    const v = sel.value, tb = CO.tb;
    if (!v) { CO.linkCol = null; if (tb.sort === 'links') { tb.sort = 'n'; tb.dir = -1; } help.textContent = ''; drawTable(); return; }
    help.textContent = 'Counting the links…';
    relations().then((rel) => {
      CO.lc ||= linkCounts(rel);
      CO.linkCol = v; tb.sort = 'links'; tb.dir = -1; tb.page = 0;
      const a = CO.lc[v]; let top = 0, nz = 0; for (let g = 0; g < a.length; g++) { if (a[g]) nz++; if (a[g] > a[top]) top = g; }
      const what = { sp: 'supertype', sb: 'subtype', all: 'linked concept' }[v];
      help.textContent = '';
      help.append(`The ${what === 'linked concept' ? 'Links' : what[0].toUpperCase() + what.slice(1) + 's'} column counts the different concepts linked to a concept as ${what === 'linked concept' ? 'a supertype, subtype, synonym or related term' : 'a ' + what}, whether its own papers name them or theirs name it${what === 'linked concept' ? '' : '; a concept that some papers call broader and others narrower counts on both sides, as in its family'}. A `, gl('link', 'link'), ` needs the name to be a concept of the dataset. ${nf(nz)} concepts have at least one. The most: `, conceptLink(top), ` with ${plural(a[top], what)}.`);
      drawTable();
    }).catch((e) => { console.error(e); help.textContent = 'The links could not be loaded: ' + e.message; });
  });
}

/* ------------------------------------------------------------------ the spread chart */
function spreadChart() {
  const el = CO.spEl, G = D.g; if (!el) return;
  const pts = CO.pts;
  const w = Math.max(300, el.clientWidth || 600), h = w < 480 ? 300 : 340, lw = 40, top = 26, bot = 40, rx = w - 10;
  const bl = D.meta.baseline, lo = Math.max(0, Math.floor((Math.min(bl.random_pair_cos_mean, ...pts.map((g) => G.pair[g] / 1000)) - 0.04) * 10) / 10);
  const maxN = Math.max(...pts.map((g) => G.n[g])), x = lin(1, Math.log2(maxN) + 0.25, lw + 8, rx), y = lin(lo, 1, h - bot, top);
  const svg = S('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': `Spread of ${nf(pts.length)} concepts against their number of mentions` }); el.textContent = ''; el.append(svg);
  for (const v of ticks(lo, 1, 4)) { S('line', { x1: lw, x2: rx, y1: y(v), y2: y(v), class: 'gl' }, svg); T(svg, lw - 6, y(v) + 4, f2(v), 'tx-m', 'end'); }
  T(svg, lw - 30, 12, 'spread: mean similarity of two mentions (cosine)', 'tx-m');
  for (let v = 2; v <= maxN * 1.1; v *= 2) { if (w < 480 && Math.log2(v) % 2 === 0 && v > 2) continue; S('line', { x1: x(Math.log2(v)), x2: x(Math.log2(v)), y1: h - bot, y2: h - bot + 4, class: 'ax' }, svg); T(svg, x(Math.log2(v)), h - bot + 16, String(v), 'tx-m', 'middle'); }
  S('line', { x1: lw, x2: rx, y1: h - bot, y2: h - bot, class: 'ax' }, svg);
  T(svg, (lw + rx) / 2, h - 4, 'mentions of the concept (log scale)', 'tx-m', 'middle');
  const refs = S('g', {}, svg);
  for (const [v, t, dy] of [[bl.random_pair_cos_mean, `two random mentions ${f2(bl.random_pair_cos_mean)}`, -5], [bl.nearest_neighbour_cos_mean, `a mention and its nearest neighbour ${f2(bl.nearest_neighbour_cos_mean)}`, 14]]) { S('line', { x1: lw, x2: rx, y1: y(v), y2: y(v), stroke: css('--ink-2'), 'stroke-dasharray': '4 3' }, refs); T(refs, rx - 2, y(v) + dy, t, 'tx-s', 'end'); }
  const P = new Float32Array(pts.length * 3), dots = S('g', {}, svg);
  pts.forEach((g, j) => {
    const jx = ((g * 2654435761) % 1000) / 1000 - 0.5, cx = x(Math.log2(G.n[g]) + jx * (G.n[g] < 6 ? 0.3 : 0.06)), cy = y(G.pair[g] / 1000), r = Math.min(7, 2 + Math.sqrt(G.n[g]) * 0.5);
    P[j * 3] = cx; P[j * 3 + 1] = cy; P[j * 3 + 2] = r;
    S('circle', { class: 'dot', cx, cy, r, style: `fill:var(--tc${CO.top[g]})`, 'fill-opacity': 0.6 }, dots);
  });
  svg.append(refs); CO.P = P; CO.svg = svg; CO.selG = S('g', {}, svg);
  const hov = S('circle', { class: 'hov', r: 0 }, svg);
  const near = (ev) => {
    const rc = svg.getBoundingClientRect(), sx = ((ev.clientX - rc.left) * w) / rc.width, sy = ((ev.clientY - rc.top) * h) / rc.height;
    let best = -1, bd = 14 * 14;
    for (let j = 0; j < pts.length; j++) { const dx = P[j * 3] - sx, dy = P[j * 3 + 1] - sy, d = dx * dx + dy * dy - P[j * 3 + 2] * P[j * 3 + 2] * 0.5; if (d < bd) { bd = d; best = j; } }
    return best;
  };
  svg.addEventListener('pointermove', (ev) => {
    const j = near(ev); svg.style.cursor = j < 0 ? 'default' : 'pointer';
    if (j < 0) { hov.setAttribute('r', 0); tipEl.hidden = true; return; }
    const g = pts[j]; hov.setAttribute('cx', P[j * 3]); hov.setAttribute('cy', P[j * 3 + 1]); hov.setAttribute('r', P[j * 3 + 2] + 3);
    tipEl.textContent = `${G.name[g]}: ${plural(G.n[g], 'mention')}, ${nf(CO.strictIn[g])} strict; spread ${f3(G.pair[g] / 1000)}${nSpell(g) > 1 ? `; ${nSpell(g)} spellings` : ''}. Click to open.`;
    tipEl.hidden = false; place(tipEl, hov);
  });
  svg.addEventListener('pointerleave', () => { hov.setAttribute('r', 0); tipEl.hidden = true; });
  svg.addEventListener('click', (ev) => { const j = near(ev); if (j >= 0) { tipEl.hidden = true; openConcept(pts[j]); } });
  markSel();
  const lg = CO.spLeg; lg.textContent = '';
  lg.append(H('span', {}, H('span', { class: 'co-dotk' }), 'one concept, in the colour of its main topic'), H('span', {}, H('span', { class: 'co-dash' }), 'reference similarities'), H('span', { class: 'muted', text: 'dot size: mentions' }));
  // the caption, with worked examples read from the dots
  const big = pts.reduce((a, b) => (G.n[b] > G.n[a] ? b : a)), ten = pts.filter((g) => G.n[g] >= 10);
  const low = ten.reduce((a, b) => (G.pair[b] < G.pair[a] ? b : a)), high = ten.reduce((a, b) => (G.pair[b] > G.pair[a] ? b : a));
  const below = pts.filter((g) => G.pair[g] / 1000 < bl.random_pair_cos_mean + 0.1).length, same = pts.filter((g) => G.pair[g] >= 950).length;
  const cap = CO.spCap; cap.textContent = '';
  cap.append(`One dot per concept with two or more mentions; higher means its mentions are defined in more similar words. Most sit well above two random mentions. `, conceptLink(big), `, the concept with the most mentions (${nf(G.n[big])}), sits at ${f3(G.pair[big] / 1000)}. Among concepts with 10 or more mentions, `, conceptLink(high), ` is the most uniform (${f3(G.pair[high] / 1000)}) and `, conceptLink(low),
    ` the least (${f3(G.pair[low] / 1000)}): its papers mean different things by it. ${below === 1 ? '1 concept sits' : nf(below) + ' concepts sit'} within 0.10 of the random line; ${nf(same)} reach 0.95 or more, often because their papers quote one source. Hover a dot for its numbers; click to open it.`);
}
function markSel() {
  if (!CO.selG || !CO.selG.isConnected) return; CO.selG.textContent = '';
  const j = CO.pts.indexOf(CO.sel); if (j < 0) return;
  const P = CO.P, cx = P[j * 3], cy = P[j * 3 + 1], w = +CO.svg.getAttribute('viewBox').split(' ')[2];
  S('circle', { class: 'sel', cx, cy, r: P[j * 3 + 2] + 3.5 }, CO.selG);
  const bg = S('rect', { class: 'selbg', rx: 4 }, CO.selG), t = T(CO.selG, cx > w * 0.7 ? cx - 12 : cx + 12, cy - 10, D.g.name[CO.sel].slice(0, 40), 'tx-s', cx > w * 0.7 ? 'end' : 'start');
  try { const b = t.getBBox(); for (const [k, v] of Object.entries({ x: b.x - 4, y: b.y - 2, width: b.width + 8, height: b.height + 4 })) bg.setAttribute(k, v); } catch (e) { bg.remove(); }
}
TABS.concepts = { init: concepts, open: (g) => openConcept(g) };
