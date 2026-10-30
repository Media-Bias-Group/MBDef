/* MBDef Explorer: the shared core. Every tab file (overview.js, map.js, concepts.js, relevance.js,
   papers.js) is wrapped in its own block by `mbdef explorer build` and registers itself as TABS.<name> = { init(), onShow() }.

   What a tab can use from here:
     D            the data: D.m mentions (columnar), D.g label groups, D.p papers, D.meta, D.merges (spelling traces),
                  derived: D.score[k] / D.rank[k] / D.order[k] for k in s c b j (ensemble, classifier, basal, judge),
                  D.strict, D.members[g] (mention indices of label group g, which is one concept), D.pmentions[p], D.topicOf
                  Every mention is a released one (MBDef's instances table): there are no dropped mentions on the page;
                  D.meta.counts.dropped is how many the relevance filter left out, a number only.
     H(tag, attrs, ...kids), S(svgTag, attrs, parent), T(svg, x, y, text, cls, anchor), seg(el, opts, val, onChange)
     nf f2 f3 pc lin ticks css(token) store(key, value)
     defsOf(i) -> Promise<string[]>   the definitions of mention i (loaded in shards)
     pill(i) mcard(i, extra) scoreBars(i)   a mention's decision pill, card, score bars
     openMention(i)                    the shared drawer
     gl(key, text)                     a glossary term: hover for the short meaning, click for the full one with an example
     GL                                the glossary entries
     show(tab)                         switch tabs; TABS.map.focus(i) / TABS.concepts.open(g) / TABS.papers.open(p) when present
     bars(el, labels, values, opt)     a small bar chart
     SC, SCORE_KEYS, COLS (12 categorical colours), TOPIC_COLS (36)
   Shared components (styles in base.css):
     stat({ value, label, sub, lead, onclick })         a big-number tile (in a .stats grid)
     expander({ title, meta, open, build, id })         <details class="xp">: title and key number always shown; build(body) runs once, on first open
     secHead(title, line, ...aside)                     a section head: h2, one short line, controls on the right
     more(title, ...nodes)                              <details class="more">: a short "how is this made" one click away
     onTheme(fn)                                         fn() runs after the reader switches light/dark (redraw canvases that read css())
     EVAL                                                recall and precision at both cuts on the labelled records
     mapFrame() topicPalette(F) tiered(n, dark) okLab rgbOf rgbStr virLut isDark tshort   the map's frame and colours (see below)
   Loading: core fetches data/meta.json first and calls TABS.overview.hero(meta) at once when the page opens on the
   overview; then it loads the rest, dispatching 'dx:progress' {done, total} and, per mention shard, 'dx:shard' {s, part}
   (part = that shard's columns, e.g. part.xyz, part.topic) on document; every tab's init() runs after all data is in D.
*/
const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
function H(tag, a = {}, ...kids) { const e = document.createElement(tag); for (const k in a) { const v = a[k]; if (v == null) continue; if (k === 'class') e.className = v; else if (k === 'text') e.textContent = v; else if (k === 'html') e.innerHTML = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v); } for (const c of kids.flat()) if (c != null && c !== false) e.append(c); return e; }
const NS = 'http://www.w3.org/2000/svg';
function S(tag, a = {}, p) { const e = document.createElementNS(NS, tag); for (const k in a) if (a[k] != null) e.setAttribute(k, a[k]); if (p) p.append(e); return e; }
const T = (p, x, y, t, cls = 'tx', anc = 'start') => { const e = S('text', { x, y, class: cls, 'text-anchor': anc }, p); e.textContent = t; return e; };
const nf = (x) => Math.round(x).toLocaleString('en-US'), f3 = (x) => (+x).toFixed(3), f2 = (x) => (+x).toFixed(2), pc = (x, d = 1) => (100 * x).toFixed(d) + '%';
const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function seg(el, opts, val, on) { el.textContent = ''; const box = H('span', { class: 'seg', role: 'group' }); for (const [v, t] of opts) { const b = H('button', { type: 'button', 'aria-pressed': String(v === val), text: t }); b.addEventListener('click', () => { $$('button', box).forEach((x) => x.setAttribute('aria-pressed', 'false')); b.setAttribute('aria-pressed', 'true'); on(v); }); box.append(b); } el.append(box); return box; }
function ticks(d0, d1, n = 5) { const st0 = (d1 - d0) / n, mag = Math.pow(10, Math.floor(Math.log10(st0))), r = st0 / mag, st = (r >= 5 ? 5 : r >= 2 ? 2 : 1) * mag, out = []; for (let v = Math.ceil(d0 / st - 1e-9) * st; v <= d1 + 1e-9; v += st) out.push(+v.toFixed(10)); return out; }
function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } return null; }

const D = {};
const TABS = {};
const SC = { s: 'ensemble', c: 'classifier', b: 'basal', j: 'judge' };
const SCORE_KEYS = ['s', 'c', 'b', 'j'];
const COLS = ['#2a78d6', '#e8743b', '#1baf7a', '#c9a100', '#d6589b', '#3f9e3f', '#7a5cd6', '#e34948', '#0aa3b5', '#9c6b3f', '#5b6bd6', '#b0b7bf'];
// 36 topic colours: a perceptually spread wheel (hue steps of the golden angle, two lightness bands)
const TOPIC_COLS = Array.from({ length: 36 }, (_, i) => { const h = (i * 137.508) % 360, l = i % 2 ? 58 : 46, s = i % 3 === 0 ? 62 : 72; return `hsl(${h.toFixed(1)} ${s}% ${l}%)`; });

/* ------------------------------------------------------------------ the glossary */
// [title, one-sentence meaning (hover), meaning plus a real example (click)]. The numbers are the ones
// docs/step-5-post-processing/postprocess.md gives for the relevance ensemble (members, weights, cuts, precision and recall);
// topics: k-means (k = 36) on 50 PCA dimensions of the mention vectors, the same vectors as the map.
// the ensemble's recall and precision on the 1,700 labelled train + dev records of MBDef-Gold, at the cut and at the best-F1
// (strict) cut, as data/frozen/dataset/card-numbers.json records them (train_dev); the page data does not hold the labels
const EVAL = { n: 1700, cut: { r: 0.9509, p: 0.6663 }, strict: { r: 0.811, p: 0.811 } };
const GL = {
  concept: ['Concept', 'One entry of the released dataset: the mentions of one label, merged across papers.', 'A concept gathers every mention whose label is the same after normalisation, with all their definitions, so one concept can hold the definitions of many papers. Merging removes nothing; it only groups. Example: “fake news”, “Fake News”, “Fake news (FN)” and “fake-news” in different papers become the one concept “fake news”.'],
  mention: ['Term mention', 'One term as defined in one paper, with every definition that paper gives it.', 'Mentions are counted before merging. Relevance is decided per mention, because the same label can mean different things in different papers. Example: a paper that defines “framing” twice gives one mention with two definitions; “framing” in a second paper is a second mention. If both are kept, both end up in the one concept “framing”.'],
  definition: ['Definition', 'A sentence or passage in which a paper states what a term means.', 'Definitions were extracted automatically from the papers’ text. The relevance filter keeps or leaves out a mention as a whole, with all its definitions. Example: a survey that defines “misinformation” in its introduction and again in its glossary gives one mention with two definitions.'],
  labelgroup: ['Label group', 'All mentions whose labels are the same after normalisation; on this page each label group is one concept.', 'Normalisation ignores letter case, plural endings, British against American spelling, hyphens and spaces, word order, small words such as “of” and “the”, quotes, and a bracketed acronym after the term. Example: “Fake News”, “fake-news” and “Fake news (FN)” form one label group, the concept “fake news”.'],
  kept: ['Kept', 'The relevance decision for one mention: kept when its ensemble score reaches the cut. MBDef holds only kept mentions, so every mention on this page is kept.', 'Kept means the term, as this paper defines it, belongs in a taxonomy of media bias. The relevance filter left out 5,328 mentions of the corpus whose score missed the cut; they are not part of MBDef, so this page does not show them. Example: every mention here has an ensemble score of at least 0.4150.'],
  strict: ['Strict', 'A kept mention whose score also reaches the stricter best-F1 cut; a strict concept has at least one such mention.', 'The dataset keeps every mention at the cut and marks the surer ones strict. On the 1,700 labelled train + dev records, the cut finds 95.1% of the relevant records (recall 0.951) and 66.6% of what it keeps is relevant (precision 0.666); the best-F1 cut has recall 0.811 and precision 0.811. Use strict concepts when you prefer fewer, surer entries. Example: “social capital” is kept in several papers, but none of its mentions reaches the best-F1 cut, so it is a concept but not a strict one.'],
  cut: ['Cut', 'The ensemble score a mention needs to be kept: 0.4150.', 'Chosen on the 1,700 labelled train + dev records so that 95% of the relevant ones reach it (659 of 693). Example: a mention with an ensemble score of 0.40 is dropped; one with 0.45 is kept.'],
  bestf1: ['Best-F1 cut', 'The stricter cut, 0.5850: the score where F1, which balances precision and recall, is highest on the labelled records.', 'Kept mentions at or above it are marked strict. On the labelled train + dev records it has precision 0.811 and recall 0.811, against 0.666 and 0.951 at the cut: fewer irrelevant mentions, more relevant ones missed. Example: a mention scoring 0.50 is kept but not strict; one scoring 0.70 is strict.'],
  ensemble: ['Ensemble score', 'The relevance score that decides kept or dropped: a weighted mean of three models’ ranks, from 0 to 1.', 'Each model’s score is first placed among that model’s scores on the 1,700 labelled train + dev records, as a rank from 0 (below all of them) to 1 (above all), so the three scales become comparable. Then s = 0.344·r(classifier) + 0.339·r(basal) + 0.316·r(judge), with the weights fitted on the same records. Example: a mention that all three models rank above 60% of the labelled records scores about 0.60.'],
  classifier: ['Classifier', 'A logistic regression that predicts relevance from the meaning of a mention’s definitions; its score is the predicted probability of “relevant”.', 'Each definition, written as “term: definition”, becomes a vector of 4,096 numbers from the Qwen3-Embedding-8B text-embedding model; the mention’s vector is the mean over its definitions. The regression was trained on the 1,700 labelled train + dev records. Example: a mention with three definitions is scored once, on the mean of its three vectors.'],
  basal: ['basal', 'A large language model (basal-omni-1.0-ultra) that sorts a term into core, satellite or irrelevant for a taxonomy of media bias; its score is P(core) + P(satellite).', 'Core: a type of bias, such as confirmation bias. Satellite: not a bias itself, but a concept needed to describe, detect or explain bias, such as framing. The model reads the term with its first three definitions and is not trained on our labels. Example: a score of 0.90 means the model gives “irrelevant” a probability of 0.10.'],
  judge: ['Judge', 'A language model that reads one definition at a time next to 32 similar labelled examples and answers yes, uncertain or no.', 'The 32 examples are labelled training records chosen by embedding similarity, never from the same paper; the model (Gemma 4, 12 billion parameters, 4-bit weights) is not trained. A definition scores P(yes) + ½·P(uncertain), and a mention takes the highest score over its definitions. Example: a mention whose two definitions score 0.30 and 0.80 gets 0.80.'],
  seed: ['Seed paper', 'A paper of the media-bias literature collection the corpus started from.', 'The cited papers were added from the seed papers’ reference lists. Example: “Fake news attributions as a source of nonspecific structure” is a seed paper.'],
  closure: ['Cited paper', 'A paper that a seed paper cites, added to the corpus by following the citations (the citation closure).', 'Cited papers bring in the terms the seed papers build on. Example: “Social Science Research: Principles, Methods and Practices” is a cited paper.'],
  paper: ['Paper', 'One publication with at least one term mention in MBDef: a seed paper or a cited paper.', 'A paper is in the dataset when at least one of its mentions is kept. Example: most papers define a handful of terms; a few long works, such as books and surveys with a glossary, define over a hundred.'],
  single: ['Single-mention concept', 'A concept with exactly one mention, so only one paper defines it.', 'Most concepts are of this kind. Example: “viralization” is defined in one paper of the corpus only.'],
  spelling: ['Spelling', 'One exact form of a label as written in a paper, before normalisation.', 'Several spellings can merge into one concept. Example: “Fake News”, “fake-news” and “Fake news (FN)” are three spellings of the concept “fake news”.'],
  spread: ['Spread', 'How alike the mentions of one label group are in meaning: the mean cosine similarity between every pair of their vectors.', 'Cosine similarity runs from about 0 (unrelated) to 1 (same meaning). Two random mentions average 0.28, and a mention and its nearest neighbour 0.80. A spread near 1 means the label is defined in nearly the same words everywhere; near 0.3, it means different things in different papers. Example: the mentions of “fake news” are far more alike than two random mentions.'],
  topic: ['Topic', 'One of 36 groups of mentions with similar definitions, named after their most frequent concepts; the colours of the map.', 'Found by k-means clustering (k = 36) of the vectors of MBDef’s mentions, reduced to 50 dimensions with PCA; the map’s layout is a UMAP of the same vectors. Topics ignore the labels, so different terms with similar definitions share a topic, and one label can fall in several. Example: of the 58 mentions labelled “satire”, 45 are in the topic “clickbait · satire · sarcasm” and 11 in “fake news · hoaxes · false news”.'],
  neighbours: ['Nearest mentions', 'The 8 other mentions of MBDef whose definition vectors are most similar to this one’s (cosine similarity).', 'Often the same concept in other papers, or a closely related one. Two random mentions average a cosine of 0.28, a mention and its nearest neighbour 0.80. Example: a neighbour at cosine 0.90 is defined in very similar words.'],
  percentile: ['Percentile', 'The share of MBDef’s mentions that score lower under the chosen score.', 'Example: a mention at the 90th percentile of the ensemble score scores higher than 90% of MBDef’s mentions.'],
};
Object.assign(GL, {
  supertype: ['Supertype', 'What a paper says the term is a kind of.', 'Extracted from the paper with the term: many papers list “disinformation” as a supertype of “fake news”. Items that read the same are merged, and ×n says how many papers state one.'],
  subtype: ['Subtype', 'A kind of the term that a paper names.', 'Extracted with the term; the reverse of a supertype. Several papers often list the same subtypes.'],
  synonym: ['Synonym', 'Another name a paper gives for the term.', 'Extracted with the term. Synonyms with different words are not merged into one concept; the explorer shows them as links between concepts.'],
  relatedterm: ['Related term', 'A term a paper relates to this one, without saying how.', 'The most common name field: many papers relate “misinformation” to “fake news”.'],
  description: ['Description', 'What a paper says the term is or does, besides its definitions.', 'Extracted with the term: a sentence that describes it without defining it.'],
  example: ['Example', 'An example a paper gives of the term.', 'Extracted with the term: a phrase or sentence the paper offers as an instance.'],
  link: ['Link between concepts', 'A name item that is itself one of the dataset\'s concepts.', 'When a paper lists “disinformation” as a supertype of “fake news”, and “disinformation” is a concept, the two concepts are linked; the link\'s weight is how many papers state it.'],
});
const tipEl = H('div', { class: 'gtip', role: 'tooltip', hidden: '' });
const popEl = H('div', { class: 'gpop', role: 'dialog', hidden: '' });
function gl(key, text) {
  const g = GL[key]; if (!g) return H('span', { text: text || key });
  const b = H('button', { type: 'button', class: 'g', 'aria-label': `${text || g[0]}: what this means`, text: text || g[0] });
  b.addEventListener('pointerenter', () => { tipEl.textContent = g[1]; tipEl.hidden = false; place(tipEl, b); });
  b.addEventListener('pointerleave', () => { tipEl.hidden = true; });
  b.addEventListener('focus', () => { tipEl.textContent = g[1]; tipEl.hidden = false; place(tipEl, b); });
  b.addEventListener('blur', () => { tipEl.hidden = true; });
  b.addEventListener('click', (e) => { e.stopPropagation(); tipEl.hidden = true; popEl.textContent = ''; popEl.append(H('div', { class: 'gpop-h' }, H('b', { text: g[0] }), H('button', { type: 'button', class: 'btn', text: 'Close', onclick: () => (popEl.hidden = true) })), H('p', { text: g[1] }), H('p', { class: 'muted', text: g[2] })); popEl.hidden = false; place(popEl, b); });
  return b;
}
function place(el, anchor) { const r = anchor.getBoundingClientRect(), w = el.offsetWidth || 300, h = el.offsetHeight || 80; el.style.left = Math.max(8, Math.min(innerWidth - w - 8, r.left)) + 'px'; el.style.top = (r.bottom + h + 12 > innerHeight ? r.top - h - 8 : r.bottom + 8) + 'px'; }
addEventListener('click', (e) => { if (!popEl.hidden && !popEl.contains(e.target)) popEl.hidden = true; });

/* ------------------------------------------------------------------ the other extracted fields
   FIELDS: [code, label, singular, colour token]; codes as in data/fields and data/terms. NAMES are the fields whose items
   are names and can link to a concept. fieldsOf(i): a mention's raw items {code: [[text, citations]]}. termOf(g): a
   concept's merged view {nd, df, de, sy, rt, sb, sp, ex}, each item [text, papers, linked group or -1, variants
   [[text, papers]], mention indices]. relations(): {fields, edges
   [[from g, to g, field index into fields, papers]], out: g -> edge indices, in: g -> edge indices}. */
const FIELDS = [['sp', 'Supertypes', 'supertype', '--f-sp'], ['sb', 'Subtypes', 'subtype', '--f-sb'], ['sy', 'Synonyms', 'synonym', '--f-sy'],
  ['rt', 'Related terms', 'related term', '--f-rt'], ['de', 'Description', 'description', '--f-de'], ['ex', 'Examples', 'example', '--f-ex']];
const FIELD = Object.fromEntries(FIELDS.map((f) => [f[0], f]));
const NAMES = ['sy', 'rt', 'sb', 'sp'];
const fieldCache = {}, termCache = {};
let relPromise = null;
function fieldsOf(i) { const s = Math.floor(i / D.meta.shard); return (fieldCache[s] ||= fetch(`data/fields/${String(s).padStart(2, '0')}.json`).then((r) => r.json())).then((a) => a[i - s * D.meta.shard] || {}); }
function termOf(g) { const n = D.meta.fields.terms_shard, s = Math.floor(g / n); return (termCache[s] ||= fetch(`data/terms/${String(s).padStart(2, '0')}.json`).then((r) => r.json())).then((a) => a[g - s * n]); }
function relations() {
  return (relPromise ||= fetch('data/relations.json').then((r) => r.json()).then((rel) => {
    rel.out = new Map(); rel.in = new Map();
    rel.edges.forEach((e, k) => { (rel.out.get(e[0]) || rel.out.set(e[0], []).get(e[0])).push(k); (rel.in.get(e[1]) || rel.in.set(e[1], []).get(e[1])).push(k); });
    return rel;
  }));
}
function fieldChip(code, text, papers, link, onLink) {
  const c = H('span', { class: 'fchip', style: `--fc:var(${FIELD[code][3]})` }, H('span', { text }));
  if (papers > 1) c.append(H('b', { class: 'mono', title: `${papers} papers state this`, text: `×${papers}` }));
  if (link >= 0 && onLink) c.append(H('button', { type: 'button', class: 'linkbtn small', title: `open the concept “${D.g.name[link]}”`, text: '→', onclick: () => onLink(link) }));
  return c;
}
/* what else the paper says about mention i: its raw items, with the merged view's paper counts and links */
function fieldsBlock(i) {
  const box = H('div', { class: 'fblock' }, H('p', { class: 'small muted', text: 'loading…' }));
  const gi = D.m.group[i];
  Promise.all([fieldsOf(i), termOf(gi)]).then(([raw, tv]) => {
    box.textContent = '';
    const codes = FIELDS.map((f) => f[0]).filter((c) => raw[c] && raw[c].length);
    if (!codes.length) { box.append(H('p', { class: 'small muted', text: 'The extraction lists nothing else for this term in this paper.' })); return; }
    const open = (g) => { show('concepts'); TABS.concepts && TABS.concepts.open && TABS.concepts.open(g); $('#drawer').hidden = true; };
    for (const c of codes) {
      const merged = tv ? tv[c].filter((it) => it[4].includes(i)) : [];
      const sec = H('div', { class: 'fsec' }, H('h4', {}, gl(FIELD[c][2].replace(' ', ''), FIELD[c][1]), H('span', { class: 'small muted', text: ` ${raw[c].length}` })));
      if (NAMES.includes(c)) {
        const row = H('div', { class: 'chips' });
        for (const [t] of raw[c]) { const m = merged.find((it) => it[0] === t || it[3].some((v) => v[0] === t)); row.append(fieldChip(c, t, m ? m[1] : 1, m ? m[2] : -1, open)); }
        sec.append(row);
      } else {
        const ol = H('ul', { class: 'flist' });
        for (const [t, cit] of raw[c]) ol.append(H('li', {}, t, cit && cit.length ? H('span', { class: 'small muted', text: `  (${cit.join('; ')})` }) : null));
        sec.append(ol);
      }
      box.append(sec);
    }
    if (tv) box.append(H('p', { class: 'small muted', text: '×n: that many papers state the same item for this concept (merged across spellings); → opens the concept the item names.' }));
  });
  return box;
}

/* ------------------------------------------------------------------ definitions, cards and the drawer */
const shardCache = {};
function defsOf(i) { const s = Math.floor(i / D.meta.shard); return (shardCache[s] ||= fetch(`data/defs/${String(s).padStart(2, '0')}.json`).then((r) => r.json())).then((a) => a[i - s * D.meta.shard]); }
function pill(i) { return H('span', { class: 'pill ' + (D.strict[i] ? 'strict' : 'kept'), text: D.strict[i] ? 'kept, strict' : 'kept' }); }
function scoreBars(i) {
  const box = H('div', { class: 'bars' });
  for (const k of SCORE_KEYS) {
    const v = D.score[k][i];
    const tr = H('span', { class: 'tr' }, H('i', { style: `width:${100 * v}%;background:${k === 's' ? 'var(--accent)' : 'var(--muted)'}` }));
    if (k === 's') tr.append(H('b', { style: `left:${100 * D.meta.cut}%` }));
    box.append(gl(SC[k] === 'ensemble' ? 'ensemble' : SC[k], SC[k]), tr, H('span', { class: 'mono', text: f3(v) }));
  }
  return box;
}
function mcard(i, extra) {
  const card = H('button', { class: 'mcard', type: 'button' }, H('span', { class: 't', text: D.m.label[i] }), H('span', { class: 'd', text: '…' }),
    H('span', { class: 'm' }, pill(i), H('span', { class: 'mono', text: `ensemble ${f3(D.score.s[i])}` }), extra ? H('span', { text: extra }) : null));
  defsOf(i).then((d) => { $('.d', card).textContent = (d[0] || '').slice(0, 220) + ((d[0] || '').length > 220 ? '…' : '') + (d.length > 1 ? `  (+${d.length - 1} more)` : ''); });
  card.addEventListener('click', () => openMention(i));
  return card;
}
function openMention(i) {
  const dr = $('#drawer'); dr.hidden = false; dr.textContent = '';
  const gi = D.m.group[i], pi = D.m.paper[i], tp = D.meta.topics[D.m.topic[i]];
  dr.append(H('div', { class: 'ctl', style: 'justify-content:space-between' }, H('span', { class: 'small muted' }, gl('mention')), H('button', { class: 'btn', type: 'button', text: 'Close', onclick: () => (dr.hidden = true) })),
    H('h2', { text: D.m.label[i] }), H('div', { class: 'ctl' }, pill(i), H('span', { class: 'small muted', text: `${D.m.ndef[i]} definition${D.m.ndef[i] > 1 ? 's' : ''}` })),
    scoreBars(i), H('p', { class: 'small muted', text: `The line on the ensemble bar is the cut (${D.meta.cut.toFixed(4)}); kept mentions past ${D.meta.best_f1_cut.toFixed(4)} are also strict.` }));
  const ol = H('ol', { class: 'defs' }, H('li', { text: 'loading…' }));
  dr.append(H('h3', { text: 'Definitions in this paper' }), ol);
  defsOf(i).then((d) => { ol.textContent = ''; for (const t of d) ol.append(H('li', { text: t })); });
  dr.append(H('h3', { text: 'What else this paper says about it' }), fieldsBlock(i));
  dr.append(H('dl', { class: 'dl' },
    H('dt', { text: 'Paper' }), H('dd', {}, H('button', { class: 'linkbtn', type: 'button', text: D.p.title[pi] || D.p.id[pi], onclick: () => { show('papers'); TABS.papers && TABS.papers.open && TABS.papers.open(pi); dr.hidden = true; } })),
    H('dt', {}, gl('seed', 'Corpus')), H('dd', { text: D.p.corpus[pi] === 'seeds' ? 'seed paper' : 'cited (closure) paper' }),
    H('dt', {}, gl('concept')), H('dd', {}, H('button', { class: 'linkbtn', type: 'button', text: `${D.g.name[gi]} · ${nf(D.g.n[gi])} mention${D.g.n[gi] > 1 ? 's' : ''}`, onclick: () => { show('concepts'); TABS.concepts && TABS.concepts.open && TABS.concepts.open(gi); dr.hidden = true; } })),
    H('dt', {}, gl('topic')), H('dd', { text: tp ? tp.name : '–' }),
    H('dt', { text: 'Mention id' }), H('dd', { class: 'mono', text: D.m.mention_id[i] })));
  const nbl = H('ol', { class: 'nblist' });
  for (let k = 0; k < 8; k++) { const j = D.m.nb[i * 8 + k], s = D.m.nbs[i * 8 + k] / 1000; nbl.append(H('li', {}, H('button', { type: 'button', class: 'linkbtn', text: D.m.label[j], onclick: () => openMention(j) }), H('span', { class: 'small muted', text: `  cos ${f3(s)}${D.m.group[j] === gi ? ' · same concept' : ''}` }))); }
  dr.append(H('h3', {}, gl('neighbours', 'Nearest mentions in embedding space')), nbl, H('div', { class: 'ctl' }, H('button', { class: 'btn', type: 'button', text: 'Show on the map', onclick: () => { show('map'); TABS.map && TABS.map.focus && TABS.map.focus(i); dr.hidden = true; } })));
  dr.scrollTop = 0;
}
addEventListener('keydown', (e) => { if (e.key === 'Escape') { $('#drawer').hidden = true; popEl.hidden = true; } });

/* ------------------------------------------------------------------ a small bar chart */
function bars(el, labels, values, opt = {}) {
  const w = Math.max(300, el.clientWidth || 600), h = opt.h || 220, lw = 52, top = 22, bot = 36;
  const max = Math.max(...values), x = lin(-0.5, labels.length - 0.5, lw, w - 8), y = opt.log ? ((v) => lin(0, Math.log10(max) + 0.05, h - bot, top)(Math.log10(Math.max(v, 1)))) : lin(0, max * 1.08, h - bot, top);
  const bw = Math.min(54, (w - lw) / labels.length * 0.66);
  const svg = S('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img' }); el.textContent = ''; el.append(svg);
  const tk = opt.log ? [1, 10, 100, 1000, 10000, 100000].filter((v) => v <= max * 1.1) : ticks(0, max, 4);
  for (const v of tk) { S('line', { x1: lw, x2: w - 8, y1: y(v), y2: y(v), class: 'gl' }, svg); T(svg, lw - 6, y(v) + 4, nf(v), 'tx-m', 'end'); }
  labels.forEach((l, i) => { const r = S('rect', { x: x(i) - bw / 2, y: y(values[i]), width: bw, height: Math.max(0, y(opt.log ? 1 : 0) - y(values[i])), rx: 2, fill: opt.color || css('--accent') }, svg); if (opt.onclick) { r.style.cursor = 'pointer'; r.addEventListener('click', () => opt.onclick(i)); } T(svg, x(i), y(values[i]) - 5, nf(values[i]), 'tx-m', 'middle'); T(svg, x(i), h - bot + 15, l, 'tx-m', 'middle'); });
  if (opt.xlabel) T(svg, (lw + w) / 2, h - 4, opt.xlabel, 'tx-m', 'middle');
  return svg;
}

/* ------------------------------------------------------------------ colours and the map frame (shared by the map and the overview)
   mapFrame(): positions scaled so 98% of the dots lie within radius 10 (pos3, pos2), each topic's anchor (anc3, anc2), its
   colour slot and members; computed once. topicPalette(F, dark): one [r, g, b] per topic, for a dark ground when dark is true
   (any map on the stage), else for the reader's theme. */
const okToLin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
const okFromLin = (c) => { const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.max(c, 0) ** (1 / 2.4) - 0.055; return Math.round(Math.min(1, Math.max(0, v)) * 255); };
function okLab([r, g, b]) {
  const R = okToLin(r), G = okToLin(g), B = okToLin(b);
  const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B), m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B), s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
function okLinear(L, a, b) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3, s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s];
}
function okLch(L, C, hDeg) { // the most saturated colour of this lightness and hue that sRGB can show
  const h = (hDeg * Math.PI) / 180;
  for (let c = C; c >= 0; c -= 0.004) { const v = okLinear(L, c * Math.cos(h), c * Math.sin(h)); if (v.every((x) => x >= -0.0005 && x <= 1.0005)) return v.map(okFromLin); }
  return okLinear(L, 0, 0).map(okFromLin);
}
const okDist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const rgbStr = (c) => `rgb(${c[0]} ${c[1]} ${c[2]})`;
const rgbCtx = document.createElement('canvas').getContext('2d');
function rgbOf(s) { // any CSS colour to [r, g, b] 0-255
  rgbCtx.fillStyle = '#808080'; rgbCtx.fillStyle = String(s || '').trim() || '#808080';
  const v = rgbCtx.fillStyle; if (v[0] === '#') return [1, 3, 5].map((k) => parseInt(v.slice(k, k + 2), 16));
  const m = v.match(/[\d.]+/g); return m ? m.slice(0, 3).map((x) => Math.round(+x)) : [128, 128, 128];
}
// many categories: evenly spaced hues in lightness tiers (12 hues x 3 tiers for 36 topics); lighter tiers on a dark ground
function tiered(n, dark) {
  const tiers = Math.min(4, Math.max(1, Math.ceil(n / 12))), k = Math.ceil(n / tiers), out = [];
  const Ls = dark ? [0.74, 0.62, 0.86, 0.54] : [0.6, 0.47, 0.74, 0.4], Cs = [0.145, 0.13, 0.115, 0.12];
  for (let t = 0; t < tiers; t++) for (let j = 0; j < k; j++) out.push(okLch(Ls[t], Cs[t], 28 + (j * 360) / k + (t * 180) / k));
  return out;
}
// score colours: viridis, without its palest end on a light ground and its darkest end on a dark ground
const VIR = [[0, [68, 1, 84]], [0.25, [59, 82, 139]], [0.5, [33, 145, 140]], [0.75, [94, 201, 98]], [1, [253, 231, 37]]];
function virAt(t) { t = Math.max(0, Math.min(1, t)); for (let k = 1; k < VIR.length; k++) if (t <= VIR[k][0]) { const [a, ca] = VIR[k - 1], [b, cb] = VIR[k], u = (t - a) / (b - a); return ca.map((v, j) => Math.round(v + (cb[j] - v) * u)); } return VIR[VIR.length - 1][1]; }
const virRange = (dark) => (dark ? [0.14, 1] : [0, 0.9]);
function virLut(dark) { const [lo, hi] = virRange(dark); return Array.from({ length: 256 }, (_, k) => virAt(lo + ((hi - lo) * k) / 255)); }
function virCss(dark) { const [lo, hi] = virRange(dark); return `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((u) => `${rgbStr(virAt(lo + (hi - lo) * u))} ${u * 100}%`).join(', ')})`; }
const isDark = () => okLab(rgbOf(css('--bg')))[0] < 0.5;
const trunc = (s, n) => (s.length > n ? s.slice(0, n - 1).replace(/\s+\S*$/, '') + '…' : s);
const tshort = (t) => D.meta.topics[t].top.slice(0, 2).join(' · ');

/* ------------------------------------------------------------------ positions, topic anchors and topic colour slots (no three.js needed) */
let mapFrameCache = null;
function mapFrame() { return (mapFrameCache ||= mapFrameOf()); }
function mapFrameOf() {
  const N = D.m.n, T = D.m.topic, NT = D.meta.topics.length;
  function fit(src, dims) { // centre on the mean; scale so that 98% of the dots lie within radius 10
    const out = new Float32Array(N * 3), m = [0, 0, 0], r = new Float32Array(N);
    for (let i = 0; i < N; i++) for (let a = 0; a < dims; a++) m[a] += src[i * dims + a];
    for (let a = 0; a < dims; a++) m[a] /= N;
    for (let i = 0; i < N; i++) { let s = 0; for (let a = 0; a < dims; a++) { const v = src[i * dims + a] - m[a]; out[i * 3 + a] = v; s += v * v; } r[i] = Math.sqrt(s); }
    const k = 10 / (r.sort()[Math.floor(0.98 * (N - 1))] || 1);
    for (let j = 0; j < N * 3; j++) out[j] *= k;
    return out;
  }
  const pos3 = fit(D.m.xyz, 3), pos2 = fit(D.m.xy, 2);
  const members = Array.from({ length: NT }, () => []);
  for (let i = 0; i < N; i++) members[T[i]].push(i);
  function anchors(P) { // the member nearest each topic's coordinate-wise median: always on the topic, even a curved one
    const out = new Float32Array(NT * 3);
    members.forEach((ix, t) => {
      if (!ix.length) return;
      const med = [0, 1, 2].map((a) => { const v = Float32Array.from(ix, (i) => P[i * 3 + a]).sort(); return v[v.length >> 1]; });
      let best = ix[0], bd = Infinity;
      for (const i of ix) { const d = (P[i * 3] - med[0]) ** 2 + (P[i * 3 + 1] - med[1]) ** 2 + (P[i * 3 + 2] - med[2]) ** 2; if (d < bd) { bd = d; best = i; } }
      out.set(P.subarray(best * 3, best * 3 + 3), t * 3);
    });
    return out;
  }
  const anc3 = anchors(pos3), anc2 = anchors(pos2);
  // colour slots: the biggest topic first; each takes the free colour farthest (in OKLab) from its 6 nearest coloured neighbours
  const pal = tiered(NT, false).map(okLab), size = members.map((ix) => ix.length), order = [...size.keys()].sort((a, b) => size[b] - size[a]);
  const slot = new Int16Array(NT).fill(-1), used = new Set();
  const dist = (a, b) => Math.hypot(anc3[a * 3] - anc3[b * 3], anc3[a * 3 + 1] - anc3[b * 3 + 1], anc3[a * 3 + 2] - anc3[b * 3 + 2]);
  for (const t of order) {
    const near = order.filter((o) => slot[o] >= 0).map((o) => [dist(t, o), o]).sort((a, b) => a[0] - b[0]).slice(0, 6);
    let best = -1, bs = -1;
    for (let s = 0; s < pal.length; s++) {
      if (used.has(s)) continue;
      let sc = Infinity; near.forEach(([, o], k) => { sc = Math.min(sc, okDist(pal[s], pal[slot[o]]) * (1 + 0.12 * k)); });
      if (sc > bs + 1e-9) { bs = sc; best = s; }
    }
    slot[t] = best < 0 ? t % pal.length : best; used.add(slot[t]);
  }
  return { N, T, NT, pos3, pos2, anc3, anc2, slot, size, members };
}
// dark = true for a map on the stage (dark in both themes); by default the reader's theme
function topicPalette(F, dark = isDark()) { const pal = tiered(F.NT, dark); return Array.from({ length: F.NT }, (_, t) => pal[F.slot[t]]); }

/* ------------------------------------------------------------------ shared components */
function stat({ value, label, sub, lead, onclick }) {
  const kids = [H('span', { class: 'sv', text: typeof value === 'number' ? nf(value) : value }), H('span', { class: 'sl', text: label }), sub ? H('span', { class: 'ss' }, sub) : null];
  return onclick ? H('button', { type: 'button', class: 'stat' + (lead ? ' lead' : ''), onclick }, ...kids) : H('div', { class: 'stat' + (lead ? ' lead' : '') }, ...kids);
}
function expander({ title, meta, open, build, id }) {
  const body = H('div', { class: 'xp-body' }), d = H('details', { class: 'xp', id }, H('summary', {}, H('span', { class: 'xp-t' }, title), meta != null ? H('span', { class: 'xp-m' }, meta) : null), body);
  let built = false;
  const run = () => { if (built || !build) return; built = true; try { build(body); } catch (e) { console.error(e); body.append(H('p', { class: 'note', text: 'This could not be drawn: ' + e.message })); } };
  d.addEventListener('toggle', () => { if (d.open) run(); });
  if (open) { d.open = true; requestAnimationFrame(run); }
  return d;
}
function secHead(title, line, ...aside) {
  return H('div', { class: 'sec-head' }, H('div', {}, typeof title === 'string' ? H('h2', { text: title }) : title, line ? H('p', {}, line) : null), aside.length ? H('div', { class: 'ctl' }, ...aside) : null);
}
function more(title, ...nodes) { return H('details', { class: 'more' }, H('summary', { text: title || 'How this is made' }), H('div', {}, ...nodes)); }
const themeFns = [];
function onTheme(fn) { themeFns.push(fn); }
function isDarkTheme() { const t = document.documentElement.dataset.theme; return t ? t === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches; }
$('#theme-btn') && $('#theme-btn').addEventListener('click', () => {
  const next = isDarkTheme() ? 'light' : 'dark';
  document.documentElement.dataset.theme = next; try { localStorage.setItem('dx-theme', next); } catch (e) { /* the switch still works for this visit */ }
  for (const fn of themeFns) { try { fn(); } catch (e) { console.error(e); } }
  document.dispatchEvent(new Event('dx:theme'));
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (!document.documentElement.dataset.theme) { for (const fn of themeFns) { try { fn(); } catch (e) { console.error(e); } } document.dispatchEvent(new Event('dx:theme')); } });

/* ------------------------------------------------------------------ tabs and boot */
const TAB_ORDER = ['overview', 'map', 'concepts', 'relations', 'relevance', 'papers'];
const started = {};
let pendingTab = null, heroShown = false;
function show(tab) {
  for (const b of $$('nav.tabs button')) b.setAttribute('aria-selected', String(b.dataset.tab === tab));
  store('dx-tab', tab);
  try { history.replaceState(null, '', '#' + tab); } catch (e) { /* a sandboxed page keeps its address */ }
  if (!D.ready) {   // still loading: the overview's hero can show, any other view waits behind the loading line
    pendingTab = tab;
    const early = tab === 'overview' && heroShown;
    for (const s of $$('section.view')) s.hidden = !(early && s.id === 'v-overview');
    $('#loading').hidden = early;
    return;
  }
  for (const s of $$('section.view')) s.hidden = s.id !== 'v-' + tab;
  const t = TABS[tab]; if (!t) return;
  if (!started[tab]) { started[tab] = true; try { t.init(); } catch (e) { console.error(e); $('#v-' + tab).append(H('p', { class: 'note', text: 'This view could not be drawn: ' + e.message })); } }
  if (t.onShow) t.onShow();
}
for (const b of $$('nav.tabs button')) b.addEventListener('click', () => show(b.dataset.tab));
$('.brand').addEventListener('click', (e) => { e.preventDefault(); show('overview'); scrollTo({ top: 0 }); });

async function boot() {
  document.body.append(tipEl, popEl);
  const get = (n) => fetch(`data/${n}.json`).then((r) => { if (!r.ok) throw new Error(`data/${n}.json: ${r.status}`); return r.json(); });
  const start = (location.hash || '').slice(1), first = TAB_ORDER.includes(start) ? start : (TAB_ORDER.includes(store('dx-tab')) ? store('dx-tab') : 'overview');
  const meta = await get('meta');
  D.meta = meta;
  // the overview can show its headline numbers before the rest arrives
  if (first === 'overview' && TABS.overview && TABS.overview.hero) {
    for (const b of $$('nav.tabs button')) b.setAttribute('aria-selected', String(b.dataset.tab === 'overview'));
    $('#v-overview').hidden = false; $('#loading').hidden = true;
    try { TABS.overview.hero(meta); heroShown = true; } catch (e) { console.error(e); }
  }
  // the mentions table comes in shards of meta.shard mentions, data/mentions/00.json, … (as data/defs/): join their columns
  const nShards = Math.ceil(meta.counts.terms / meta.shard), total = nShards + 3;
  let done = 0;
  const tick = (x) => { done++; const f = done / total; $('#progress i').style.width = (100 * f).toFixed(1) + '%'; const pc = $('#loading-pc'); if (pc) pc.textContent = `${Math.round(100 * f)}%`; document.dispatchEvent(new CustomEvent('dx:progress', { detail: { done, total } })); return x; };
  const shard = (s) => get(`mentions/${String(s).padStart(2, '0')}`).then((part) => { tick(); document.dispatchEvent(new CustomEvent('dx:shard', { detail: { s, part } })); return part; });
  const [parts, g, p, merges] = await Promise.all([Promise.all(Array.from({ length: nShards }, (_, s) => shard(s))), get('groups').then(tick), get('papers').then(tick), get('merges').then(tick)]);
  $('#progress').classList.add('done');
  const m = Object.fromEntries(Object.keys(parts[0]).map((k) => [k, k === 'n' ? parts.reduce((t, q) => t + q.n, 0) : [].concat(...parts.map((q) => q[k]))]));
  Object.assign(D, { m, g, p, merges });
  const N = m.n;
  D.score = {}; D.rank = {}; D.order = {};
  for (const k of SCORE_KEYS) {
    D.score[k] = Float32Array.from(m[k], (v) => v / 10000);
    const ord = Array.from({ length: N }, (_, i) => i).sort((a, b) => D.score[k][a] - D.score[k][b]);
    const r = new Float32Array(N); ord.forEach((i, j) => (r[i] = (j + 0.5) / N)); D.rank[k] = r; D.order[k] = ord;
  }
  D.strict = Uint8Array.from(m.strict);   // from the full-precision score; the page's scores are rounded to 4 decimals
  D.members = Array.from({ length: g.n.length }, () => []);
  m.group.forEach((gi, i) => D.members[gi].push(i));
  D.pmentions = Array.from({ length: p.id.length }, () => []);
  m.paper.forEach((pi, i) => D.pmentions[pi].push(i));
  $('#loading').hidden = true;
  D.ready = true;
  window.DX = { D, TABS, GL, MAPSEL, show, openMention, fieldsOf, termOf, relations, stat, expander, secHead, more };   // for the browser console
  show(pendingTab || first);
  $('#foot').textContent = `Built from MBDef, the Media Bias Definitions Dataset: its ${nf(meta.counts.terms)} term mentions, those whose relevance ensemble score reaches the cut ${meta.cut.toFixed(4)}; the spellings of one label are merged by the label-match rule; the map is a UMAP of these mentions’ vectors (the mean Qwen3-Embedding-8B vector of a mention's definitions).`;
}

/* ------------------------------------------------------------------ shared by several tabs */
const MAPSEL = new Set();            // label groups coloured on the map (Concepts tab adds, Map tab draws)
function pager(el, st, pages, draw) { el.textContent = ''; const pv = H('button', { class: 'btn', type: 'button', text: '← previous', onclick: () => { st.page--; draw(); } }), nx = H('button', { class: 'btn', type: 'button', text: 'next →', onclick: () => { st.page++; draw(); } }); pv.disabled = st.page === 0; nx.disabled = st.page >= pages - 1; el.append(pv, H('span', { class: 'small muted', text: `page ${st.page + 1} of ${pages}` }), nx); }
function cardsFor(ix, el, what) { el.textContent = ''; for (const i of ix) el.append(mcard(i, what ? what(i) : '')); }
function bandCounts(vals, bands) { return bands.map(([a, b]) => vals.reduce((s, v) => s + (v >= a && v <= b ? 1 : 0), 0)); }
const bandLabel = ([a, b]) => (a === b ? String(a) : b >= 1e6 ? `${a}+` : `${a}–${b}`);
