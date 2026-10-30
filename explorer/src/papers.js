/* papers tab: registers TABS.papers = { init, open }. Uses the helpers of core.js. The drawer calls TABS.papers.open(p) right
   after show('papers'). D.p holds per paper: id, title, corpus ('seeds' | 'closure'), n (term mentions, all released), ndef.
   Layout: a stage head with the headline numbers; the centrepiece, a list of papers to search, filter and sort, with tiny
   bars, and the open paper beside it (wide screens) or under its row (narrow); the scatter, the size distribution and the
   terms only one paper defines in expanders below. A paper's "terms" on this tab are its term mentions. */
const PA = { q: [], corpus: new Set(), bands: new Set(), sort: 'n', shown: 25, sel: -1, tsort: 's', tstrict: false };
const PAX = {};   // derived once: per paper strict and only-here counts, lower-case titles, size band; the scatter's dots
const PA_STEP = 25, PA_TERMS = 30;
const PA_BANDS = [[1, 1], [2, 4], [5, 9], [10, 19], [20, 49], [50, 1e9]];
const PA_SORTS = [['n', 'terms'], ['ndef', 'definitions'], ['strict', 'strict'], ['only', 'only here'], ['title', 'title']];
const PA_SORTED = { n: 'most terms first', ndef: 'most definitions first', strict: 'most strict terms first', only: 'most only-here terms first', title: 'by title' };
const PA_WIDE = matchMedia('(min-width: 980px)');
const paTitle = (p) => D.p.title[p] || D.p.id[p];
const paShort = (t, n = 90) => (t.length > n ? t.slice(0, n - 1).trimEnd() + '…' : t);
const paMed = (a) => { const s = Array.from(a).sort((x, y) => x - y); return s[Math.floor((s.length - 1) / 2)]; };
const paSeed = (p) => D.p.corpus[p] === 'seeds';
const paCorpus = (p) => (paSeed(p) ? gl('seed', 'seed paper') : gl('closure', 'cited paper'));
const paSw = (p) => H('i', { class: 'pa-sw ' + (paSeed(p) ? 'pa-seed' : 'pa-cited'), 'aria-hidden': 'true' });
const paQ = (p) => PA.q.every((w) => PAX.lt[p].includes(w) || PAX.lid[p].includes(w));
const paC = (p) => !PA.corpus.size || PA.corpus.has(D.p.corpus[p]);
const paB = (p) => !PA.bands.size || PA.bands.has(PAX.band[p]);
const paShown = (p) => paQ(p) && paC(p) && paB(p);
const paToList = () => $('#pa-browse').scrollIntoView({ behavior: 'smooth', block: 'start' });
const paConcept = (g) => { show('concepts'); TABS.concepts && TABS.concepts.open && TABS.concepts.open(g); };

function papers() {
  const P = D.p, n = P.id.length, G = D.g.n.length;
  const strict = new Uint32Array(n), only = new Uint32Array(n), first = new Int32Array(G).fill(-1), multi = new Uint8Array(G);
  for (let i = 0; i < D.m.n; i++) {
    const p = D.m.paper[i], g = D.m.group[i]; if (D.strict[i]) strict[p]++;
    if (first[g] < 0) first[g] = p; else if (first[g] !== p) multi[g] = 1;
  }
  for (let g = 0; g < G; g++) if (first[g] >= 0 && !multi[g]) only[first[g]]++;
  const lt = P.title.map((t, p) => paTitle(p).toLowerCase()), lid = P.id.map((s) => s.toLowerCase());
  const tk = lt.map((t) => t.replace(/^[^\p{L}\p{N}]+/u, ''));
  const band = Int8Array.from(P.n, (v) => PA_BANDS.findIndex(([a, b]) => v >= a && v <= b));
  let big = 0, bigOnly = 0; for (let p = 1; p < n; p++) { if (P.n[p] > P.n[big]) big = p; if (only[p] > only[bigOnly]) bigOnly = p; }
  Object.assign(PAX, { strict, only, lt, lid, tk, band, big, bigOnly, det: $('#pa-detail') });
  GL.onlyhere = ['Only here', 'A concept that no other paper defines: every mention of it comes from this one paper.',
    `Most concepts are of this kind; they are what a paper adds that the rest of the corpus does not have. Example: “${paShort(paTitle(bigOnly), 70)}” defines ${nf(only[bigOnly])} concepts that no other paper defines, more than any other paper.`];
  paHero(); paTools(); paDraw(); paXps();
  PA_WIDE.addEventListener('change', () => { if (PA.sel < 0) paEmpty(); paPlace(); });
  onTheme(() => { if (PA.sel >= 0 && PAX.topics) paTopics(PA.sel, PAX.topics); });
  // on a wide screen the paper pane is never empty at first: it shows the first paper of the list (unless open(p) came first)
  requestAnimationFrame(() => { if (PA.sel < 0) { if (PA_WIDE.matches && PAX.R.length) paOpen(PAX.R[0], false); else paEmpty(); } });
}

/* ---- the head: headline numbers on the stage */
function paHero() {
  const P = D.p, n = P.id.length, seeds = P.corpus.filter((c) => c === 'seeds').length, big = PAX.big;
  const terms = P.n.reduce((s, v) => s + v, 0), defs = P.ndef.reduce((s, v) => s + v, 0);
  const only = (c) => () => { PA.corpus = new Set([c]); paChange(); paToList(); };
  const el = $('#pa-hero'); el.textContent = '';
  el.append(H('div', { class: 'pa-hh' }, H('span', { class: 'eyebrow', text: 'Where the definitions come from' }), H('h2', { text: 'The papers behind MBDef' }),
    H('p', {}, `Each of the ${nf(defs)} definitions comes from one of these papers. Open one to see every term it defines.`)),
  H('div', { class: 'stats' },
    stat({ value: n, label: 'papers', sub: H('span', {}, nf(terms), ' ', gl('mention', 'term mentions')), lead: true }),
    stat({ value: seeds, label: 'seed papers', sub: 'the media-bias collection · list them', onclick: only('seeds') }),
    stat({ value: n - seeds, label: 'cited papers', sub: 'cited by seed papers · list them', onclick: only('closure') }),
    stat({ value: paMed(P.n), label: 'terms in a typical paper', sub: `median; ${nf(paMed(P.ndef))} definitions` }),
    stat({ value: P.n[big], label: 'terms in the largest paper', sub: paShort(paTitle(big), 58), onclick: () => paOpen(big, true) })));
}

/* ---- the controls: search, facet chips with live counts, sort */
function paTools() {
  const q = $('#pa-q');
  q.addEventListener('input', () => { PA.q = q.value.trim().toLowerCase().split(/\s+/).filter(Boolean); paChange(); });
  q.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { const r = $('.pa-row', $('#pa-list')); if (r) { e.preventDefault(); r.focus(); } }
    else if (e.key === 'Enter' && PAX.R.length) paOpen(PAX.R[0], false);
  });
  const fx = $('#pa-facets'); fx.textContent = ''; PAX.chipC = {}; PAX.chipB = [];
  const cg = H('div', { class: 'chips', role: 'group', 'aria-label': 'corpus' }, H('span', { class: 'eyebrow pa-lab', text: 'Corpus' }));
  for (const [c, t, cls] of [['seeds', 'seed', 'pa-seed'], ['closure', 'cited', 'pa-cited']]) {
    const b = H('button', { type: 'button', class: 'chip', 'aria-pressed': 'false', onclick: () => { PA.corpus.has(c) ? PA.corpus.delete(c) : PA.corpus.add(c); paChange(); } },
      H('i', { class: 'pa-sw ' + cls, 'aria-hidden': 'true' }), t, H('span', { class: 'n' }));
    PAX.chipC[c] = b; cg.append(b);
  }
  const bg = H('div', { class: 'chips', role: 'group', 'aria-label': 'number of terms in the paper' }, H('span', { class: 'eyebrow pa-lab', text: 'Terms' }));
  PA_BANDS.forEach((bd, k) => {
    const b = H('button', { type: 'button', class: 'chip', 'aria-pressed': 'false', onclick: () => { PA.bands.has(k) ? PA.bands.delete(k) : PA.bands.add(k); paChange(); } }, bandLabel(bd), H('span', { class: 'n' }));
    PAX.chipB.push(b); bg.append(b);
  });
  fx.append(cg, bg);
  seg($('#pa-sort'), PA_SORTS, PA.sort, (v) => { PA.sort = v; paChange(); });
  $('#pa-clear').addEventListener('click', () => { PA.q = []; q.value = ''; PA.corpus.clear(); PA.bands.clear(); paChange(); });
  const lh = $('#pa-lhead'); lh.textContent = '';
  lh.append(H('span', { class: 'pa-hp' }, 'Paper ', H('span', { class: 'pa-key' }, H('span', {}, H('i', { class: 'pa-sw pa-seed' }), 'seed'), H('span', {}, H('i', { class: 'pa-sw pa-cited' }), 'cited'))),
    H('span', {}, gl('mention', 'Terms'), H('span', { class: 'pa-key' }, H('span', {}, H('i', { class: 'pa-sw pa-sw-s' }), gl('strict', 'strict')))),
    H('span', {}, gl('definition', 'Definitions')), H('span', { class: 'pa-ho' }, gl('onlyhere', 'Only here')));
  $('#pa-list').addEventListener('keydown', (e) => {
    if ((e.key !== 'ArrowDown' && e.key !== 'ArrowUp') || !e.target.classList.contains('pa-row')) return;
    const rows = $$('.pa-row', $('#pa-list')), k = rows.indexOf(e.target), nx = rows[k + (e.key === 'ArrowDown' ? 1 : -1)];
    e.preventDefault(); if (nx) nx.focus(); else if (e.key === 'ArrowUp') $('#pa-q').focus();
  });
  $('#pa-moreline').after(more('How this list is made',
    H('p', {}, 'A paper’s terms are its ', gl('mention', 'term mentions'), ': each term it defines, with every definition it gives it. Two spellings of one concept in the same paper count as two terms.'),
    H('p', {}, 'Bars are to scale against the largest paper in the list; the violet part of a terms bar is the ', gl('strict', 'strict'), ' terms. ', gl('onlyhere', 'Only here'), ' counts the concepts that no other paper defines.'),
    H('p', {}, 'The counts on the filter chips follow the search and the other filters. Titles are as recorded in the corpus; a few carry extraction errors, such as a stray author name.')));
}
function paChange() { PA.shown = PA_STEP; paDraw(); }

/* ---- the list */
function paDraw() {
  const P = D.p, n = P.id.length, R = [], cC = { seeds: 0, closure: 0 }, cB = new Array(PA_BANDS.length).fill(0);
  for (let p = 0; p < n; p++) {
    if (!paQ(p)) continue;
    const okC = paC(p), okB = paB(p);
    if (okB) cC[P.corpus[p]]++;
    if (okC) cB[PAX.band[p]]++;
    if (okC && okB) R.push(p);
  }
  const key = { n: P.n, ndef: P.ndef, strict: PAX.strict, only: PAX.only }[PA.sort];
  if (key) R.sort((a, b) => key[b] - key[a] || P.n[b] - P.n[a] || a - b);
  else R.sort((a, b) => (PAX.tk[a] < PAX.tk[b] ? -1 : PAX.tk[a] > PAX.tk[b] ? 1 : a - b));
  PAX.R = R;
  for (const c in PAX.chipC) { PAX.chipC[c].setAttribute('aria-pressed', String(PA.corpus.has(c))); $('.n', PAX.chipC[c]).textContent = nf(cC[c]); }
  PAX.chipB.forEach((b, k) => { b.setAttribute('aria-pressed', String(PA.bands.has(k))); $('.n', b).textContent = nf(cB[k]); });
  const any = PA.q.length || PA.corpus.size || PA.bands.size;
  $('#pa-clear').hidden = !any;
  $('#pa-count').textContent = `${any ? `${nf(R.length)} of ${nf(n)}` : nf(n)} papers, ${PA_SORTED[PA.sort]}`;
  paRows(); paMark();
}
function paRows() {
  const P = D.p, R = PAX.R, list = $('#pa-list'); list.textContent = '';
  let mn = 1, md = 1; for (const p of R) { if (P.n[p] > mn) mn = P.n[p]; if (P.ndef[p] > md) md = P.ndef[p]; }
  for (const p of R.slice(0, PA.shown)) list.append(paRow(p, mn, md));
  if (!R.length) list.append(H('p', { class: 'pa-none', text: 'No paper matches: try fewer words, or clear the filters.' }));
  const ml = $('#pa-moreline'); ml.textContent = '';
  if (R.length > PA.shown) ml.append(H('button', { type: 'button', class: 'btn', text: `Show ${nf(Math.min(PA_STEP, R.length - PA.shown))} more`, onclick: () => { PA.shown += PA_STEP; paRows(); } }),
    H('span', { class: 'small muted', text: `${nf(R.length - PA.shown)} more in the list` }));
  paPlace();
}
function paRow(p, mn, md) {
  const P = D.p, n = P.n[p], s = PAX.strict[p], d = P.ndef[p], o = PAX.only[p];
  const sum = `${nf(n)} term${n === 1 ? '' : 's'} (${nf(s)} strict), ${nf(d)} definition${d === 1 ? '' : 's'}, ${nf(o)} only here`;
  return H('button', { type: 'button', class: 'pa-row', 'data-p': p, 'aria-current': String(PA.sel === p), title: sum,
    'aria-label': `${paTitle(p)}. ${paSeed(p) ? 'Seed' : 'Cited'} paper: ${sum}`, onclick: () => paOpen(p, false) },
  H('span', { class: 'pa-rt' }, paSw(p), paHi(paTitle(p))),
  H('span', { class: 'pa-rn' }, H('span', { class: 'pa-bar' }, s ? H('i', { class: 's', style: `width:${(100 * s) / mn}%` }) : null, n > s ? H('i', { style: `width:${(100 * (n - s)) / mn}%` }) : null), H('span', { class: 'num', text: nf(n) })),
  H('span', { class: 'pa-rd' }, H('span', { class: 'pa-bar d' }, H('i', { style: `width:${(100 * d) / md}%` })), H('span', { class: 'num', text: nf(d) })),
  H('span', { class: 'pa-ro num', text: o ? nf(o) : '–' }));
}
// the title with the search words marked
function paHi(t) {
  if (!PA.q.length) return t;
  const lt = t.toLowerCase(); if (lt.length !== t.length) return t;
  const rg = [];
  for (const w of PA.q) for (let k = lt.indexOf(w); k >= 0; k = lt.indexOf(w, k + w.length)) rg.push([k, k + w.length]);
  if (!rg.length) return t;
  rg.sort((a, b) => a[0] - b[0]);
  const out = []; let at = 0;
  for (const [a, b] of rg) { if (b <= at) continue; const a2 = Math.max(a, at); if (a2 > at) out.push(t.slice(at, a2)); out.push(H('mark', { text: t.slice(a2, b) })); at = b; }
  out.push(t.slice(at));
  return H('span', {}, ...out);
}
function paRowMark() { for (const r of $$('.pa-row', $('#pa-list'))) r.setAttribute('aria-current', String(+r.dataset.p === PA.sel)); }
// the paper pane: beside the list on a wide screen, under the open paper's row on a narrow one
function paPlace() {
  const det = PAX.det; if (!det) return;
  if (PA_WIDE.matches) { if (det.parentNode !== $('#pa-side')) $('#pa-side').append(det); det.hidden = false; return; }
  if (PA.sel < 0) { det.hidden = true; return; }
  det.hidden = false;
  const row = $(`.pa-row[data-p="${PA.sel}"]`, $('#pa-list'));
  if (row) { if (row.nextSibling !== det) row.after(det); } else if ($('#pa-list').firstChild !== det) $('#pa-list').prepend(det);
}

/* ---- one paper: its numbers, its topics and its terms, each opening the drawer or its concept */
function paOpen(p, jump) {
  if (!PAX.strict) return;
  PA.sel = p;
  if (jump) { const at = PAX.R.indexOf(p); if (at >= PA.shown) { PA.shown = Math.ceil((at + 1) / PA_STEP) * PA_STEP; paRows(); } }
  paRowMark(); paDetail(p); paPlace(); paMark();
  if (jump) { if (PA_WIDE.matches) paToList(); else PAX.det.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  else if (!PA_WIDE.matches) PAX.det.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
function paClose() {
  const p = PA.sel; PA.sel = -1; paRowMark(); paMark();
  if (PA_WIDE.matches) paEmpty(); else { paPlace(); const r = $(`.pa-row[data-p="${p}"]`, $('#pa-list')); if (r) r.focus(); }
}
function paEmpty() {
  const det = PAX.det; det.textContent = ''; det.classList.add('empty');
  const R = PAX.R.length ? PAX.R : D.p.id.map((_, p) => p);
  det.append(H('span', { class: 'eyebrow', text: 'One paper' }), H('h3', { text: 'Pick a paper from the list' }),
    H('p', { class: 'small muted', text: 'Its terms open here, each with its definitions and its concept.' }),
    H('div', { class: 'ctl' }, H('button', { type: 'button', class: 'btn', text: 'The largest paper', onclick: () => paOpen(PAX.big, false) }),
      H('button', { type: 'button', class: 'btn', text: 'A random paper', onclick: () => paOpen(R[Math.floor(Math.random() * R.length)], false) })));
}
const paFact = (v, label, cls) => H('span', { class: 'pa-fact' }, H('b', { class: cls, text: nf(v) }), label);
function paDetail(p) {
  const P = D.p, det = PAX.det, ms = D.pmentions[p];
  det.textContent = ''; det.classList.remove('empty'); det.scrollTop = 0;
  const topics = H('div', { class: 'pa-topics' });
  det.append(H('div', { class: 'pa-dh' }, H('div', { class: 'pa-dt' }, H('span', { class: 'eyebrow' }, paSw(p), paCorpus(p)), H('h3', { text: paTitle(p) })),
    H('button', { type: 'button', class: 'btn pa-x', text: 'Close', 'aria-label': 'close this paper', onclick: paClose })),
  H('div', { class: 'pa-facts' }, paFact(P.n[p], gl('mention', P.n[p] === 1 ? 'term' : 'terms')), paFact(PAX.strict[p], gl('strict', 'strict'), 'c-s'),
    paFact(P.ndef[p], gl('definition', P.ndef[p] === 1 ? 'definition' : 'definitions')), paFact(PAX.only[p], gl('onlyhere', 'only here'))), topics);
  PAX.topics = topics; paTopics(p, topics);
  const sortEl = H('span'), list = H('div', { class: 'pa-terms' }), foot = H('div', { class: 'pa-tfoot' });
  const strictChip = H('button', { type: 'button', class: 'chip', 'aria-pressed': String(PA.tstrict), onclick: () => { PA.tstrict = !PA.tstrict; strictChip.setAttribute('aria-pressed', String(PA.tstrict)); fill(false); } },
    'strict only', H('span', { class: 'n', text: nf(PAX.strict[p]) }));
  const fill = (all) => {
    const S2 = D.score.s, L = D.m.label;
    const ix = ms.filter((i) => !PA.tstrict || D.strict[i]);
    if (PA.tsort === 'az') ix.sort((a, b) => L[a].localeCompare(L[b], 'en', { sensitivity: 'base' }));
    else if (PA.tsort === 'nd') ix.sort((a, b) => D.m.ndef[b] - D.m.ndef[a] || S2[b] - S2[a]);
    else ix.sort((a, b) => S2[b] - S2[a]);
    list.textContent = ''; foot.textContent = '';
    for (const i of all ? ix : ix.slice(0, PA_TERMS)) list.append(paTerm(i));
    if (!ix.length) list.append(H('p', { class: 'small muted pa-none', text: 'None of its terms is strict.' }));
    if (!all && ix.length > PA_TERMS) foot.append(H('button', { type: 'button', class: 'btn', text: `Show all ${nf(ix.length)} terms`, onclick: () => fill(true) }));
  };
  seg(sortEl, [['s', 'score'], ['az', 'A–Z'], ['nd', 'definitions']], PA.tsort, (v) => { PA.tsort = v; fill(false); });
  det.append(H('div', { class: 'pa-th' }, H('h4', { text: 'Its terms' }), sortEl, strictChip), list, foot,
    H('p', { class: 'small muted pa-id' }, 'Click a term for its definitions and scores, or its concept for every paper that defines it. Paper id ', H('span', { class: 'mono', text: P.id[p] })));
  fill(false);
}
// the topics of a paper's terms: one bar in the map's topic colours, the three largest named
function paTopics(p, el) {
  const F = mapFrame(), pal = topicPalette(F), ms = D.pmentions[p], tc = new Map();
  for (const i of ms) tc.set(D.m.topic[i], (tc.get(D.m.topic[i]) || 0) + 1);
  const arr = [...tc.entries()].sort((a, b) => b[1] - a[1]), nm = (t) => (D.meta.topics[t] ? D.meta.topics[t].name : String(t));
  el.textContent = '';
  const bar = H('div', { class: 'pa-tbar', role: 'img', 'aria-label': 'its terms by topic: ' + arr.map(([t, c]) => `${nm(t)} ${c}`).join('; ') });
  for (const [t, c] of arr) bar.append(H('i', { style: `width:${(100 * c) / ms.length}%;background:${rgbStr(pal[t])}`, title: `${nm(t)}: ${c} term${c === 1 ? '' : 's'}` }));
  const keys = H('div', { class: 'pa-tkeys' }, H('span', { class: 'eyebrow pa-lab' }, gl('topic', 'Topics')));
  for (const [t, c] of arr.slice(0, 3)) keys.append(H('span', { title: nm(t) }, H('i', { style: `background:${rgbStr(pal[t])}` }), tshort(t), H('span', { class: 'n', text: ` ${c}` })));
  if (arr.length > 3) keys.append(H('span', { class: 'muted', text: `+${arr.length - 3} more` }));
  el.append(keys, bar);
}
function paTerm(i) {
  const g = D.m.group[i], nd = D.m.ndef[i], lab = D.m.label[i], np = D.g.papers[g];
  const d = H('span', { class: 'pa-td', text: '…' });
  defsOf(i).then((a) => { d.textContent = (a && a[0]) || ''; });
  const same = D.g.name[g].toLowerCase() === lab.toLowerCase();
  return H('div', { class: 'pa-term' },
    H('button', { type: 'button', class: 'pa-tb', onclick: () => openMention(i), title: 'its definitions and scores' }, H('span', { class: 'pa-tl', text: lab }), d),
    H('div', { class: 'pa-tm' }, H('span', {}, D.strict[i] ? H('span', { class: 'pill strict', text: 'strict' }) : null, H('span', { class: 'mono', text: ` ${nd} def.` })),
      H('button', { type: 'button', class: 'linkbtn', title: `open the concept “${D.g.name[g]}”`, onclick: () => paConcept(g) },
        `${same ? 'concept' : D.g.name[g]} · ${np === 1 ? 'only here' : `${nf(np)} papers`} →`)));
}

/* ---- everything else, one click away */
function paXps() {
  const P = D.p, n = P.id.length, terms = P.n.reduce((s, v) => s + v, 0), defs = P.ndef.reduce((s, v) => s + v, 0);
  const small = P.n.filter((v) => v < 10).length, withOnly = PAX.only.filter((v) => v > 0).length;
  $('#pa-xps').append(
    expander({ title: 'Terms against definitions, paper by paper', meta: H('span', {}, H('b', { text: f2(defs / terms) }), ' defs per term'), build: paScatter }),
    expander({ title: 'How many terms a paper defines', meta: H('span', {}, H('b', { text: pc(small / n, 0) }), ' under 10 terms'), build: paDist }),
    expander({ title: 'Terms no other paper defines', meta: H('span', {}, 'in ', H('b', { text: pc(withOnly / n, 0) }), ' of papers'), build: paOnly }));
}
const paLegend = (...items) => H('div', { class: 'pa-legend' }, ...items);
function paSeedCited() { const s = [], c = []; for (let p = 0; p < D.p.id.length; p++) (paSeed(p) ? s : c).push(p); return [s, c]; }

/* the scatter: term mentions (across) against definitions (up), log axes, one dot per paper */
function paScatter(body) {
  const el = H('div', { class: 'chart pa-chart' }), lg = paLegend(), cap = H('p', { class: 'cap' }), ol = H('ol', { class: 'pa-out' });
  body.append(lg, el, cap, ol);
  const P = D.p, n = P.id.length;
  const w = Math.max(300, el.clientWidth || 600), h = Math.round(Math.min(400, Math.max(260, w * 0.56))), lw = 46, rw = 14, top = 12, bot = 40;
  const mx = Math.max(...P.n), my = Math.max(...P.ndef), L = Math.log10;
  const x = lin(-0.06, L(mx * 1.3), lw, w - rw), y = lin(-0.06, L(my * 1.3), h - bot, top);
  // a deterministic spread of +-0.03 decades, so papers with the same counts do not hide each other
  const jit = (p, k) => { const t = Math.sin((p + 1) * (k ? 12.9898 : 78.233) + k) * 43758.5453; return t - Math.floor(t) - 0.5; };
  const X = new Float32Array(n), Y = new Float32Array(n);
  for (let p = 0; p < n; p++) { X[p] = x(L(P.n[p]) + 0.06 * jit(p, 0)); Y[p] = y(L(P.ndef[p]) + 0.06 * jit(p, 1)); }
  const svg = S('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': 'Each paper: its terms across against its definitions up, on log scales; seed and cited papers in two colours' });
  el.append(svg);
  const tk = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
  for (const v of tk.filter((v) => v <= mx * 1.3)) { S('line', { x1: x(L(v)), x2: x(L(v)), y1: top, y2: h - bot, class: 'gl' }, svg); T(svg, x(L(v)), h - bot + 15, String(v), 'tx-m', 'middle'); }
  for (const v of tk.filter((v) => v <= my * 1.3)) { S('line', { x1: lw, x2: w - rw, y1: y(L(v)), y2: y(L(v)), class: 'gl' }, svg); T(svg, lw - 6, y(L(v)) + 4, String(v), 'tx-m', 'end'); }
  T(svg, (lw + w) / 2, h - 4, 'terms in the paper (log scale)', 'tx-m', 'middle');
  const yl = T(svg, 12, (top + h - bot) / 2, 'definitions (log scale)', 'tx-m', 'middle'); yl.setAttribute('transform', `rotate(-90 12 ${(top + h - bot) / 2})`);
  // reference lines: one and three definitions per term
  const ref = (k, dash) => { const x1 = Math.min(mx, my / k); S('line', { x1: x(0), y1: y(L(k)), x2: x(L(x1)), y2: y(L(x1 * k)), class: 'pa-ref', 'stroke-dasharray': dash }, svg); };
  ref(1, null); ref(3, '5 4');
  const order = Array.from({ length: n }, (_, p) => p).sort((a, b) => jit(a, 2) - jit(b, 2));
  const dots = new Array(n), gD = S('g', {}, svg);
  for (const p of order) dots[p] = S('circle', { cx: X[p].toFixed(1), cy: Y[p].toFixed(1), r: 2.7, class: 'pa-dot ' + (paSeed(p) ? 'pa-seed' : 'pa-cited') }, gD);
  // labelled outliers, each picked by a rule; a paper picked by two rules is labelled once, with both reasons
  const by = (f, ok = () => true) => { let b = -1; for (let p = 0; p < n; p++) if (ok(p) && (b < 0 || f(p) > f(b))) b = p; return b; };
  const picked = [];
  const add = (p, why) => { if (p < 0) return; const o = picked.find((q) => q.p === p); if (o) o.why += '; ' + why; else picked.push({ p, why }); };
  add(by((p) => P.n[p]), 'most terms');
  add(by((p) => P.ndef[p]), 'most definitions');
  add(by((p) => P.n[p], paSeed), 'most terms of a seed paper');
  add(by((p) => P.ndef[p], paSeed), 'most definitions of a seed paper');
  add(by((p) => PAX.only[p]), 'most terms no other paper defines');
  add(by((p) => P.ndef[p] / P.n[p], (p) => P.n[p] >= 10), 'most definitions per term, of papers with 10 or more terms');
  const placed = [], gM = S('g', {}, svg);
  picked.forEach((o, k) => {
    const px = X[o.p], py = Y[o.p];
    const cand = [[16, -16], [-16, -16], [18, 8], [-18, 8], [0, -22], [0, 22], [28, -4], [-28, -4], [26, -26], [-26, -26]].map(([dx, dy]) => [px + dx, py + dy]).filter(([a, b]) => a > lw + 8 && a < w - 8 && b > top + 8 && b < h - bot - 8);
    const [mxp, myp] = cand.find(([a, b]) => placed.every(([c, d]) => Math.hypot(a - c, b - d) > 17)) || cand[0] || [px, py];
    placed.push([mxp, myp]);
    const m = S('g', { class: 'pa-mk', tabindex: '0', role: 'button', 'aria-label': `${k + 1}: ${paTitle(o.p)}` }, gM);
    S('line', { x1: px, y1: py, x2: mxp, y2: myp, class: 'pa-lead' }, m); S('circle', { cx: px, cy: py, r: 4.2, class: 'pa-ring' }, m);
    S('circle', { cx: mxp, cy: myp, r: 8 }, m); T(m, mxp, myp + 3.8, String(k + 1), 'pa-mkt', 'middle');
    m.addEventListener('click', (e) => { e.stopPropagation(); paOpen(o.p, true); }); m.addEventListener('keydown', (e) => { if (e.key === 'Enter') paOpen(o.p, true); });
  });
  for (const [k, o] of picked.entries()) ol.append(H('li', {}, H('span', { class: 'pa-num', text: String(k + 1) }), H('span', {},
    H('button', { type: 'button', class: 'linkbtn', text: paShort(paTitle(o.p)), onclick: () => paOpen(o.p, true) }),
    H('span', { class: 'small muted' }, ` ${o.why}: ${nf(P.n[o.p])} terms, ${nf(P.ndef[o.p])} definitions · `, paCorpus(o.p)))));
  const hl = S('circle', { r: 6, class: 'pa-hl', visibility: 'hidden' }, svg), sel = S('circle', { r: 6.5, class: 'pa-sel', visibility: 'hidden' }, svg);
  const tip = H('div', { class: 'pa-tip', hidden: '' }); el.append(tip);
  const near = (e) => {
    const r = svg.getBoundingClientRect(), s = w / r.width, vx = (e.clientX - r.left) * s, vy = (e.clientY - r.top) * s;
    let b = -1, bd = (14 * s) ** 2;
    for (let p = 0; p < n; p++) { if (!paShown(p)) continue; const d = (X[p] - vx) ** 2 + (Y[p] - vy) ** 2; if (d < bd) { bd = d; b = p; } }
    return [b, r, s];
  };
  svg.addEventListener('pointermove', (e) => {
    const [p, r, s] = near(e); if (p < 0) { hl.setAttribute('visibility', 'hidden'); tip.hidden = true; svg.style.cursor = ''; return; }
    hl.setAttribute('cx', X[p]); hl.setAttribute('cy', Y[p]); hl.setAttribute('visibility', 'visible'); svg.style.cursor = 'pointer';
    tip.textContent = ''; tip.append(H('b', { text: paShort(paTitle(p), 110) }), H('span', { text: `${paSeed(p) ? 'seed paper' : 'cited paper'} · ${nf(P.n[p])} terms · ${nf(P.ndef[p])} definitions` }), H('span', { class: 'muted', text: 'click to open it' }));
    tip.hidden = false;
    const cx = X[p] / s, cy = Y[p] / s, tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = Math.max(0, Math.min(r.width - tw, cx - tw / 2)) + 'px'; tip.style.top = (cy - th - 12 < 0 ? cy + 14 : cy - th - 12) + 'px';
  });
  svg.addEventListener('pointerleave', () => { hl.setAttribute('visibility', 'hidden'); tip.hidden = true; });
  svg.addEventListener('click', (e) => { const [p] = near(e); if (p >= 0) paOpen(p, true); });
  Object.assign(PAX, { dots, X, Y, sel });
  const [seeds, cited] = paSeedCited();
  lg.append(H('span', {}, H('i', { class: 'pa-sw pa-seed' }), gl('seed', 'seed papers'), ` ${nf(seeds.length)}`), H('span', {}, H('i', { class: 'pa-sw pa-cited' }), gl('closure', 'cited papers'), ` ${nf(cited.length)}`),
    H('span', {}, H('i', { class: 'pa-ln' }), 'one definition per term'), H('span', {}, H('i', { class: 'pa-ln pa-ln3' }), 'three per term'),
    H('span', {}, H('i', { class: 'pa-sw pa-swnum', text: '1' }), 'papers listed below'));
  cap.textContent = 'One dot per paper, spread slightly so equal counts stay visible. No dot falls below the solid line: every term has at least one definition. The search and filters above dim the papers they leave out; click a dot to open its paper.';
  paMark();
}
function paMark() {
  if (!PAX.dots) return;
  const q = PA.q.length > 0 || PA.corpus.size > 0 || PA.bands.size > 0;
  PAX.dots.forEach((d, p) => { const on = paShown(p); d.classList.toggle('off', !on); d.classList.toggle('hit', q && PA.q.length > 0 && on); });
  const s = PA.sel; if (s >= 0) { PAX.sel.setAttribute('cx', PAX.X[s]); PAX.sel.setAttribute('cy', PAX.Y[s]); PAX.sel.setAttribute('visibility', 'visible'); } else PAX.sel.setAttribute('visibility', 'hidden');
}

/* grouped bars, seed papers beside cited papers, one group per category; hover for the values, click to act */
const paBarPath = (x, y, w, h, r) => { if (h <= 0.5) return ''; r = Math.min(r, w / 2, h); return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`; };
function paGroup(el, cats, series, opt) {
  const w = Math.max(300, el.clientWidth || 600), h = opt.h || 220, lw = 44, rw = 6, top = 14, bot = 40;
  const max = Math.max(...series.flatMap((s) => s.v)) || 1, ns = series.length, gap = 2;
  const y = lin(0, max * 1.08, h - bot, top), bw0 = (w - lw - rw) / cats.length, bw = Math.min(30, (bw0 * 0.7 - gap * (ns - 1)) / ns);
  const svg = S('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': opt.aria });
  el.textContent = ''; el.append(svg);
  for (const v of ticks(0, max, 4)) { S('line', { x1: lw, x2: w - rw, y1: y(v), y2: y(v), class: 'gl' }, svg); T(svg, lw - 6, y(v) + 4, opt.fmt(v), 'tx-m', 'end'); }
  const tip = H('div', { class: 'pa-tip', hidden: '' }); el.append(tip);
  cats.forEach((c, k) => {
    const cx = lw + bw0 * (k + 0.5), x0 = cx - (ns * bw + (ns - 1) * gap) / 2;
    const g = S('g', { class: 'pa-gb' + (opt.onclick ? ' act' : ''), tabindex: '0', role: opt.onclick ? 'button' : 'img', 'aria-label': `${opt.cat(c)}: ${series.map((s) => `${s.name} ${opt.tip(s, k)}`).join(', ')}` }, svg);
    S('rect', { x: lw + bw0 * k + 1, y: top, width: bw0 - 2, height: h - bot - top, rx: 4, class: 'pa-hit' }, g);
    series.forEach((s, j) => { const v = s.v[k]; S('path', { d: paBarPath(x0 + j * (bw + gap), y(v), bw, y(0) - y(v), 3), class: 'pa-b ' + s.cls }, g); });
    T(svg, cx, h - bot + 15, c, 'tx-m', 'middle');
    const on = () => {
      tip.textContent = ''; tip.append(H('b', { text: opt.cat(c) }), ...series.map((s) => H('span', { text: `${s.name}: ${opt.tip(s, k)}` })), opt.onclick ? H('span', { class: 'muted', text: opt.hint }) : null);
      tip.hidden = false; const r = svg.getBoundingClientRect(), sc = r.width / w, tw = tip.offsetWidth;
      tip.style.left = Math.max(0, Math.min(r.width - tw, cx * sc - tw / 2)) + 'px'; tip.style.top = Math.max(0, top * sc) + 'px';
    };
    g.addEventListener('pointerenter', on); g.addEventListener('focus', on);
    g.addEventListener('pointerleave', () => (tip.hidden = true)); g.addEventListener('blur', () => (tip.hidden = true));
    if (opt.onclick) { g.addEventListener('click', () => opt.onclick(k)); g.addEventListener('keydown', (e) => { if (e.key === 'Enter') opt.onclick(k); }); }
  });
  S('line', { x1: lw, x2: w - rw, y1: y(0), y2: y(0), class: 'ax' }, svg);
  if (opt.xlabel) T(svg, (lw + w) / 2, h - 4, opt.xlabel, 'tx-m', 'middle');
}
function paSplit(bands, val) { // per corpus: the count and the share of papers in each band of val(p)
  const [seeds, cited] = paSeedCited();
  return [[seeds, 'seed papers', 'pa-seed'], [cited, 'cited papers', 'pa-cited']].map(([ps, name, cls]) => {
    const c = bands.map(([a, b]) => ps.filter((p) => val(p) >= a && val(p) <= b).length);
    return { name, cls, c, v: c.map((x) => x / ps.length), total: ps.length };
  });
}
const paSeedLegend = (series) => paLegend(...series.map((s) => H('span', {}, H('i', { class: 'pa-sw ' + s.cls }), gl(s.cls === 'pa-seed' ? 'seed' : 'closure', s.name), ` ${nf(s.total)}`)));

/* how many terms a paper defines: the size bands of the filter chips, seed papers beside cited papers */
function paDist(body) {
  const P = D.p, series = paSplit(PA_BANDS, (p) => P.n[p]), ch = H('div', { class: 'chart' });
  body.append(paSeedLegend(series), ch);
  paGroup(ch, PA_BANDS.map(bandLabel), series, { fmt: (v) => pc(v, 0), cat: (c) => `${c} term${c === '1' ? '' : 's'}`, tip: (s, k) => `${pc(s.v[k])} (${nf(s.c[k])})`,
    xlabel: 'terms the paper defines', aria: 'Share of seed papers and of cited papers by how many terms they define', hint: 'click to list these papers',
    onclick: (k) => { PA.bands = new Set([k]); paChange(); paToList(); } });
  const [seeds, cited] = paSeedCited(), all = P.id.map((_, p) => p);
  const row = (name, ps) => H('tr', {}, H('td', { text: name }), H('td', { class: 'n', text: nf(ps.length) }), H('td', { class: 'n', text: nf(paMed(ps.map((p) => P.n[p]))) }),
    H('td', { class: 'n', text: nf(paMed(ps.map((p) => P.ndef[p]))) }), H('td', { class: 'n', text: nf(Math.max(...ps.map((p) => P.n[p]))) }));
  body.append(H('p', { class: 'cap', text: 'Shares of each corpus, so the two compare despite their sizes. Click a pair of bars to list those papers.' }),
    H('div', { class: 'tblwrap pa-tbl' }, H('table', {}, H('thead', {}, H('tr', {}, ...['', 'papers', 'median terms', 'median definitions', 'most terms'].map((t) => H('th', { text: t })))),
      H('tbody', {}, row('seed', seeds), row('cited', cited), row('all', all)))));
}

/* terms no other paper defines: how many each paper has */
function paOnly(body) {
  const P = D.p, OB = [[0, 0], [1, 1], [2, 4], [5, 9], [10, 19], [20, 1e9]], series = paSplit(OB, (p) => PAX.only[p]), ch = H('div', { class: 'chart' });
  const tot = PAX.only.reduce((s, v) => s + v, 0), G = D.g.n.length, b = PAX.bigOnly;
  body.append(H('p', { class: 'cap' }, `${nf(tot)} of the ${nf(G)} `, gl('concept', 'concepts'), ` (${pc(tot / G)}) are defined in one paper only. Most papers add at least one; `,
    H('button', { type: 'button', class: 'linkbtn', text: paShort(paTitle(b), 70), onclick: () => paOpen(b, true) }), ` adds the most, ${nf(PAX.only[b])}.`),
  paSeedLegend(series), ch);
  paGroup(ch, OB.map(bandLabel), series, { fmt: (v) => pc(v, 0), cat: (c) => `${c} only-here term${c === '1' ? '' : 's'}`, tip: (s, k) => `${pc(s.v[k])} (${nf(s.c[k])})`,
    xlabel: 'terms in the paper that no other paper defines', aria: 'Share of seed papers and of cited papers by how many of their terms no other paper defines' });
  body.append(H('div', { class: 'ctl' }, H('button', { type: 'button', class: 'btn', text: 'List the papers, most only-here terms first',
    onclick: () => { PA.sort = 'only'; seg($('#pa-sort'), PA_SORTS, PA.sort, (v) => { PA.sort = v; paChange(); }); paChange(); paToList(); } })));
}
TABS.papers = { init: papers, open: (p) => paOpen(p, true) };
