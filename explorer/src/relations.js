/* relations tab: registers TABS.relations = { init, onShow }. Uses the helpers of core.js.
   What papers say about each other's terms, from the six other extracted fields and the links between concepts they make:
     head       the headline numbers: links, two-sided pairs, pairs confirmed by different papers, links stated by 2+ papers, items
     centre     where papers agree and disagree (on the stage): every linked pair as one diverging bar of what both concepts'
                papers say about it (the first is a kind of the second | the same | the second is a kind of the first; related
                apart), in sets picked by the tiles #re-mir-tiles .tile (the first, "Sides agree", is read by the poster builder),
                and a lens that reads the chosen pair side by side
     expanders  the six fields; the strongest links; roles against roles; how these links are made; probably the same concept;
                meaning against shared links; names of near pairs by cosine band (each built on first open)
   The merged items of every concept (data/terms, about 12 MB) load in the background after the first paint; the numbers that
   need them (the papers on each side) fill in when they arrive. Every number is counted from the data when the tab opens;
   every example is picked by a rule named beside it. */

Object.assign(GL, {
  re_item: ['Merged item', 'One entry of a field for one concept, after the items that say the same thing are merged across its papers.', 'Names (supertypes, subtypes, synonyms, related terms) merge by the term-merging rule, so “Disinformation” and “disinformation” are one item. Sentences (descriptions, examples) merge only when they read the same once case, punctuation and spacing are set aside. The papers of an item are the papers that state it.'],
  re_role: ['Role', 'The field a link comes from: supertype, subtype, synonym or related term.', 'When papers that define “fake news” list “disinformation” as a supertype, the link fake news → disinformation has the role supertype. The same two concepts can be linked in several roles, by different papers or by the same one.'],
  re_stmt: ['Statement', 'One paper naming the other concept of a pair in one role, from one side.', 'The papers that define “fake news” and list “disinformation” as a supertype each make one statement that fake news is a kind of disinformation; the papers that define “disinformation” and list “fake news” as a subtype say the same from the other side. A paper that defines both terms can speak from both sides.'],
  re_two: ['Two-sided pair', 'Two concepts whose papers name each other: X’s papers list Y and Y’s papers list X, in any role.', 'The two sides are counted apart. When the only paper on both sides is one paper that defines both terms, the pair is two-sided but not confirmed by different papers.'],
  re_conf: ['Confirmed by different papers', 'A two-sided pair where some paper on one side is not the paper on the other side.', 'One paper that defines both X and Y and names each in the other’s fields is that paper being consistent. The pair is confirmed by different papers when one paper names Y for X and a different paper names X for Y.'],
  re_cos: ['Embedding cosine', 'How alike two concepts are in meaning, judged from their definitions: about 0.28 for two random mentions, 1 for the same wording.', 'Each concept’s vector is the normalised mean of its mentions’ vectors (Qwen3-Embedding-8B, read as “term: definition”). Each concept lists its 8 nearest other concepts.'],
  re_jaccard: ['Shared-link Jaccard', 'Of all concepts that either of two concepts links to, the share that both link to.', 'Links in any role and either direction count; a link between the two themselves does not. Example: if X links to A, B and C, and Y links to B, C and D, they share 2 of 4 linked concepts: Jaccard 0.50.'],
  re_names: ['Name likeness', 'How alike two concept names are as text: the same words with more added, spelled alike, or different words.', '“Same words, one adds more”: every content word of one name is in the other (small words such as “the”, “of”, “el” and “la” set aside). “Spelled alike”: at least 60% of their three-letter pieces are shared (Dice coefficient), as for “downplay” and “downplaying”. Otherwise “different words”.'],
});

const RE = { draws: [], w: 0, T: null, sel: null, set: 'disp', q: '', page: 0, cell: null };
const RROLES = ['sp', 'sb', 'sy', 'rt'];
const RNAME = { sp: 'supertype', sb: 'subtype', sy: 'synonym', rt: 'related term', de: 'description', ex: 'example' };
const RGLK = { sp: 'supertype', sb: 'subtype', sy: 'synonym', rt: 'relatedterm', de: 'description', ex: 'example' };
const RBINS = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 9], [10, 19], [20, 1e9]];
const rBin = (p) => (p >= 20 ? 6 : p >= 10 ? 5 : p >= 5 ? 4 : p - 1);
const rq = (t) => `“${t}”`, rn = (g) => D.g.name[g];
const rpl = (n, w, ws) => `${nf(n)} ${n === 1 ? w : ws || w + 's'}`;
const rcap = (t) => t[0].toUpperCase() + t.slice(1);
const rshort = (t, n = 110) => (t.length > n ? t.slice(0, n - 1).trimEnd() + '…' : t);
const rk = (v) => (v >= 10000 || (v >= 1000 && v % 1000 === 0) ? Math.round(v / 1000) + 'k' : v >= 1000 ? (v / 1000).toFixed(1) + 'k' : String(v));
const rpv = (a, b) => RE.pair.get(a * RE.G + b);
const Z4 = [0, 0, 0, 0];
function reConcept(g) { show('concepts'); if (TABS.concepts && TABS.concepts.open) TABS.concepts.open(g); }
const rbtn = (text, on, title) => H('button', { type: 'button', class: 'linkbtn', text, title, onclick: (e) => { e.stopPropagation(); on(); } });
const rcb = (g) => rbtn(rn(g), () => reConcept(g), `open the concept “${rn(g)}”`);
const rchip = (r, text) => H('span', { class: 'fchip re-chip', style: `--fc:var(${FIELD[r][3]})` }, H('span', { text: text || RNAME[r] }));
// what a link says in words: the papers that define a list b in role r
function reRead(r, a, b) { const x = rq(rn(a)), y = rq(rn(b)); return r === 'sp' ? `${x} is a kind of ${y}` : r === 'sb' ? `${y} is a kind of ${x}` : r === 'sy' ? `${y} is another name for ${x}` : `${y} is related to ${x}`; }
const rWho = (n, a) => `${rpl(n, 'paper')} that define${n === 1 ? 's' : ''} ${rq(rn(a))}`;
const reSays = (r, a, b, n) => `${rWho(n, a)} list${n === 1 ? 's' : ''} ${rq(rn(b))} as a ${RNAME[r]}`;
function rPager(el, page, pages, go) {
  el.textContent = ''; if (pages <= 1) return;
  const pv = H('button', { class: 'btn', type: 'button', text: '←', 'aria-label': 'previous page', onclick: () => go(page - 1) }), nx = H('button', { class: 'btn', type: 'button', text: '→', 'aria-label': 'next page', onclick: () => go(page + 1) });
  pv.disabled = page === 0; nx.disabled = page >= pages - 1;
  el.append(pv, H('span', { class: 'small muted', text: `${page + 1} / ${nf(pages)}` }), nx);
}
const rMeta = (b, rest) => [H('b', { text: b }), rest];

/* ------------------------------------------------------------------ data: the links, the pairs, the merged items */
function reIndex(rel) {
  const G = D.g.name.length, fi = rel.fields.map((c) => RROLES.indexOf(c));
  const pair = new Map(), nbr = new Array(G).fill(null);
  for (const [a, b, f, n] of rel.edges) {
    const k = a * G + b; let v = pair.get(k); if (!v) pair.set(k, (v = [0, 0, 0, 0])); v[fi[f]] = n;
    (nbr[a] ||= new Set()).add(b); (nbr[b] ||= new Set()).add(a);
  }
  Object.assign(RE, { rel, G, fi, pair, nbr });
  RE.multi = rel.edges.filter((e) => e[3] >= 2).length;
  RE.linkedC = nbr.filter(Boolean).length;
  let roles2 = 0; for (const v of pair.values()) if (v.filter((x) => x).length >= 2) roles2++;
  RE.roles2 = roles2;
}
// what both sides say about the pair {a, b}, in statements: [a is a kind of b, the same, b is a kind of a, related]
function reBal(a, b) { const v = rpv(a, b) || Z4, w = rpv(b, a) || Z4; return [v[0] + w[1], v[2] + w[2], v[1] + w[0], v[3] + w[3]]; }
// the concept more statements call the narrower goes first; no kind-of statement either way: the one with more mentions
function reOrient(a, b) { const s = reBal(a, b); return s[2] > s[0] || (s[2] === s[0] && (D.g.n[b] > D.g.n[a] || (D.g.n[b] === D.g.n[a] && b < a))) ? [b, a] : [a, b]; }
const rRow = ([a, b]) => ({ a, b, s: reBal(a, b) });
const rDir = (s) => s[0] + s[1] + s[2];
function reSets() {
  const G = RE.G, agree = [], msp = [], msb = [], all = [], disp = [], two = [];
  for (const [k, v] of RE.pair) {
    const a = Math.floor(k / G), b = k % G, w = rpv(b, a);
    if (w) {
      if (v[0] && w[1]) agree.push(rRow([a, b]));                  // a's papers: b is broader; b's papers: a is narrower
      if (a < b && v[0] && w[0]) msp.push(rRow(reOrient(a, b)));   // each calls the other its supertype
      if (a < b && v[1] && w[1]) msb.push(rRow(reOrient(a, b)));   // each calls the other its subtype
      if (a < b) two.push([a, b]);
    }
    if (!w || a < b) { const r = rRow(reOrient(a, b)); all.push(r); if (r.s[0] && r.s[2]) disp.push(r); }
  }
  const byN = (x, y) => rDir(y.s) - rDir(x.s) || y.s[3] - x.s[3] || rn(x.a).localeCompare(rn(y.a)) || rn(x.b).localeCompare(rn(y.b));
  for (const L of [agree, msp, msb, all]) L.sort(byN);
  disp.sort((x, y) => Math.min(y.s[0], y.s[2]) - Math.min(x.s[0], x.s[2]) || byN(x, y));
  RE.sets = { agree, disp, msp, msb, all, two };
}
function reLoadTerms(tick) {
  const n = D.meta.fields.terms_shard, G = RE.G, S = Math.ceil(G / n); let done = 0;
  return Promise.all(Array.from({ length: S }, (_, s) => termOf(s * n).then(() => tick(++done, S))))
    .then(() => Promise.all(Array.from({ length: G }, (_, g) => termOf(g))));
}
// the papers that state item b in field r of concept a (from the item's mention indices); a side: every role together
function rePapers(a, b, r) { const L = RE.T && RE.T[a] && RE.T[a][r]; if (!L) return null; for (const it of L) if (it[2] === b) return new Set(it[4].map((i) => D.m.paper[i])); return null; }
function reSide(a, b) { const out = new Set(); for (const r of RROLES) { const P = rePapers(a, b, r); if (P) for (const p of P) out.add(p); } return out; }
function reOverlap(x, y) { if (!x || !y) return 0; let n = 0; for (const p of x) if (y.has(p)) n++; return n; }
const rConf = (A, B) => A.size > 1 || B.size > 1 || (A.size === 1 && B.size === 1 && [...A][0] !== [...B][0]);   // some paper on one side is not the paper on the other
// what the terms add: two-sided pairs confirmed by different papers, sides-agree pairs stated by one paper, the six fields' counts
function reCountT() {
  let conf = 0; for (const [a, b] of RE.sets.two) if (rConf(reSide(a, b), reSide(b, a))) conf++;
  let shared = 0; for (const { a, b } of RE.sets.agree) if (reOverlap(rePapers(a, b, 'sp'), rePapers(b, a, 'sb'))) shared++;
  RE.conf = conf; RE.agreeShared = shared;
  const TS = RE.T, st = {}; let nd = 0, df = 0;
  for (const [c] of FIELDS) st[c] = { items: 0, conc: 0, multi: 0, linked: 0, self: 0, hist: RBINS.map(() => 0), top: null };
  for (let g = 0; g < TS.length; g++) {
    const t = TS[g]; if (!t) continue; nd += t.nd; df += t.df.length;
    for (const [c] of FIELDS) {
      const Lc = t[c]; if (!Lc || !Lc.length) continue; const s = st[c]; s.conc++; s.items += Lc.length;
      for (const it of Lc) {
        const p = it[1]; s.hist[rBin(p)]++; if (p >= 2) s.multi++;
        if (it[2] >= 0) { if (it[2] === g) s.self++; else s.linked++; }
        if (!s.top || p > s.top[1] || (p === s.top[1] && D.g.n[g] > D.g.n[s.top[0]])) s.top = [g, p, it];   // most papers; ties: the concept with more mentions
      }
    }
  }
  Object.assign(RE, { fst: st, nd, df });
}

/* ------------------------------------------------------------------ head: one line and the headline numbers */
function reHead() {
  const line = H('p', {}, 'Besides definitions, papers give a term’s ', gl('supertype', 'supertypes'), ', ', gl('subtype', 'subtypes'), ', ', gl('synonym', 'synonyms'), ', ', gl('relatedterm', 'related terms'), ', ',
    gl('description', 'descriptions'), ' and ', gl('example', 'examples'), '; a name that is itself a concept ', gl('link', 'links'), ' the two concepts.');
  $('#re-head').append(secHead(H('div', { class: 're-ttl' }, H('p', { class: 'eyebrow', text: 'Relations' }), H('h2', { text: 'What papers say about each other’s terms' })), line));
  const E = RE.rel.edges, fc = D.meta.fields.counts, raw = FIELDS.map(([c]) => fc[c][0]), tot = raw.reduce((x, y) => x + y, 0);
  const names = FIELDS.reduce((t, [c], j) => t + (NAMES.includes(c) ? raw[j] : 0), 0);
  const fbar = H('span', { class: 're-fbar', role: 'img', 'aria-label': FIELDS.map(([, l], j) => `${l} ${nf(raw[j])}`).join(', ') },
    FIELDS.map(([, l, , col], j) => H('i', { style: `flex:${raw[j]};background:var(${col})`, title: `${l}: ${nf(raw[j])} items` })));
  RE.confTile = stat({ value: '…', label: 'Confirmed by different papers', sub: 'counting the papers on each side…' });
  const two = RE.sets.two.length;
  $('#re-stats').append(
    stat({ value: E.length, label: 'Links between concepts', sub: `among ${nf(RE.linkedC)} of ${nf(RE.G)} concepts`, lead: true }),
    stat({ value: two, label: 'Two-sided pairs', sub: 'each concept’s papers name the other', onclick: () => reSetPick('all', true) }),
    RE.confTile,
    stat({ value: RE.multi, label: 'Links stated by 2+ papers', sub: `${pc(RE.multi / E.length)}; the rest come from one paper`, onclick: () => reOpenXp('links') }),
    stat({ value: tot, label: 'Items in six fields', sub: H('span', {}, fbar, `${pc(names / tot, 0)} names, ${pc(1 - names / tot, 0)} sentences`), onclick: () => reOpenXp('fields') }));
}
function reHeadT() {
  const two = RE.sets.two.length;
  $('.sv', RE.confTile).textContent = nf(RE.conf);
  $('.ss', RE.confTile).textContent = `of ${nf(two)} two-sided pairs; in the rest one paper speaks for both sides`;
}

/* ------------------------------------------------------------------ the centrepiece: where papers agree and disagree */
const RSETS = [['agree', 'Sides agree'], ['disp', 'Direction disputed'], ['msp', 'Each other’s supertype'], ['msb', 'Each other’s subtype'], ['all', 'All linked pairs']];
const RSETLINE = {
  agree: 'X’s papers: Y is broader; Y’s papers: X is narrower.',
  disp: 'Some statements say X is a kind of Y, others the reverse.',
  msp: 'Each side says it is a kind of the other.',
  msb: 'Each side names the other as a kind of it.',
  all: 'Every pair linked in any role, the most direction statements first.',
};
const RCOL = ['re-xy', 're-sy', 're-yx', 're-rl'];
function reCentre() {
  $('#re-sline').append('Each bar weighs what both concepts’ papers say about a pair, in ', gl('re_stmt', 'statements'), '; pick one to read both sides.');
  const box = $('#re-mir-tiles'); box.textContent = '';
  for (const [id, lab] of RSETS) {
    const tb = H('span', { class: 'tb', text: RSETLINE[id] });
    const b = H('button', { type: 'button', class: 'tile re-set', 'aria-pressed': String(id === RE.set), onclick: () => reSetPick(id) }, H('span', { class: 'tl', text: lab }), H('span', { class: 'tv num', text: nf(RE.sets[id].length) }), tb);
    b.dataset.set = id; box.append(b); if (id === 'agree') RE.agreeTb = tb;
  }
  $('#re-lkh').append(H('span', { class: 're-hn', text: 'Pair (first · second)' }),
    H('span', { class: 're-tk' }, H('span', { class: 're-kxy', text: '◀ first is a kind of second' }), H('span', { class: 're-ksy', text: 'same' }), H('span', { class: 're-kyx', text: 'second is a kind of first ▶' })),
    H('span', { class: 're-hnum', text: 'n' }));
  let tm; $('#re-pq').addEventListener('input', (e) => { clearTimeout(tm); tm = setTimeout(() => { RE.q = e.target.value.trim().toLowerCase(); RE.page = 0; reList(); }, 150); });
  $('#re-cpmore').append(more('How the bars are made',
    H('p', {}, 'A ', gl('re_stmt', 'statement'), ' is one paper naming the other concept in one role. X’s papers listing Y as a supertype and Y’s papers listing X as a subtype both say “X is a kind of Y”; synonyms on either side say they are the same; related terms say nothing about direction, so they are counted apart (in the lens).'),
    H('p', {}, `Each bar shows the shares of the direction statements: “the same” in the middle, the first concept as the narrower to the left, the second to the right; n is how many there are. The first concept is the one more statements call the narrower (in “Sides agree”: the one whose papers name the broader). A paper that defines both terms counts once per side.`),
    RE.confP = H('p', {}, `${nf(RE.sets.two.length)} pairs are `, gl('re_two', 'two-sided'), '.'),
    H('p', {}, 'The bars weigh what papers say; they change nothing in the dataset, and no person checked the links.')));
  const first = RE.sets[RE.set][0]; if (first) RE.sel = { a: first.a, b: first.b };
  reList(); reLens();
}
function reSetPick(id, scroll) {
  RE.set = id; RE.page = 0;
  $$('.re-set', $('#re-mir-tiles')).forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.set === id)));
  const L = reRows(); if (L.length) { RE.sel = { a: L[0].a, b: L[0].b }; reLens(); }
  reList();
  if (scroll) $('#re-stage').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
function reRows() { const q = RE.q, L = RE.sets[RE.set]; return q ? L.filter((r) => rn(r.a).toLowerCase().includes(q) || rn(r.b).toLowerCase().includes(q)) : L; }
function reTrack(s, cls) {
  const t = rDir(s), tr = H('span', { class: 're-tr ' + (cls || '') }, H('i', { class: 're-c0' }));
  if (!t) { tr.append(H('span', { class: 're-only', text: s[3] ? `related only · ${nf(s[3])}` : 'no statement' })); return tr; }
  const h = (v) => (50 * v) / t, m = h(s[1]) / 2;
  [[50 - m - h(s[0]), h(s[0])], [50 - m, h(s[1])], [50 + m, h(s[2])]].forEach(([l, w], j) => { if (w > 0) tr.append(H('i', { class: RCOL[j], style: `left:${l.toFixed(2)}%;width:${w.toFixed(2)}%` })); });
  return tr;
}
const rSay = (a, b, s) => `${rn(a)} and ${rn(b)}: ${nf(s[0])} statements say ${rn(a)} is a kind of ${rn(b)}, ${nf(s[1])} that they are the same, ${nf(s[2])} that ${rn(b)} is a kind of ${rn(a)}, ${nf(s[3])} only relate them`;
function reList() {
  const rows = reRows(), PER = 10, pages = Math.max(1, Math.ceil(rows.length / PER)); RE.page = Math.max(0, Math.min(RE.page, pages - 1));
  const box = $('#re-lk'); box.textContent = '';
  $('#re-setline').textContent = RSETLINE[RE.set];
  for (const r of rows.slice(RE.page * PER, (RE.page + 1) * PER)) {
    const on = RE.sel && RE.sel.a === r.a && RE.sel.b === r.b, n = rDir(r.s);
    box.append(H('button', { type: 'button', class: 're-row', 'aria-pressed': String(!!on), 'aria-label': rSay(r.a, r.b, r.s), title: rSay(r.a, r.b, r.s), onclick: () => rePick(r.a, r.b) },
      H('span', { class: 're-rn' }, H('b', { text: rn(r.a) }), H('span', { class: 're-dot', text: ' · ' }), H('span', { class: 're-2', text: rn(r.b) })),
      reTrack(r.s), H('span', { class: 're-n', text: n ? nf(n) : '–' })));
  }
  if (!rows.length) box.append(H('p', { class: 'small muted re-none', text: RE.set === 'all' ? 'No linked pair matches.' : 'No pair in this set matches; try “All linked pairs”.' }));
  $('#re-lkn').textContent = `${rpl(rows.length, 'pair')}${RE.q ? ' match' : ''}`;
  rPager($('#re-lkpg'), RE.page, pages, (p) => { RE.page = p; reList(); });
}
function rePick(a, b, scroll) {
  RE.sel = { a, b };
  $$('.re-row', $('#re-lk')).forEach((x) => x.setAttribute('aria-pressed', 'false'));
  reList(); reLens();
  if (scroll) $('#re-stage').scrollIntoView({ behavior: 'smooth', block: 'start' });
  else if (innerWidth < 900) { const L = $('#re-lens'), r = L.getBoundingClientRect(); if (r.top > innerHeight - 120) L.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }
}
function reVerdict(a, b, s) {
  const A = rq(rn(a)), B = rq(rn(b));
  if (!rDir(s)) return 'The papers only relate the two, without saying how.';
  if (s[0] && s[2]) return `Disputed: ${nf(s[0])} statements say ${A} is a kind of ${B}, ${nf(s[2])} the reverse${s[1] ? `, ${nf(s[1])} that they are the same` : ''}.`;
  if (!s[0] && !s[2]) return `Every statement that says more calls them the same thing.`;
  const [x, y] = s[0] ? [A, B] : [B, A];
  return s[1] ? `No statement says the reverse of “${x.slice(1, -1)} is a kind of ${y.slice(1, -1)}”; ${nf(s[1])} call them the same.` : `Every direction statement agrees: ${x} is a kind of ${y}.`;
}
function reLens() {
  const box = $('#re-lens'); box.textContent = ''; if (!RE.sel) return;
  const { a, b } = RE.sel, s = reBal(a, b), v = rpv(a, b) || Z4, w = rpv(b, a) || Z4, na = rn(a), nb = rn(b);
  let badge = !rpv(a, b) || !rpv(b, a) ? ['one side only', 'only one concept’s papers name the other'] : ['two-sided', 're_two'];
  let papers = H('p', { class: 'small muted', text: 'counting the papers on each side…' });
  if (RE.T) {
    const A = reSide(a, b), B = reSide(b, a), both = reOverlap(A, B);
    papers = H('p', { class: 'small re-pp' }, `${rpl(A.size, 'paper')} on ${rq(na)} name ${rq(nb)} · ${rpl(B.size, 'paper')} on ${rq(nb)} name ${rq(na)} · ${nf(both)} define both`);
    if (badge[0] === 'two-sided') badge = rConf(A, B) ? ['confirmed by different papers', 're_conf'] : ['one paper speaks for both sides', 're_conf'];
  }
  const mx = Math.max(1, ...s), rd = H('div', { class: 're-rd' });
  const rowsR = [
    [`${na} is a kind of ${nb}`, s[0], `${na}’s papers: supertype ${nf(v[0])} · ${nb}’s papers: subtype ${nf(w[1])}`],
    ['the same thing', s[1], `${na}’s papers: synonym ${nf(v[2])} · ${nb}’s papers: synonym ${nf(w[2])}`],
    [`${nb} is a kind of ${na}`, s[2], `${na}’s papers: subtype ${nf(v[1])} · ${nb}’s papers: supertype ${nf(w[0])}`],
    ['related, without saying how', s[3], `${na}’s papers: related term ${nf(v[3])} · ${nb}’s papers: related term ${nf(w[3])}`]];
  rowsR.forEach(([t, n, sub], j) => rd.append(H('span', { class: 're-rdt' }, H('i', { class: RCOL[j], 'aria-hidden': 'true' }), t),
    H('span', { class: 're-rdb' }, H('i', { class: RCOL[j], style: `width:${((100 * n) / mx).toFixed(1)}%` })), H('b', { class: 'num', text: nf(n) }), H('span', { class: 're-rds', text: sub })));
  box.append(H('div', { class: 're-lt' }, H('p', { class: 'eyebrow', text: 'Both sides of the pair' }), H('span', { class: 'pill plain re-badge' }, badge[1].startsWith('re_') ? gl(badge[1], badge[0]) : H('span', { title: badge[1], text: badge[0] }))),
    H('h4', { class: 're-lh' }, rcb(a), H('span', { class: 're-dot', text: ' · ' }), rcb(b)),
    reTrack(s, 're-big'), H('p', { class: 're-verdict', text: reVerdict(a, b, s) }), rd, papers,
    H('div', { class: 'ctl' }, H('button', { type: 'button', class: 'btn', text: `Open “${trunc(na, 28)}”`, onclick: () => reConcept(a) }),
      H('button', { type: 'button', class: 'btn', text: `Open “${trunc(nb, 28)}”`, onclick: () => reConcept(b) })));
}
function reCentreT() {
  const n = RE.sets.agree.length;
  RE.agreeTb.textContent = `${RSETLINE.agree} In ${nf(RE.agreeShared)} (${pc(RE.agreeShared / n)}) one paper states both sides.`;
  RE.confP.append(` In ${nf(RE.conf)} of them some paper on one side is not the paper on the other: `, gl('re_conf', 'confirmed by different papers'), `. In the other ${nf(RE.sets.two.length - RE.conf)}, one paper that defines both terms names each in the other’s fields.`);
  reLens();
}

/* ------------------------------------------------------------------ the expanders */
const RXP = {};
function reOpenXp(id) { const d = RXP[id]; if (!d) return; d.open = true; d.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
function reXps() {
  const box = $('#re-xps'), fc = D.meta.fields.counts, raw = FIELDS.reduce((t, [c]) => t + fc[c][0], 0);
  RE.simMeta = { same: H('span', { text: '…' }), heat: H('span', { text: '…' }), bands: H('span', { text: '…' }) };
  const X = (id, o) => (RXP[id] = expander(o));
  box.append(
    X('fields', { title: 'What each of the six fields holds', meta: rMeta(nf(raw), ' items'), build: reFieldsBuild }),
    X('links', { title: 'The strongest links', meta: rMeta(nf(RE.multi), ' stated by 2+ papers'), build: reLinksBuild }),
    X('roles', { title: 'Roles against roles: one pair, several roles', meta: rMeta(nf(RE.roles2), ' pairs in 2+ roles'), build: reMatrixBuild }),
    X('how', { title: 'How these links are made', meta: rMeta('5', ' steps, one real link'), build: reHowBuild }),
    X('same', { title: 'Probably the same concept, kept apart by name', meta: RE.simMeta.same, build: reSimTableBuild }),
    X('heat', { title: 'Close in meaning: do they share links?', meta: RE.simMeta.heat, build: reHeatBuild }),
    X('bands', { title: 'Close in meaning: are the names alike?', meta: RE.simMeta.bands, build: reBandsBuild }));
}

/* ---------- 1. the six fields */
function reHist(el, hist, c, ymax) {
  const w = Math.max(220, el.clientWidth || 300), h = 124, lw = 34, top = 16, bot = 28, L = Math.log10, base = 0.6;
  const y = (v) => lin(L(base), L(ymax) + 0.12, h - bot, top)(L(Math.max(v, base))), x = lin(-0.5, hist.length - 0.5, lw, w - 4), bw = Math.min(28, ((w - lw - 4) / hist.length) * 0.7);
  const svg = S('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': `${FIELD[c][1]}: merged items by the number of papers that state them, log scale` });
  for (const v of [1, 10, 100, 1000, 10000, 100000]) if (v <= ymax * 1.3) { S('line', { x1: lw, x2: w - 4, y1: y(v), y2: y(v), class: 'gl' }, svg); T(svg, lw - 5, y(v) + 4, rk(v), 'tx-m', 'end'); }
  hist.forEach((v, i) => {
    if (v > 0) {
      const r = S('rect', { x: x(i) - bw / 2, y: y(v), width: bw, height: y(base) - y(v), rx: 2, style: `fill:var(${FIELD[c][3]})` }, svg);
      S('title', {}, r).append(`${nf(v)} merged ${FIELD[c][2]} items stated by ${bandLabel(RBINS[i])} paper${RBINS[i][1] === 1 ? '' : 's'}`);
      T(svg, x(i), y(v) - 4, rk(v), 'tx-m', 'middle');
    }
    T(svg, x(i), h - bot + 14, bandLabel(RBINS[i]), 'tx-m', 'middle');
  });
  T(svg, (lw + w) / 2, h - 2, 'papers that state the item', 'tx-m', 'middle');
  el.textContent = ''; el.append(svg);
}
function reFieldsBuild(body) {
  const fc = D.meta.fields.counts, N = D.m.n, sm = H('div', { class: 're-sm' }), ex = H('p', { class: 'cap re-ex' }), how = H('div');
  body.append(H('p', { class: 'cap', text: 'One card per field: its items, how often papers repeat them, and how many name a concept.' }), sm, ex, how);
  const P = {};
  for (const [c, label, , col] of FIELDS) {
    const [items, withAny] = fc[c];
    P[c] = { merged: H('p', { class: 'small muted', text: 'merging…' }), chart: H('div', { class: 'chart' }), link: H('div', { class: 'small' }), top: H('p', { class: 'small re-fex' }) };
    sm.append(H('div', { class: 're-f', style: `--fc:var(${col})` },
      H('h4', {}, H('i', { class: 're-sw', 'aria-hidden': 'true' }), gl(RGLK[c], label), H('span', { class: 're-fn num', text: nf(items) })),
      H('p', { class: 'small muted', text: `items, in ${pc(withAny / N)} of term mentions` }), P[c].merged, P[c].chart, P[c].link, P[c].top));
  }
  RE.tp.then(() => {
    const st = RE.fst, ymax = Math.max(...FIELDS.map(([c]) => Math.max(...st[c].hist)));
    for (const [c, , sing, col] of FIELDS) {
      const s = st[c], p = P[c];
      p.merged.textContent = ''; p.merged.append(H('b', { class: 'num', text: nf(s.items) }), ' ', gl('re_item', 'merged'), ` in ${nf(s.conc)} concepts · `, H('b', { text: pc(s.multi / s.items) }), ' by 2+ papers');
      const draw = () => reHist(p.chart, s.hist, c, ymax); draw(); RE.draws.push(draw);
      p.link.textContent = '';
      if (NAMES.includes(c)) {
        const oth = s.items - s.linked - s.self;
        p.link.append(H('div', { class: 're-lbar', role: 'img', 'aria-label': `${pc(s.linked / s.items)} name another concept, ${pc(s.self / s.items)} the concept itself, ${pc(oth / s.items)} no concept` },
          H('i', { style: `flex:${s.linked};background:var(${col})`, title: `${nf(s.linked)} name another concept` }), H('i', { style: `flex:${s.self};background:var(--ink-2)`, title: `${nf(s.self)} name the concept itself` }), H('i', { style: `flex:${oth};background:transparent`, title: `${nf(oth)} name no concept` })),
          H('p', { class: 'muted', style: 'margin-top:3px' }, H('b', { class: 're-ink', text: pc(s.linked / s.items) }), ` name another concept: links (${nf(s.linked)})`));
      } else p.link.append(H('p', { class: 'muted', text: 'Sentences: never a link.' }));
      const [g, n, it] = s.top; p.top.textContent = '';
      if (NAMES.includes(c)) p.top.append('Top: ', rcb(g), ' → ', it[2] >= 0 && it[2] !== g ? rcb(it[2]) : rq(it[0]), ` ×${n}`);
      else p.top.append('Top: ', rq(rshort(it[0], 90)), ' for ', rcb(g), ` ×${n}`);
    }
    const nm = NAMES.map((c) => st[c].multi / st[c].items), de = st.de, exs = st.ex;
    let fk = D.g.name.indexOf('fake news');
    if (fk < 0) { let best = -1; RE.T.forEach((t, g) => { if (!t) return; const v = FIELDS.reduce((a, [c]) => a + t[c].length, 0); if (v > best) { best = v; fk = g; } }); }   // no "fake news": the concept with the most items
    const t = RE.T[fk], one = t.sp.filter((x) => x[1] === 1).length;
    ex.textContent = ''; ex.append(H('b', { text: 'Example: ' }), rcb(fk), ` (${rpl(D.g.n[fk], 'mention')}) has ` + FIELDS.map(([c]) => `${nf(t[c].length)} ${RNAME[c]}`).join(', ') + ` items; ${nf(one)} of its ${nf(t.sp.length)} supertypes come from one paper only.`);
    how.append(more('How these are counted',
      H('p', { text: `Items: as the extraction lists them, over all ${nf(D.m.n)} term mentions. Merged: per concept, the items that say the same thing across its papers become one; names merge by the term-merging rule, sentences when they read the same once case, punctuation and spacing are set aside.` }),
      H('p', { text: `Bars: merged items by how many papers state them, on one log scale for all six cards (each grid line is ten times the one below). Names repeat: ${pc(Math.min(...nm))} to ${pc(Math.max(...nm))} are stated by two or more papers. Sentences almost never do (${nf(de.multi)} of ${nf(de.items)} descriptions, ${nf(exs.multi)} of ${nf(exs.items)} examples), because two papers rarely word a sentence the same way; for comparison, ${nf(RE.df)} of the ${nf(RE.nd)} merged definitions are stated by two or more papers.` }),
      H('p', { text: 'The bar under a name field: merged items that name another concept (they become links), the concept itself (dark), or no concept of the dataset (empty).' })));
  });
}

/* ---------- 2. the strongest links */
const RL = { role: 'all', q: '', minp: 1, page: 0 };
function reRoles(v, skip) { return RROLES.map((r, j) => (v && v[j] && r !== skip ? rchip(r, `${RNAME[r]} ${nf(v[j])}`) : null)).filter(Boolean); }
function reLinksBuild(body) {
  const E = RE.rel.edges, segEl = H('span'), sel = H('select', { 'aria-label': 'minimum number of papers' }), q = H('input', { type: 'search', placeholder: 'search a concept, e.g. fake news', 'aria-label': 'search the links by concept name' });
  RL.el = { count: H('p', { class: 'small muted', role: 'status' }), tbl: H('div', { class: 'tblwrap' }), pager: H('div', { class: 'pager' }) };
  body.append(H('p', { class: 'cap', text: 'Each row is one link and the number of papers that state it; click a row to read both sides of the pair above.' }),
    H('div', { class: 'ctl' }, q, segEl, H('label', { class: 'small' }, 'at least ', sel, ' papers')), RL.el.count, RL.el.tbl, RL.el.pager);
  seg(segEl, [['all', 'all'], ...RROLES.map((r) => [r, RNAME[r]])], RL.role, (v) => { RL.role = v; RL.page = 0; reLinksDraw(); });
  for (const v of [1, 2, 3, 5, 10, 20]) sel.append(H('option', { value: v, text: v, selected: v === RL.minp ? '' : null }));
  sel.addEventListener('change', () => { RL.minp = +sel.value; RL.page = 0; reLinksDraw(); });
  let tm; q.addEventListener('input', (e) => { clearTimeout(tm); tm = setTimeout(() => { RL.q = e.target.value.trim().toLowerCase(); RL.page = 0; reLinksDraw(); }, 150); });
  RE.order = E.map((_, k) => k).sort((x, y) => E[y][3] - E[x][3] || rn(E[x][0]).localeCompare(rn(E[y][0])) || rn(E[x][1]).localeCompare(rn(E[y][1])));
  const n1 = E.filter((e) => e[3] === 1).length, n10 = E.filter((e) => e[3] >= 10).length, byR = RROLES.map((r) => [r, E.filter((e) => RE.rel.fields[e[2]] === r).length]);
  const [a, b, f, n] = E[RE.order[0]], r = RE.rel.fields[f];
  body.append(more('How to read the links',
    H('p', { text: `A link: the papers that define the first concept list the second in one role. ${nf(E.length)} links in all: ${byR.map(([rr, m]) => `${nf(m)} ${RNAME[rr]}`).join(', ')}. ${nf(n1)} (${pc(n1 / E.length)}) rest on a single paper and ${nf(n10)} on ten or more: a link with many papers is a strong one, a link with one paper may be one paper’s view or an extraction slip.` }),
    H('p', { text: `Under each link: what it says in words, the other roles in which the same side lists the pair, and the roles from the other side. The first row: ${reSays(r, a, b, n)}: ${reRead(r, a, b)}.` })));
  reLinksDraw();
}
function reLinksDraw() {
  const E = RE.rel.edges, f = RL.role === 'all' ? -1 : RE.rel.fields.indexOf(RL.role), q = RL.q;
  const rows = RE.order.filter((k) => { const e = E[k]; return e[3] >= RL.minp && (f < 0 || e[2] === f) && (!q || rn(e[0]).toLowerCase().includes(q) || rn(e[1]).toLowerCase().includes(q)); });
  const PER = 15, pages = Math.max(1, Math.ceil(rows.length / PER)); RL.page = Math.max(0, Math.min(RL.page, pages - 1));
  RL.el.count.textContent = `${rpl(rows.length, 'link')} match, the most papers first.`;
  const tb = H('tbody');
  for (const k of rows.slice(RL.page * PER, (RL.page + 1) * PER)) {
    const [a, b, fi, n] = E[k], r = RE.rel.fields[fi], same = reRoles(rpv(a, b), r), back = reRoles(rpv(b, a)), go = () => rePick(...reOrient(a, b), true);
    tb.append(H('tr', { class: 'row', tabindex: '0', title: 'read both sides of this pair', onclick: go, onkeydown: (e) => { if (e.key === 'Enter' && e.target === e.currentTarget) go(); } },
      H('td', { class: 'n', text: nf(n) }),
      H('td', { class: 'wrapc' }, H('div', { class: 're-link' }, rcb(a), H('span', { class: 'muted', 'aria-label': 'lists', text: '→' }), rcb(b), rchip(r)),
        H('div', { class: 'small muted', text: reRead(r, a, b) }),
        same.length || back.length ? H('div', { class: 're-chips' }, same.length ? 'same side also: ' : null, ...same, back.length ? `${same.length ? ' · ' : ''}other side: ` : null, ...back) : null)));
  }
  RL.el.tbl.textContent = '';
  RL.el.tbl.append(H('table', {}, H('thead', {}, H('tr', {}, H('th', { class: 'n', text: 'Papers' }), H('th', { text: 'Link, and what it says' }))), tb));
  rPager(RL.el.pager, RL.page, pages, (p) => { RL.page = p; reLinksDraw(); });
}

/* ---------- 3. roles against roles */
const RXP_T = {
  '00': 'Every paper that links the two lists the second as a supertype only: the first is a kind of the second.',
  '11': 'Every paper that links the two lists the second as a subtype only: the second is a kind of the first.',
  '22': 'Every paper that links the two lists the second as a synonym only: another name for the first.',
  '33': 'Every paper that links the two lists the second as a related term only, without saying how.',
  '10': 'Direction disputed: some say the first is a kind of the second, others the reverse. Both cannot hold for one sense of the words.',
  '20': 'Broader, or the same thing? Some list the second as a supertype, others as a synonym.',
  '21': 'Narrower, or the same thing? Some list the second as a subtype, others as a synonym.',
};
const rRelX = (role) => `Related term links two terms without saying how, so being related in some papers and a ${role} in others is no contradiction.`;
function reMatrixBuild(body) {
  const cells = RROLES.map(() => RROLES.map(() => []));
  for (const [k, v] of RE.pair) {
    const rs = []; for (let j = 0; j < 4; j++) if (v[j]) rs.push(j);
    if (rs.length === 1) cells[rs[0]][rs[0]].push(k);
    else for (let x = 0; x < rs.length; x++) for (let y = x + 1; y < rs.length; y++) cells[rs[y]][rs[x]].push(k);   // lower triangle: row after column
  }
  RE.cells = cells;
  const mx = H('div', { class: 're-mx', role: 'group', 'aria-label': 'concept pairs, role against role' }); RE.mxd = H('div', { class: 're-mxd', 'aria-live': 'polite' }); RE.mx = mx;
  const max = Math.max(...cells.flat().map((l) => l.length)), SHORT = { sp: 'super', sb: 'sub', sy: 'syn', rt: 'related' };
  const head = (r, cls) => H('span', { class: 're-mxc re-mxh ' + (cls || ''), style: `color:var(${FIELD[r][3]})` }, H('span', { class: 're-l', text: RNAME[r] }), H('span', { class: 're-s', text: SHORT[r] }));
  mx.append(H('span', { class: 're-mxc' }));
  for (const r of RROLES) mx.append(head(r));
  RROLES.forEach((r, i) => {
    mx.append(head(r, 're-mxr'));
    RROLES.forEach((r2, j) => {
      if (j > i) { mx.append(H('span', { class: 're-mxc' })); return; }
      const n = cells[i][j].length, a = n ? 6 + 34 * (Math.log(n + 1) / Math.log(max + 1)) : 0;
      const lab = i === j ? `${nf(n)} pairs linked as ${RNAME[r]} only` : `${nf(n)} pairs linked both as ${RNAME[r2]} and as ${RNAME[r]}`;
      const b = H('button', { type: 'button', class: 're-mxc re-mxb' + (i === j ? ' diag' : ''), 'aria-pressed': 'false', 'aria-label': lab, title: lab, style: `background:color-mix(in srgb, var(--accent) ${a.toFixed(0)}%, var(--surface))`, onclick: () => reCell(i, j, 0) },
        H('b', { text: nf(n) }), H('span', { text: i === j ? 'only' : 'both' }));
      b.dataset.ij = `${i}${j}`; mx.append(b);
    });
  });
  const c = cells, nd = c[0][0].length + c[1][1].length + c[2][2].length + c[3][3].length;
  let nconf = 0; for (const v of RE.pair.values()) if ((v[0] ? 1 : 0) + (v[1] ? 1 : 0) + (v[2] ? 1 : 0) >= 2) nconf++;   // two or more of supertype, subtype, synonym
  body.append(H('p', { class: 'cap' }, `A pair: the papers that define X list Y in one or more `, gl('re_role', 'roles'), `. ${pc(nd / RE.pair.size)} of the ${nf(RE.pair.size)} pairs have one role only; ${nf(nconf)} mix supertype, subtype or synonym.`),
    H('div', { class: 're-mxwrap' }, H('div', {}, mx, H('p', { class: 'small muted re-mxkey', text: 'Dashed: one role only. Solid: both roles. Darker: more pairs. Click a square.' })), RE.mxd),
    more('What the squares mean', H('p', { text: `Of the ${nf(RE.pair.size)} pairs, ${nf(nd)} (${pc(nd / RE.pair.size)}) have one role only: the papers agree. ${nf(RE.roles2)} have two roles or more. In ${nf(RE.roles2 - nconf)} of those the extra role is related term (the bottom row), which is no contradiction. The other ${nf(nconf)} are real disagreements, in the three squares between supertype, subtype and synonym: ${nf(c[1][0].length)} pairs are listed both as a supertype and as a subtype, ${nf(c[2][0].length)} both as a supertype and as a synonym, ${nf(c[2][1].length)} both as a subtype and as a synonym. A pair can sit in several squares.` })));
  reCell(1, 0, 0);   // supertype against subtype: the plainest contradiction
  RE.tp.then(() => { if (RE.cell) reCell(RE.cell[0], RE.cell[1], RE.cell[2]); });
}
function reCellSorted(i, j) {
  const sc = (k) => { const v = RE.pair.get(k); return i === j ? v[i] : Math.min(v[i], v[j]) * 1e6 + v[i] + v[j]; };   // both roles strongly stated first
  return RE.cells[i][j].slice().sort((x, y) => sc(y) - sc(x));
}
function reCell(i, j, page) {
  RE.cell = [i, j, page];
  $$('.re-mxb', RE.mx).forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.ij === `${i}${j}`)));
  const ri = RROLES[i], rj = RROLES[j], ks = RE.cells[i][j].length < 30000 ? reCellSorted(i, j) : RE.cells[i][j], G = RE.G;
  const box = RE.mxd; box.textContent = '';
  box.append(H('h4', { class: 're-h4', text: i === j ? `${rcap(RNAME[ri])} only: ${rpl(ks.length, 'pair')}` : `${rcap(RNAME[rj])} and ${RNAME[ri]}: ${rpl(ks.length, 'pair')}` }),
    H('p', { class: 'cap', text: RXP_T[`${i}${j}`] || rRelX(RNAME[rj]) }));
  if (i !== j) {
    const bp = H('p', { class: 'small muted' }); box.append(bp);
    if (RE.T) {
      const nb = ks.filter((k) => reOverlap(rePapers(Math.floor(k / G), k % G, ri), rePapers(Math.floor(k / G), k % G, rj)) > 0).length;
      bp.textContent = `In ${nf(nb)} of these, one paper lists the second concept in both roles itself; in the other ${nf(ks.length - nb)}, the roles come from different papers.`;
    } else bp.textContent = 'Counting whether one paper states both roles…';
  }
  const PER = 6, pages = Math.max(1, Math.ceil(ks.length / PER)); page = Math.max(0, Math.min(page, pages - 1)); RE.cell[2] = page;
  const ol = H('ol', { class: 're-exlist', start: String(page * PER + 1) });
  for (const k of ks.slice(page * PER, (page + 1) * PER)) ol.append(reExRow(k, i !== j ? [rj, ri] : null));
  const pg = H('div', { class: 'pager' });
  box.append(ol, pg); rPager(pg, page, pages, (p) => reCell(i, j, p));
}
function reRoleBar(v) {
  const bar = H('div', { class: 're-rbar', role: 'img', 'aria-label': RROLES.map((r, j) => (v[j] ? `${RNAME[r]} ${v[j]}` : null)).filter(Boolean).join(', ') });
  RROLES.forEach((r, j) => { if (v[j]) bar.append(H('i', { style: `flex:${v[j]};background:var(${FIELD[r][3]})`, title: `${RNAME[r]}: ${rpl(v[j], 'paper')}` })); });
  return bar;
}
function reExRow(k, both) {
  const a = Math.floor(k / RE.G), b = k % RE.G, v = RE.pair.get(k);
  const txt = RROLES.map((r, j) => (v[j] ? `${RNAME[r]} ${nf(v[j])}` : null)).filter(Boolean).join(' · ');
  const nb = both && RE.T ? reOverlap(rePapers(a, b, both[0]), rePapers(a, b, both[1])) : null;
  return H('li', { title: 'read both sides of this pair', onclick: () => rePick(...reOrient(a, b), true) }, H('div', { class: 're-link' }, rcb(a), H('span', { class: 'muted', text: '→' }), rcb(b)), reRoleBar(v),
    H('div', { class: 'small muted', text: `papers on ${rq(rn(a))} list ${rq(rn(b))} as: ${txt}` + (nb != null ? (nb ? ` · ${rpl(nb, 'paper')} state${nb === 1 ? 's' : ''} both` : ' · no paper states both') : '') }));
}

/* ---------- 4. how these links are made: one real link, the most stated supertype link, from a paper to the bars */
const RHOW = [
  ['A paper names other terms', '--f-de', 'Besides definitions, the automatic extraction lists, for each term in each paper, the names and sentences the paper gives in six more fields.'],
  ['Items merge per concept', '--f-ex', 'Per concept, the items its papers state are merged: names by the term-merging rule, sentences when they read the same. Each merged item counts its papers.'],
  ['A name that is a concept links', '--f-sp', 'When a merged name is itself a concept, it becomes a link between two concepts, in the field’s role, weighted by its papers.'],
  ['The other side speaks too', '--f-sb', 'The second concept’s own papers may name the first, in any role: the pair is then two-sided.'],
  ['Both sides weighed', '--f-rt', 'This tab counts each paper naming the other concept in one role as one statement, and reads both sides’ roles as one direction.'],
];
function reHowBuild(body) {
  const ol = H('ol', { class: 're-how' }), S5 = [];
  RHOW.forEach(([t, col, what], i) => { const samp = H('div', { class: 're-samp' }, H('span', { class: 'muted', text: 'loading the real sample…' })); S5.push(samp); ol.append(H('li', { style: `--sc:var(${col})` }, H('h4', {}, H('span', { class: 're-stn', text: String(i + 1).padStart(2, '0') }), t), H('p', { class: 'small', text: what }), samp)); });
  body.append(H('p', { class: 'cap', text: 'The supertype link stated by the most papers, followed from one paper to the bars above, with the real data at each step.' }), ol,
    H('p', { class: 'note' }, H('b', { text: 'Not done anywhere: ' }), 'no item is moved to another field, no count removes or re-types a link, and no person checked the links. The counts on this tab only weigh what the papers say.'));
  RE.tp.then(async () => {
    const E = RE.rel.edges, spf = RE.rel.fields.indexOf('sp');
    let best = null; for (const e of E) if (e[2] === spf && (!best || e[3] > best[3] || (e[3] === best[3] && D.g.n[e[0]] > D.g.n[best[0]]))) best = e;   // most papers; ties: the concept with more mentions
    const [a, b, , n] = best, it = RE.T[a].sp.find((x) => x[2] === b), forms = new Set([it[0], ...it[3].map((x) => x[0])].map((x) => x.toLowerCase()));
    const kv = (k, ...v) => H('div', {}, H('span', { class: 're-sk', text: k }), ' ', ...v);
    // step 1: the first mention (by index) of the concept whose own supertypes hold the item
    let i0 = -1, raw = null;
    for (const i of it[4].slice().sort((x, y) => x - y)) { const f = await fieldsOf(i); if ((f.sp || []).some(([t]) => forms.has(t.toLowerCase()))) { i0 = i; raw = f; break; } }
    S5[0].textContent = '';
    if (i0 >= 0) {
      const others = FIELDS.filter(([c]) => c !== 'sp' && raw[c] && raw[c].length).map(([c]) => `${nf(raw[c].length)} ${RNAME[c]}${raw[c].length > 1 ? 's' : ''}`);
      S5[0].append(kv('paper', rbtn(rshort(D.p.title[D.m.paper[i0]] || D.p.id[D.m.paper[i0]], 80), () => { show('papers'); if (TABS.papers && TABS.papers.open) TABS.papers.open(D.m.paper[i0]); }, 'open the paper')),
        kv('term', rbtn(D.m.label[i0], () => openMention(i0), 'open this mention')), kv('supertypes', raw.sp.map(([t]) => t).join(', ')), others.length ? kv('also', others.join(', ')) : null);
    } else S5[0].append('No single mention holds this item as written.');
    S5[1].textContent = '';
    S5[1].append(kv('item', `${rq(it[0])} in the supertypes of `, rcb(a)), kv('papers', nf(it[1])), it[3].length ? kv('spellings', it[3].map(([t, m]) => `${t} ×${nf(m)}`).join(', ')) : null);
    const s = RE.fst.sp;
    S5[2].textContent = '';
    S5[2].append(H('div', { class: 're-link' }, rcb(a), H('span', { class: 'muted', text: '→' }), rcb(b), rchip('sp'), H('b', { class: 'num', text: `×${nf(n)}` })),
      H('div', { class: 'muted', text: `one of ${nf(E.length)} links; ${pc(s.linked / s.items)} of the ${nf(s.items)} merged supertypes name another concept` }));
    S5[3].textContent = '';
    const back = reRoles(rpv(b, a));
    S5[3].append(H('div', {}, `${rcap(rn(b))}’s papers name `, rq(rn(a)), ' as:'), back.length ? H('div', { class: 're-chips' }, ...back) : H('div', { class: 'muted', text: 'nothing: the pair is one-sided' }));
    const [x, y] = reOrient(a, b), A = reSide(a, b), B = reSide(b, a), sb = reBal(x, y);
    S5[4].textContent = '';
    S5[4].append(kv('papers', `${nf(A.size)} on the side of ${rq(rn(a))}, ${nf(B.size)} on the side of ${rq(rn(b))}, ${nf(reOverlap(A, B))} on both`),
      reTrack(sb), H('div', { class: 'muted', text: `${nf(sb[0])} say ${rn(x)} is a kind of ${rn(y)}, ${nf(sb[1])} the same, ${nf(sb[2])} the reverse; ${nf(sb[3])} related` }),
      H('div', {}, rbtn('Show this pair in the bars ↑', () => rePick(x, y, true))));
  }).catch((e) => { console.error(e); S5[0].textContent = 'The sample could not be loaded: ' + e.message; });
}

/* ---------- 5–7. close in meaning, kept apart by name */
const RSMALL = new Set(['the', 'a', 'an', 'of', 'and', 'or', 'in', 'on', 'for', 'to', 'by', 'with', 'el', 'la', 'los', 'las', 'le', 'les', 'de', 'del', 'der', 'die', 'das', 'y', 'et', 'und']);
const ROPP = new Map([['upward', 'downward'], ['pro', 'anti'], ['high', 'low'], ['left', 'right'], ['positive', 'negative'], ['liberal', 'conservative'], ['explicit', 'implicit'], ['overt', 'covert'],
  ['internal', 'external'], ['intrinsic', 'extrinsic'], ['active', 'passive'], ['strong', 'weak'], ['online', 'offline'], ['majority', 'minority'], ['inclusive', 'exclusive'], ['upper', 'lower'],
  ['increase', 'decrease'], ['true', 'false'], ['hard', 'soft'], ['ingroup', 'outgroup'], ['male', 'female'], ['men', 'women'], ['rich', 'poor'], ['direct', 'indirect'], ['formal', 'informal'],
  ['legal', 'illegal'], ['ethical', 'unethical'], ['before', 'after'], ['top', 'bottom'], ['primary', 'secondary']].flatMap(([x, y]) => [[x, y], [y, x]]));
const RNK = { contain: 'same words, one adds more', alike: 'spelled alike', differ: 'different words' };
const RNC = { contain: 're-c-contain', alike: 're-c-alike', differ: 're-c-differ' };
const reWordCache = new Map();
function reWords(g) {
  let w = reWordCache.get(g); if (w) return w;
  const ws = rn(g).toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').split(/[^\p{L}\p{N}]+/u).filter(Boolean);
  const content = ws.filter((x) => !RSMALL.has(x)), s = ` ${ws.join(' ')} `, tri = new Set();
  for (let i = 0; i + 3 <= s.length; i++) tri.add(s.slice(i, i + 3));
  w = { all: new Set(ws), key: new Set(content.length ? content : ws), tri }; reWordCache.set(g, w); return w;
}
function reNames(a, b) {
  const A = reWords(a), B = reWords(b), sub = (X, Y) => [...X].every((x) => Y.has(x));
  let i = 0; for (const t of A.tri) if (B.tri.has(t)) i++;
  const dice = (2 * i) / (A.tri.size + B.tri.size || 1);
  const da = [...A.all].filter((x) => !B.all.has(x)), db = [...B.all].filter((x) => !A.all.has(x));
  const opp = da.length > 0 && da.length === db.length && da.every((x) => db.includes(ROPP.get(x)));   // the names differ only by opposite words
  return { k: sub(A.key, B.key) || sub(B.key, A.key) ? 'contain' : dice >= 0.6 ? 'alike' : 'differ', dice, opp };
}
function reSimIndex() {
  const G = RE.G, nb = D.g.cnb, ns = D.g.cns, m = new Map();
  for (let g = 0; g < G; g++) for (let k = 0; k < 8; k++) {
    const h = nb[g * 8 + k]; if (h < 0 || h === g) continue;
    const key = Math.min(g, h) * G + Math.max(g, h), c = ns[g * 8 + k], o = m.get(key); if (o === undefined || c > o) m.set(key, c);
  }
  const n = m.size, A = new Int32Array(n), B = new Int32Array(n), C = new Uint16Array(n), I = new Int32Array(n), U = new Int32Array(n).fill(-1);
  let p = 0;
  for (const [key, c] of m) {
    const a = Math.floor(key / G), b = key % G, X = RE.nbr[a], Y = RE.nbr[b]; A[p] = a; B[p] = b; C[p] = c;
    if (X && Y) { let ua = 0, ub = 0, inter = 0; for (const x of X) if (x !== b) { ua++; if (Y.has(x)) inter++; } for (const y of Y) if (y !== a) ub++; const uni = ua + ub - inter; if (uni > 0) { I[p] = inter; U[p] = uni; } }
    p++;
  }
  const near = []; for (let q = 0; q < n; q++) if (C[q] >= 800) near.push(q);
  const NM = new Map(near.map((q) => [q, reNames(A[q], B[q])]));
  near.sort((x, y) => C[y] - C[x] || NM.get(y).dice - NM.get(x).dice);
  RE.sim = { n, A, B, C, I, U, near, NM };
  reHeatBins();
  // the key numbers in the expanders' headers
  const s = RE.sim, h = RE.heat, binOf = (cc) => Math.floor((cc - RHX0) / RHXS);
  let hiA = 0, hiB = 0; for (let xb = binOf(900); xb < RHNX; xb++) { hiA += h.sh[xb]; hiB += h.tot[xb]; }
  let a95 = 0, t95 = 0; for (const q of s.near) if (s.C[q] >= 950) { t95++; if (s.NM.get(q).k !== 'differ') a95++; }
  RE.sameN = s.near.filter((q) => s.C[q] >= 900 && s.NM.get(q).k !== 'differ').length;
  const put = (el, b, rest) => { el.textContent = ''; el.append(...rMeta(b, rest)); };
  put(RE.simMeta.same, nf(RE.sameN), ' pairs, cosine ≥ 0.90');
  put(RE.simMeta.heat, pc(hiA / hiB, 0), ' share a link at ≥ 0.90');
  put(RE.simMeta.bands, pc(a95 / t95, 0), ' alike at ≥ 0.95');
}
const reSimReady = () => { if (!RE.sim) reSimIndex(); };
const rJ = (q) => (RE.sim.U[q] > 0 ? RE.sim.I[q] / RE.sim.U[q] : -1);
const rJtxt = (q, they) => (RE.sim.U[q] > 0 ? `${they ? 'they ' : ''}share ${nf(RE.sim.I[q])} of ${rpl(RE.sim.U[q], 'linked concept')} (Jaccard ${f2(rJ(q))})` : `${they ? 'they are ' : ''}not both linked, so no shared links to compare`);
// the heatmap: x = cosine in bins of 0.025 from 0.30, y = Jaccard (none shared, then tenths)
const RHX0 = 300, RHXS = 25, RHNX = 28, RHNY = 11;
function reHeatBins() {
  const s = RE.sim, cnt = new Uint32Array(RHNX * RHNY), sh = new Uint32Array(RHNX), tot = new Uint32Array(RHNX), sj = new Float64Array(RHNX), cell = new Int32Array(s.n).fill(-1);
  let all = 0;
  for (let q = 0; q < s.n; q++) {
    if (!(s.U[q] > 0)) continue; all++;
    const xb = Math.max(0, Math.min(RHNX - 1, Math.floor((s.C[q] - RHX0) / RHXS))), yb = s.I[q] === 0 ? 0 : 1 + Math.min(9, Math.ceil((s.I[q] * 10) / s.U[q]) - 1);
    cnt[yb * RHNX + xb]++; tot[xb]++; if (s.I[q]) sh[xb]++; sj[xb] += s.I[q] / s.U[q]; cell[q] = yb * RHNX + xb;
  }
  RE.heat = { cnt, sh, tot, sj, cell, all };
}
const rHx = (w, lw, rw) => (c) => lw + ((c - RHX0) / (RHXS * RHNX)) * (w - lw - rw);
function reHeat(el, onCell) {
  const { cnt } = RE.heat, w = Math.max(300, el.clientWidth || 600), h = 250, lw = 50, rw = 8, top = 26, bot = 34, gap = 6;
  const cw = (w - lw - rw) / RHNX, ch = (h - top - bot - gap) / RHNY, X = rHx(w, lw, rw);
  const yTop = (yb) => (yb === 0 ? h - bot - ch : h - bot - ch - gap - yb * ch);
  const svg = S('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': 'Concept pairs by embedding cosine and shared-link Jaccard; darker squares hold more pairs' });
  const max = Math.max(...cnt);
  for (const v of [0, 0.2, 0.4, 0.6, 0.8, 1]) { const yy = h - bot - ch - gap - v * 10 * ch; S('line', { x1: lw, x2: w - rw, y1: yy, y2: yy, class: 'gl' }, svg); T(svg, lw - 6, yy + 4, f2(v), 'tx-m', 'end'); }
  T(svg, lw - 6, h - bot - ch / 2 + 4, 'none', 'tx-m', 'end');
  T(svg, 4, 12, 'shared-link Jaccard ↑', 'tx-m', 'start');
  for (let c = 300; c <= 1000; c += 100) { S('line', { x1: X(c), x2: X(c), y1: top, y2: h - bot, class: 'gl' }, svg); T(svg, X(c), h - bot + 15, f2(c / 1000), 'tx-m', 'middle'); }
  T(svg, (lw + w) / 2, h - 3, 'embedding cosine of the two concepts', 'tx-m', 'middle');
  for (let yb = 0; yb < RHNY; yb++) for (let xb = 0; xb < RHNX; xb++) {
    const n = cnt[yb * RHNX + xb]; if (!n) continue;
    const lo = (RHX0 + xb * RHXS) / 1000, jl = yb === 0 ? 'no shared linked concept' : `Jaccard ${f2((yb - 1) / 10)}–${f2(yb / 10)}`;
    const r = S('rect', { x: lw + xb * cw + 0.5, y: yTop(yb) + 0.5, width: Math.max(1, cw - 1), height: Math.max(1, ch - 1), rx: 1.5, class: 'c' + (RE.hsel === yb * RHNX + xb ? ' sel' : ''),
      style: `fill:var(--accent);fill-opacity:${(0.12 + (0.88 * Math.log(n + 1)) / Math.log(max + 1)).toFixed(3)}` }, svg);
    S('title', {}, r).append(`${nf(n)} pairs: cosine ${f3(lo)}–${f3(lo + 0.025)}, ${jl}`);
    r.addEventListener('click', () => { RE.hsel = yb * RHNX + xb; $$('rect.c', svg).forEach((x) => x.classList.remove('sel')); r.classList.add('sel'); onCell(yb * RHNX + xb); });
  }
  el.textContent = ''; el.append(svg);
}
function reLine(el) {
  const { sh, tot, sj } = RE.heat, w = Math.max(300, el.clientWidth || 600), h = 130, lw = 50, rw = 8, top = 10, bot = 22, X = rHx(w, lw, rw), y = lin(0, 1, h - bot, top);
  const svg = S('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': 'By cosine: the share of pairs sharing a linked concept, and the mean Jaccard' });
  for (const v of [0, 0.5, 1]) { S('line', { x1: lw, x2: w - rw, y1: y(v), y2: y(v), class: 'gl' }, svg); T(svg, lw - 6, y(v) + 4, pc(v, 0), 'tx-m', 'end'); }
  for (let c = 400; c <= 1000; c += 100) T(svg, X(c), h - bot + 15, f2(c / 1000), 'tx-m', 'middle');
  const pts = (f) => { const out = []; for (let xb = 0; xb < RHNX; xb++) if (tot[xb] >= 30) out.push([X(RHX0 + (xb + 0.5) * RHXS), y(f(xb)), xb]); return out; };
  const line = (P, st, lab) => { S('path', Object.assign({ d: 'M' + P.map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join('L'), fill: 'none', 'stroke-linejoin': 'round' }, st), svg); for (const [a, b, xb] of P) S('title', {}, S('circle', { cx: a, cy: b, r: 2.6, style: st.style.replace('stroke', 'fill') }, svg)).append(lab(xb)); };
  const lo = (xb) => f2((RHX0 + xb * RHXS) / 1000) + '–' + f2((RHX0 + (xb + 1) * RHXS) / 1000);
  line(pts((xb) => sh[xb] / tot[xb]), { style: 'stroke:var(--accent)', 'stroke-width': 2.2 }, (xb) => `cosine ${lo(xb)}: ${pc(sh[xb] / tot[xb])} of ${nf(tot[xb])} pairs share a linked concept`);
  line(pts((xb) => sj[xb] / tot[xb]), { style: 'stroke:var(--ink-2)', 'stroke-width': 1.6, 'stroke-dasharray': '5 3' }, (xb) => `cosine ${lo(xb)}: mean Jaccard ${f3(sj[xb] / tot[xb])}`);
  el.textContent = ''; el.append(svg,
    H('div', { class: 're-legend' }, H('span', {}, H('i', { style: 'background:var(--accent)' }), 'pairs sharing a linked concept'), H('span', {}, H('i', { style: 'background:var(--ink-2)' }), 'mean Jaccard (dashed)'), H('span', { class: 'muted', text: 'cosine bins with 30+ pairs' })));
}
function rePairLine(q) {
  const s = RE.sim, a = s.A[q], b = s.B[q];
  return H('li', { title: `open the concept “${rn(a)}”`, onclick: () => reConcept(a) }, H('div', { class: 're-link' }, rcb(a), H('span', { class: 'muted', text: '·' }), rcb(b)),
    H('div', { class: 'small muted', text: `cosine ${f3(s.C[q] / 1000)}; ${rJtxt(q)}` }));
}
function reHeatBuild(body) {
  reSimReady();
  const s = RE.sim, h = RE.heat, heat = H('div', { class: 'chart re-heat' }), ln = H('div', { class: 'chart' }), ex = H('div', { class: 'cap re-ex', 'aria-live': 'polite' });
  const binOf = (c) => Math.floor((c - RHX0) / RHXS);
  let lowA = 0, lowB = 0; for (let xb = binOf(500); xb < binOf(600); xb++) { lowA += h.sh[xb]; lowB += h.tot[xb]; }
  let hiA = 0, hiB = 0; for (let xb = binOf(900); xb < RHNX; xb++) { hiA += h.sh[xb]; hiB += h.tot[xb]; }
  body.append(H('p', { class: 'cap' }, `The ${nf(h.all)} near pairs where both concepts have links: `, gl('re_jaccard', 'shared links'), ' against ', gl('re_cos', 'cosine'), `. Sharing a linked concept rises from ${pc(lowA / lowB, 0)} at 0.50–0.60 to ${pc(hiA / hiB, 0)} at 0.90+; click a square for its pairs.`), heat, ln, ex);
  const onCell = (ci) => {
    const cell = h.cell, xb = ci % RHNX, yb = Math.floor(ci / RHNX), qs = [];
    for (let q = 0; q < s.n; q++) if (cell[q] === ci) qs.push(q);
    qs.sort((x, y) => s.C[y] - s.C[x] || s.I[y] - s.I[x]);
    const lo = (RHX0 + xb * RHXS) / 1000; ex.textContent = '';
    ex.append(H('b', { text: `${rpl(qs.length, 'pair')}, cosine ${f3(lo)}–${f3(lo + 0.025)}, ${yb === 0 ? 'no shared linked concept' : `Jaccard ${f2((yb - 1) / 10)}–${f2(yb / 10)}`}` }), '; the closest:', H('ol', { class: 're-exlist' }, qs.slice(0, 6).map((q) => rePairLine(q))));
  };
  const draw = () => { reHeat(heat, onCell); reLine(ln); }; draw(); RE.draws.push(draw);
  // worked example: "fake news" and its near concept with the highest Jaccard (else the highest-Jaccard pair at cosine 0.9+)
  const fk = D.g.name.indexOf('fake news'); let qx = -1;
  for (let q = 0; q < s.n; q++) { if (!(s.U[q] > 0)) continue; const hit = fk >= 0 ? s.A[q] === fk || s.B[q] === fk : s.C[q] >= 900; if (hit && (qx < 0 || rJ(q) > rJ(qx))) qx = q; }
  if (qx >= 0) ex.append(H('b', { text: 'Example: ' }), rcb(s.A[qx]), ' and ', rcb(s.B[qx]), `: cosine ${f3(s.C[qx] / 1000)}; ${rJtxt(qx, true)}` + (fk >= 0 ? `, the most of any near concept of ${rq(rn(fk))}.` : ', the most of any pair at 0.90 or more.'));
  body.append(more('How this is made', H('p', { text: `Each concept lists its 8 nearest other concepts by embedding cosine: ${nf(s.n)} distinct pairs, always with two different names (two labels that are the same after normalisation would be one concept). Squares: the pairs where both concepts have links, by shared-link Jaccard (up) and cosine (across); the bottom row holds pairs that share no linked concept; darker squares hold more pairs (log scale). The line chart reads the same pairs by cosine. Most pairs share no linked concept at any cosine, because most concepts have few links.` })));
}
function reBandsBuild(body) {
  reSimReady();
  const s = RE.sim, bands = [[800, 850], [850, 900], [900, 950], [950, 1001]], K = ['contain', 'alike', 'differ'];
  const cnt = bands.map(() => ({ contain: 0, alike: 0, differ: 0 })), best = { contain: -1, alike: -1, differ: -1 };
  for (const q of s.near) { const c = s.C[q], nm = s.NM.get(q), bi = bands.findIndex(([lo, hi]) => c >= lo && c < hi); cnt[bi][nm.k]++; if (best[nm.k] < 0 && !nm.opp) best[nm.k] = q; }
  const al = (bi) => (cnt[bi].contain + cnt[bi].alike) / K.reduce((a, k) => a + cnt[bi][k], 0);
  const box = H('div', { class: 're-bands' });
  bands.forEach(([lo, hi], bi) => {
    const t = K.reduce((a, k) => a + cnt[bi][k], 0);
    box.append(H('div', { class: 're-band' }, H('span', { class: 'mono', text: `${f2(lo / 1000)}–${f2(Math.min(hi, 1000) / 1000)}` }),
      H('div', { class: 're-bbar', role: 'img', 'aria-label': K.map((k) => `${RNK[k]} ${cnt[bi][k]}`).join(', ') }, K.map((k) => H('i', { class: RNC[k], style: `flex:${cnt[bi][k]}`, title: `${RNK[k]}: ${nf(cnt[bi][k])} pairs (${pc(cnt[bi][k] / t)})` }))),
      H('span', { class: 'num', text: nf(t) })));
  });
  box.append(H('div', { class: 're-legend' }, K.map((k) => H('span', {}, H('i', { class: RNC[k] }), RNK[k]))));
  body.append(H('p', { class: 'cap' }, `The ${nf(s.near.length)} pairs at cosine 0.80 or more, by `, gl('re_names', 'name likeness'), `: the closer the meaning, the likelier alike names (${pc(al(0), 0)} at 0.80–0.85, ${pc(al(3), 0)} at 0.95+).`), box,
    H('p', { class: 'cap re-ex' }, H('b', { text: 'Closest of each kind: ' }), ...K.flatMap((k, i) => (best[k] < 0 ? [] : [i ? '; ' : '', `${RNK[k]}: `, rcb(s.A[best[k]]), ' and ', rcb(s.B[best[k]]), ` (${f3(s.C[best[k]] / 1000)})`])), '.'),
    more('How the names are compared', H('p', { text: 'Many near pairs look like spellings that term merging keeps apart: a word added or dropped, another word form, an article, a typo. “Same words, one adds more”: every content word of one name is in the other; “spelled alike”: at least 60% of their three-letter pieces are shared; otherwise “different words”. The examples skip pairs whose names differ only by opposite words.' })));
}
const RS = { min: 900, names: 'same', q: '', page: 0 };
function reSimTableBuild(body) {
  reSimReady();
  const s = RE.sim, segEl = H('span'), sel = H('select', { 'aria-label': 'minimum embedding cosine' }), q = H('input', { type: 'search', placeholder: 'search a concept', 'aria-label': 'search the pairs by concept name' });
  RS.el = { count: H('p', { class: 'small muted', role: 'status' }), tbl: H('div', { class: 'tblwrap' }), pager: H('div', { class: 'pager' }) };
  const opp = s.near.find((q2) => s.NM.get(q2).opp);
  body.append(H('p', { class: 'cap' }, 'Concept pairs close in meaning (', gl('re_cos', 'cosine'), ') whose names the term-merging rule keeps apart. ', H('b', { text: 'A view, not a merge: ' }), 'reading both concepts’ definitions decides.'),
    H('div', { class: 'ctl' }, q, segEl, H('label', { class: 'small' }, 'cosine ≥ ', sel)), RS.el.count, RS.el.tbl, RS.el.pager);
  seg(segEl, [['same', 'names alike'], ['differ', 'different words'], ['all', 'all']], RS.names, (v) => { RS.names = v; RS.page = 0; reSimDraw(); });
  for (const v of [800, 850, 900, 950]) sel.append(H('option', { value: v, text: f2(v / 1000), selected: v === RS.min ? '' : null }));
  sel.addEventListener('change', () => { RS.min = +sel.value; RS.page = 0; reSimDraw(); });
  let tm; q.addEventListener('input', (e) => { clearTimeout(tm); tm = setTimeout(() => { RS.q = e.target.value.trim().toLowerCase(); RS.page = 0; reSimDraw(); }, 150); });
  const fk = D.g.name.indexOf('fake news'), qf = fk >= 0 ? s.near.find((q2) => (s.A[q2] === fk || s.B[q2] === fk) && s.NM.get(q2).k === 'contain') : s.near[0];
  const ex = qf != null ? [H('p', {}, 'Example: ', rq(rn(s.A[qf])), ` (${rpl(D.g.n[s.A[qf]], 'mention')}) and `, rq(rn(s.B[qf])), ` (${rpl(D.g.n[s.B[qf]], 'mention')}): cosine ${f3(s.C[qf] / 1000)}, names: ${RNK[s.NM.get(qf).k]}; ${rJtxt(qf, true)}. Term merging joins only labels that are the same after normalisation, so these stay two concepts.`)] : [];
  body.append(more('Why these are not merged', H('p', { text: 'The released dataset keeps each of these pairs as two concepts. A high cosine says the definitions read alike; it does not prove the terms mean the same.' +
    (opp != null ? ` ${rq(rn(s.A[opp]))} and ${rq(rn(s.B[opp]))} have a cosine of ${f3(s.C[opp] / 1000)}, yet their names differ by opposite words (marked in the list).` : '') + ' Shared links and alike names make a pair more likely to be one concept.' }), ...ex));
  reSimDraw();
}
function reSimDraw() {
  const s = RE.sim, q0 = RS.q;
  const rows = s.near.filter((q) => { const nm = s.NM.get(q); return s.C[q] >= RS.min && (RS.names === 'all' || (RS.names === 'same' ? nm.k !== 'differ' : nm.k === 'differ')) && (!q0 || rn(s.A[q]).toLowerCase().includes(q0) || rn(s.B[q]).toLowerCase().includes(q0)); });
  const PER = 15, pages = Math.max(1, Math.ceil(rows.length / PER)); RS.page = Math.max(0, Math.min(RS.page, pages - 1));
  RS.el.count.textContent = `${rpl(rows.length, 'pair')} match, the closest in meaning first.`;
  const tb = H('tbody');
  for (const q of rows.slice(RS.page * PER, (RS.page + 1) * PER)) {
    const a = s.A[q], b = s.B[q], nm = s.NM.get(q), ab = reRoles(rpv(a, b)), ba = reRoles(rpv(b, a)), go = () => reConcept(a);
    tb.append(H('tr', { class: 'row', tabindex: '0', title: `open the concept “${rn(a)}”`, onclick: go, onkeydown: (e) => { if (e.key === 'Enter' && e.target === e.currentTarget) go(); } },
      H('td', { class: 'n mono', text: f3(s.C[q] / 1000) }),
      H('td', { class: 'wrapc' }, H('div', { class: 're-link' }, rcb(a), H('span', { class: 'small muted', text: rpl(D.g.n[a], 'mention') }), H('span', { class: 'muted', text: '·' }), rcb(b), H('span', { class: 'small muted', text: rpl(D.g.n[b], 'mention') })),
        H('div', { class: 'small muted' }, `names: ${RNK[nm.k]}`, nm.opp ? H('b', { class: 're-opp', text: ' · opposite words' }) : null, ` · ${rJtxt(q)}`),
        ab.length || ba.length ? H('div', { class: 're-chips' }, 'papers link them: ', ...(ab.length ? [`${rn(a)} → `, ...ab] : []), ...(ba.length ? [`${ab.length ? ' · ' : ''}${rn(b)} → `, ...ba] : [])) : null)));
  }
  RS.el.tbl.textContent = '';
  RS.el.tbl.append(H('table', {}, H('thead', {}, H('tr', {}, H('th', { class: 'n', text: 'Cosine' }), H('th', { text: 'The two concepts, their names and their links' }))), tb));
  rPager(RS.el.pager, RS.page, pages, (p) => { RS.page = p; reSimDraw(); });
}

/* ------------------------------------------------------------------ the tab */
function reRedraw() { const w = $('#v-relations').clientWidth; if (!w || w === RE.w) return; RE.w = w; for (const f of RE.draws) f(); }
addEventListener('resize', () => { clearTimeout(RE.rt); RE.rt = setTimeout(() => { if (!$('#v-relations').hidden) reRedraw(); }, 200); });
function reInit() {
  relations().then((rel) => {
    reIndex(rel); reSets(); reHead(); reCentre(); reXps();
    RE.w = $('#v-relations').clientWidth;
    // after the first paint: the near pairs (for the expanders' headers), then the merged items of every concept
    const idle = window.requestIdleCallback || ((f) => setTimeout(f, 60));
    idle(() => { try { reSimReady(); } catch (e) { console.error(e); } });
    RE.tp = new Promise((ok, no) => setTimeout(() => reLoadTerms((d, n) => { if (!RE.T) $('.ss', RE.confTile).textContent = `counting the papers on each side… ${d} of ${n} files`; }).then(ok, no), 30))
      .then((TS) => { RE.T = TS; reCountT(); reHeadT(); reCentreT(); });
    RE.tp.catch((e) => { console.error(e); $('.ss', RE.confTile).textContent = 'The merged items could not be loaded: ' + e.message; });
  }).catch((e) => { console.error(e); $('#v-relations').append(H('p', { class: 'note', text: 'Part of this view could not be drawn: ' + e.message })); });
}
TABS.relations = { init: reInit, onShow: reRedraw };
