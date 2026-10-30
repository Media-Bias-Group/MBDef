/* relevance tab: registers TABS.relevance. Uses the helpers of core.js.
   On the stage: the headline numbers, and the centrepiece, a 3D scatter (three.js) of the three members' scores in three
   switchable spaces with the cut and strict planes, beside a worked example (click a dot to explain it). Below, built on
   first open: the 2D pair scatter (each 3D axis view is exactly one of its pairs), where two members disagree, a
   percentile browser, the three members and the ensemble, and the plane check.
   Space (a), the ensemble's own ranks, needs data/ensemble-ref.json: each member's 1,700 train + dev reference scores
   (sorted; nothing else about those records) and every mention's exact ranks. Every mention on the page is released, so
   it reached the cut; what still separates them is the strict cut. The page recomputes every ensemble score from the
   ranks and draws the planes only when every mention lies on or beyond the cut plane and on the side of the strict plane
   its strict flag says. Precision and recall at the two cuts are not in the page's data (they need the labels): the head
   shows EVAL from core.js, the dataset card's recorded numbers. */
GL.memberrank = ['Member rank r', 'Where a member’s score falls among that member’s own scores on the 1,700 labelled train + dev records of MBDef-Gold: the share of them below it, from 0 to 1.',
  'Ties count half: r = (labelled scores below + ½ labelled scores equal) / 1,700. The ensemble adds these ranks, not the raw scores, so the three members share one scale, and a mention’s rank does not depend on the other mentions (its percentile among MBDef’s mentions does). The page holds each member’s 1,700 reference scores, sorted, and nothing else about those records.'];

const RL = { k: 's', p: 0.5, pair: 'c-j', dis: 'ab', space: 'b', color: 'strict', view: '3d', plane: true, strictPlane: true, ex: -1, userSpace: false };
const RM = ['c', 'b', 'j'];
const RNAME = { c: 'classifier', b: 'basal', j: 'judge' };
const RSPACE = { a: 'ensemble ranks', b: 'percentiles', c: 'raw scores' };
const RUNIT = { a: 'rank', b: 'percentile', c: 'score' };
const RW = {};                                   // normalised ensemble weights by member key
const RREF = { ok: false, why: 'not loaded yet' }; // space (a): exact ranks, recomputed scores and the browser's plane check
const R3 = {};                                   // the 3D view, once three.js has loaded
const R2 = {};                                   // the 2D pair view's current mapping, for hovering
const RCUM = {};                                 // per score: strict counts along its order (the percentile strip's band)
const RRED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const f4 = (x) => (+x).toFixed(4);
const rt = (s) => document.createTextNode(s);
const rid = (id) => document.getElementById(id);
function ordn(x) { const r = Math.round(x * 10) / 10; if (r % 1) return r.toFixed(1) + 'th'; const n = Math.round(r), t = n % 100; return n + (t >= 11 && t <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] || 'th'); }
function rgb(s) { s = (s || '').trim(); if (s[0] === '#') { let h = s.slice(1); if (h.length === 3) h = [...h].map((c) => c + c).join(''); return [0, 2, 4].map((j) => parseInt(h.slice(j, j + 2), 16) / 255); } const m = s.match(/[\d.]+/g) || [128, 128, 128]; return m.slice(0, 3).map((v) => +v / 255); }
const RVIR = [[0, [68, 1, 84]], [0.25, [59, 82, 139]], [0.5, [33, 145, 140]], [0.75, [94, 201, 98]], [1, [253, 231, 37]]]; // the Map tab's ramp, so a score has one colour everywhere
function rvir(t) { t = Math.max(0, Math.min(1, t)); for (let k = 1; k < RVIR.length; k++) if (t <= RVIR[k][0]) { const [a, ca] = RVIR[k - 1], [b, cb] = RVIR[k], u = (t - a) / (b - a); return ca.map((v, i) => (v + (cb[i] - v) * u) / 255); } return RVIR[4][1].map((v) => v / 255); }
const RVIRS = Array.from({ length: 101 }, (_, k) => `rgb(${rvir(k / 100).map((v) => Math.round(v * 255)).join(',')})`);
// colour tokens: RTOK from the page (the panels below), R3T from the stage (the 3D view), which is dark in both themes
let RTOK = {}, R3T = {};
function rtok() {
  const of = (el) => { const cs = getComputedStyle(el), v = (n) => cs.getPropertyValue(n).trim(); return { kept: v('--kept'), strict: v('--strict'), accent: v('--accent'), ink: v('--ink'), ink2: v('--ink-2'), muted: v('--muted'), grid: v('--grid'), bg: v('--surface-2'), surface: v('--surface'), wash: v('--accent-wash') }; };
  RTOK = of(document.documentElement);
  const st = rid('rl-stage'), cs = st && getComputedStyle(st), v = (n) => cs.getPropertyValue(n).trim();
  R3T = st ? { kept: v('--kept'), strict: v('--strict'), accent: v('--stage-accent'), ink: v('--stage-ink'), muted: v('--stage-muted'), bg: v('--stage') } : RTOK;
}
function rcolStr(i, tk = RTOK) { return RL.color === 'score' ? RVIRS[Math.round(100 * Math.max(0, Math.min(1, D.score.s[i])))] : D.strict[i] ? tk.strict : tk.kept; }
let RPERM = null; // draw order: strict, then the not-strict on top, so the fewer not-strict dots stay visible
function rperm() { if (!RPERM) { const a = [], b = []; for (let i = 0; i < D.m.n; i++) (D.strict[i] ? a : b).push(i); RPERM = Uint32Array.from(a.concat(b)); } return RPERM; }
let RNS = -1;
function nStrict() { if (RNS < 0) { RNS = 0; for (let i = 0; i < D.m.n; i++) RNS += D.strict[i]; } return RNS; }
const sideOf = (i) => H('span', { class: 'rl-side ' + (D.strict[i] ? 'strict' : 'kept'), text: D.strict[i] ? 'strict' : 'not strict' });
function rv(sp, k, i) { if (sp === 'a') return k === 's' ? RREF.s[i] : RREF.r[k][i]; return sp === 'b' ? D.rank[k][i] : D.score[k][i]; }
function rs(i) { return RREF.ok ? RREF.s[i] : D.score.s[i]; }
function lowerB(a, x) { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < x) lo = m + 1; else hi = m; } return lo; }
function upperB(a, x) { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] <= x) lo = m + 1; else hi = m; } return lo; }
function rseg(el, opts, val, on, off = {}) { const box = seg(el, opts, val, on); $$('button', box).forEach((b, k) => { const why = off[opts[k][0]]; if (why) { b.disabled = true; b.title = why; b.style.opacity = '.45'; b.style.cursor = 'not-allowed'; } }); return box; }
const metaOf = (el, pre, num, post) => { if (!el) return; el.textContent = ''; el.append(pre ? `${pre} ` : '', H('b', { text: num }), post ? ` ${post}` : ''); };
const nm = (k) => (k === 's' ? 'ensemble' : RNAME[k]);

/* ------------------------------------------------------------------ space (a): load and check the exact ranks */
async function rLoadRef() {
  const N = D.m.n, w = D.meta.weights, tot = Math.abs(w.classifier) + Math.abs(w.basal) + Math.abs(w.judge);
  RW.c = w.classifier / tot; RW.b = w.basal / tot; RW.j = w.judge / tot;
  const fail = (why) => { RREF.ok = false; RREF.why = why; console.warn('relevance, space (a) off:', why); };
  let j;
  try { const r = await fetch('data/ensemble-ref.json'); if (!r.ok) throw new Error(`HTTP ${r.status}`); j = await r.json(); }
  catch (e) { return fail(`data/ensemble-ref.json could not be loaded (${e.message})`); }
  if (RM.some((k) => Math.abs(j.weights[RNAME[k]] - RW[k]) > 1e-12)) return fail('its weights differ from the dataset’s');
  if (RM.some((k) => !j.ranks || !j.ranks[k] || j.ranks[k].length !== N || j.ref[k].length !== j.n_ref)) return fail('it does not hold one rank per mention');
  let hash = 'not checked (this browser offers no SHA-256)';
  if (globalThis.crypto && crypto.subtle) {
    const h = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(D.m.mention_id.join('\n')));
    hash = [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, '0')).join('');
    if (hash !== j.ids_sha256) return fail('its mentions are in another order than data/mentions/');
    hash = 'matches';
  }
  const r = {}, ref = {}; for (const k of RM) { r[k] = new Float32Array(N); ref[k] = Float64Array.from(j.ref[k]); }
  const s = new Float64Array(N), same = new Uint8Array(N), cut = D.meta.cut, bf = D.meta.best_f1_cut, n2 = 2 * j.n_ref;
  let misK = 0, misS = 0, maxd = 0, nsame = 0, near = Infinity, nearS = Infinity, mis32 = 0, misS32 = 0, mis4 = 0, misS4 = 0;
  const fr = Math.fround, w32 = RM.map((k) => fr(RW[k])), cut32 = fr(cut), bf32 = fr(bf);
  for (let i = 0; i < N; i++) {
    const a = j.ranks.c[i] / n2, b = j.ranks.b[i] / n2, c = j.ranks.j[i] / n2;
    r.c[i] = a; r.b[i] = b; r.j[i] = c;
    const v = RW.c * a + RW.b * b + RW.j * c, st = !!D.strict[i], v4 = D.m.s[i] / 10000; s[i] = v;
    if (v < cut) misK++; if ((v >= bf) !== st) misS++;  // every mention is released: none may lie behind the cut plane
    maxd = Math.max(maxd, Math.abs(v - v4)); near = Math.min(near, v - cut); nearS = Math.min(nearS, Math.abs(v - bf));
    const v32 = fr(fr(fr(w32[0] * r.c[i]) + fr(w32[1] * r.b[i])) + fr(w32[2] * r.j[i]));  // the positions as drawn, in 32-bit floats
    if (v32 < cut32) mis32++; if ((v32 >= bf32) !== st) misS32++;
    if (v4 < cut) mis4++; if ((v4 >= bf) !== st) misS4++;  // the stored score, rounded to 4 decimals
    let ok = 1; for (const k of RM) { const x = D.m[k][i] / 10000, lo = lowerB(ref[k], x), hi = upperB(ref[k], x); if (2 * lo + (hi - lo) !== j.ranks[k][i]) { ok = 0; break; } }
    same[i] = ok; nsame += ok;
  }
  const rc = j.checks && j.checks.released;
  Object.assign(RREF, { r, s, ref, same, nsame, n: j.n_ref, misK, misS, mis32, misS32, mis4, misS4, maxd, near, nearS, hash, checks: j.checks });
  if (!rc || rc.mentions !== N) return fail('its own check was not run on these mentions');
  if (misK) return fail(`${misK} mentions land behind the cut plane when their score is recomputed`);
  if (misS) return fail(`${misS} mentions land on the wrong side of the strict cut when their score is recomputed`);
  if (maxd > 0.5e-4 + 1e-9) return fail(`a recomputed score differs from the stored one by ${maxd.toExponential(2)}`);
  RREF.ok = true; RREF.why = '';
  console.info(`relevance, plane check: ${N} mentions, ${misK} behind the cut plane (${mis32} with 32-bit positions; ${mis4} by the 4-decimal stored score; the data's own check: ${rc.below_cut}), ${misS} on the wrong side of the strict plane (${misS32}; ${misS4}); largest |s - stored s| ${maxd.toExponential(2)} (data: ${rc.max_abs_diff_vs_4dp_s.toExponential(2)}); closest past the cut ${near.toExponential(2)} (data: ${rc.closest_to_cut.toExponential(2)}), to the strict cut ${nearS.toExponential(2)}; ids ${hash}; ranks recomputable from the 4-decimal scores for ${nsame}`);
}

/* ------------------------------------------------------------------ the head: one line and the headline numbers */
// recall and precision on the labelled train + dev records: EVAL in core.js, as the dataset card records them
function rPR() { return EVAL; }
function rNLab() { return RREF.n || EVAL.n; }
function rHead() {
  const lede = rid('rl-lede'); lede.textContent = '';
  lede.append(rt('Three models score every mention; the '), gl('ensemble', 'ensemble score'), rt(' adds their ranks. A mention is in MBDef once it reaches the '), gl('cut', 'cut'),
    rt(', and '), gl('strict', 'strict'), rt(' once it also reaches the '), gl('bestf1', 'strict cut'), rt('.'));
  rStats();
}
function rStats() {
  const el = rid('rl-stats'), N = D.m.n, ns = nStrict(), pr = rPR(), nl = rNLab(); el.textContent = '';
  const tile = (o, cls) => { const t = stat(o); if (cls) t.classList.add(cls); return t; };
  el.append(tile({ value: f4(D.meta.cut), label: 'Cut', sub: `all ${nf(N)} mentions here reach it`, lead: true }));
  if (pr) el.append(tile({ value: pc(pr.cut.r), label: 'Recall at the cut', sub: `precision ${pc(pr.cut.p)} · on ${nl ? nf(nl) + ' ' : ''}labelled train + dev records` }));
  el.append(tile({ value: f4(D.meta.best_f1_cut), label: 'Strict cut', sub: 'where F1 is highest on those records' }, 'rl-st'));
  if (pr) el.append(tile({ value: pc(pr.strict.r), label: 'Recall at the strict cut', sub: `precision ${pc(pr.strict.p)} · same records` }, 'rl-st'));
  el.append(tile({ value: nf(ns), label: 'Strict mentions', sub: `of ${nf(N)} (${pc(ns / N)}); ${nf(N - ns)} are not` }, 'rl-st'));
  el.append(tile({ value: nf(D.meta.counts.dropped), label: 'Left out', sub: 'corpus mentions under the cut, not in MBDef' }));
}

/* ------------------------------------------------------------------ the tab */
async function relevance() {
  rtok();
  rHead();
  rPanels();
  await rLoadRef();
  if (RREF.ok) RL.space = 'a';
  RL.ex = rPickExample();
  rStats(); rSegs();
  rid('rl3-plane').addEventListener('change', (e) => { RL.plane = e.target.checked; R3.update && R3.update(); rCap3(); });
  rid('rl3-strict').addEventListener('change', (e) => { RL.strictPlane = e.target.checked; R3.update && R3.update(); });
  rExplain(); rCap3(); rMetas();
  onTheme(() => { rtok(); R3.recolor && R3.recolor(); rScatter(); rStrip(); });
  let lastW = 0, tmr = 0;
  new ResizeObserver(() => { const w = rid('v-relevance').clientWidth; if (!w || w === lastW) return; lastW = w; clearTimeout(tmr); tmr = setTimeout(() => { rScatter(); rStrip(); }, 120); }).observe(rid('v-relevance'));
  r3Start();
}
function rSegs() {
  const off = RREF.ok ? {} : { a: 'Not available: ' + RREF.why }, at = (id, fn) => { const el = rid(id); if (el) fn(el); };
  const pairs = [['c-j', 'classifier · judge'], ['c-b', 'classifier · basal'], ['b-j', 'basal · judge'], ['c-s', 'classifier · ensemble']];
  at('rl3-space', (el) => rseg(el, Object.entries(RSPACE), RL.space, (v) => rSetSpace(v, true), off));
  at('rl2-space', (el) => rseg(el, Object.entries(RSPACE), RL.space, (v) => rSetSpace(v, true), off));
  at('rl3-view', (el) => seg(el, [['3d', '3D'], ['c-b', 'classifier · basal'], ['c-j', 'classifier · judge'], ['b-j', 'basal · judge']], RL.view, (v) => rSetView(v)));
  at('rl3-color', (el) => seg(el, [['strict', 'strict / not strict'], ['score', 'ensemble score']], RL.color, (v) => { RL.color = v; R3.recolor && R3.recolor(); rScatter(); }));
  at('rl2-pair', (el) => seg(el, pairs, RL.pair, (v) => rSetPair(v)));
  at('rl-dis-pair', (el) => seg(el, pairs, RL.pair, (v) => rSetPair(v)));
  at('rl-dis', (el) => seg(el, [['ab', 'first high, second low'], ['ba', 'first low, second high']], RL.dis, (v) => { RL.dis = v; rDis(); }));
  at('rl-score', (el) => seg(el, SCORE_KEYS.map((k) => [k, SC[k]]), RL.k, (v) => { RL.k = v; rStrip(); rByPct(); }));
  const pl = rid('rl3-plane'), st = rid('rl3-strict'), canPlane = RREF.ok && RL.space === 'a';
  for (const c of [pl, st]) { c.disabled = !canPlane; c.parentElement.classList.toggle('off', !canPlane); c.parentElement.title = canPlane ? '' : (RREF.ok ? 'The cuts are flat planes only in ensemble ranks' : 'Not available: ' + RREF.why); }
  const to3 = rid('rl2-to3d'); if (to3) to3.disabled = RL.pair === 'c-s';
}
function rSetSpace(sp, user) { if (sp === 'a' && !RREF.ok) return; RL.space = sp; if (user) RL.userSpace = true; rSegs(); R3.setSpace && R3.setSpace(sp); rScatter(); rDis(); rExplain(); rCap3(); rMetas(); }
function rSetPair(p) { RL.pair = p; rSegs(); rScatter(); rDis(); rMetas(); }
function rSetView(v, instant) { RL.view = v; if (v !== '3d') RL.pair = v; rSegs(); if (v !== '3d') { rScatter(); rDis(); rMetas(); } R3.goView && R3.goView(v, instant); rCap3(); }

/* ------------------------------------------------------------------ the worked example beside the 3D view */
function rPickExample() {
  // the mention nearest the strict plane (the decision still open for released mentions) among those whose members disagree
  // (ranks spread 0.5 or more) and whose ranks the page can recompute from the 4-decimal scores it shows; without the ranks,
  // the mention at the strict cut's percentile
  const N = D.m.n, bf = D.meta.best_f1_cut; let best = -1, bd = Infinity;
  if (RREF.ok) for (let i = 0; i < N; i++) { if (!RREF.same[i]) continue; const a = RREF.r.c[i], b = RREF.r.b[i], c = RREF.r.j[i]; if (Math.max(a, b, c) - Math.min(a, b, c) < 0.5) continue; const d = Math.abs(RREF.s[i] - bf); if (d < bd) { bd = d; best = i; } }
  return best < 0 ? D.order.s[N - nStrict()] : best;
}
function rRandomExample() { const N = D.m.n; for (let t = 0; t < 200; t++) { const i = Math.floor(Math.random() * N); if (!RREF.ok || RREF.same[i]) return i; } return Math.floor(Math.random() * N); }
function rExplain() {
  const el = rid('rl3-explain'), sp = RL.space, cut = D.meta.cut, bf = D.meta.best_f1_cut, i = RL.ex; el.textContent = '';
  if (i < 0) return;
  const s = rs(i), box = H('div', { class: 'ex' }), raw = (k) => f4(D.m[k][i] / 10000);
  box.append(H('span', { class: 'eyebrow', text: 'Worked example · click any dot' }), H('div', { class: 't', text: `“${D.m.label[i]}”` }),
    H('div', { class: 'ctl' }, pill(i), H('span', { class: 'small muted', text: `${D.m.ndef[i]} definition${D.m.ndef[i] > 1 ? 's' : ''} · ${D.p.corpus[D.m.paper[i]] === 'seeds' ? 'seed paper' : 'cited paper'}` })));
  const tb = H('table', { class: 'rl-tab' }), mcell = (k) => H('td', {}, gl(RNAME[k], RNAME[k]));
  if (sp === 'a') {
    tb.append(H('tr', {}, H('th', { text: 'model' }), H('th', { text: 'score' }), H('th', {}, gl('memberrank', 'rank r')), H('th', { text: '× weight' })));
    for (const k of RM) tb.append(H('tr', {}, mcell(k), H('td', { text: raw(k) }), H('td', { text: f4(RREF.r[k][i]) }), H('td', { title: `${f3(RW[k])} × ${f4(RREF.r[k][i])}`, text: f4(RW[k] * RREF.r[k][i]) })));
    tb.append(H('tr', { class: 'sum' }, H('td', {}, gl('ensemble', 'ensemble s')), H('td'), H('td'), H('td', { text: f4(s) })));
  } else {
    tb.append(H('tr', {}, ...['model', 'score', 'percentile', ...(RREF.ok ? ['rank r'] : [])].map((x) => H('th', { text: x }))));
    for (const k of RM) tb.append(H('tr', {}, mcell(k), H('td', { text: raw(k) }), H('td', { text: f3(D.rank[k][i]) }), ...(RREF.ok ? [H('td', { text: f4(RREF.r[k][i]) })] : [])));
    tb.append(H('tr', { class: 'sum' }, H('td', {}, gl('ensemble', 'ensemble s')), H('td', { text: f4(s) }), H('td', { text: f3(D.rank.s[i]) }), ...(RREF.ok ? [H('td')] : [])));
  }
  box.append(tb);
  // where s sits between the cut and the strict cut
  const P = (v) => `${(100 * v).toFixed(2)}%`;
  box.append(H('div', { class: 'rl-scale', style: `--c:${P(cut)};--b:${P(bf)}`, role: 'img', 'aria-label': `ensemble score ${f4(s)} on a scale from 0 to 1, with the cut at ${f4(cut)} and the strict cut at ${f4(bf)}` },
    H('span', { class: 'bar' }), H('span', { class: 'tk', style: `left:${P(cut)}`, text: 'cut' }), H('span', { class: 'tk', style: `left:${P(bf)}`, text: 'strict' }), H('span', { class: 'mk', style: `left:${P(s)}` })));
  const st = s >= bf, fd = (d) => (d < 0.0001 ? d.toFixed(6) : f4(d));
  box.append(H('p', { class: 'rl-verdict' }, rt('s = '), H('b', { text: f4(s) }), rt(`: ${fd(s - cut)} past the cut, ${st ? `${fd(s - bf)} past` : `${fd(bf - s)} short of`} the strict cut, so it is `), sideOf(i), rt('.')));
  if (sp === 'a' && RREF.same[i]) {
    const cnt = RM.map((k) => { const x = D.m[k][i] / 10000, lo = lowerB(RREF.ref[k], x), hi = upperB(RREF.ref[k], x); return [lo, hi - lo]; });
    const say = (k, [lo, eq]) => `${RNAME[k]} ${nf(lo)}${eq ? ` + ½·${eq} equal` : ''}`;
    box.append(more('How its ranks are counted', H('p', { text: `A rank is the share of that model’s ${nf(RREF.n)} labelled scores below this score (ties count half): ${RM.map((k, q) => say(k, cnt[q])).join(', ')} of ${nf(RREF.n)}. So r(classifier) = ${cnt[0][1] ? `(${nf(cnt[0][0])} + ½·${cnt[0][1]})` : nf(cnt[0][0])} / ${nf(RREF.n)} = ${f4(RREF.r.c[i])}.` })));
  } else if (sp === 'a') box.append(H('p', { class: 'small muted', text: 'Scores shown rounded to 4 decimals; the ranks come from the unrounded scores.' }));
  else if (RREF.ok) box.append(H('p', { class: 'small muted', text: 'The ensemble adds the ranks r, not these percentiles.' }));
  box.append(H('div', { class: 'btnrow' }, H('button', { class: 'btn', type: 'button', text: 'Open this mention', onclick: () => openMention(i) }),
    H('button', { class: 'btn', type: 'button', text: 'Another example', onclick: () => { RL.ex = rRandomExample(); rExplain(); R3.mark && R3.mark(); } })));
  el.append(box);
}
// one line under the 3D view: how to read the current space, and which 2D view an axis view is
function rCap3() {
  const el = rid('rl3-cap'), v = RL.view, sp = RL.space; el.textContent = '';
  const look = { 'c-b': ['judge', 'classifier · basal'], 'c-j': ['basal', 'classifier · judge'], 'b-j': ['classifier', 'basal · judge'] }[v];
  if (sp === 'a' && RREF.ok) el.append(rt('Each dot is a mention, placed by its three '), gl('memberrank', 'ranks'), rt(`. The ensemble score is their weighted sum, so each cut is a flat plane: every dot lies beyond the cut plane, the strict ones beyond the strict plane too${RL.plane ? '; dots seen through the cut plane are drawn fainter' : ''}.`));
  else if (sp === 'a') el.append(rt(`The cut planes are not drawn: ${RREF.why}.`));
  else if (sp === 'b') el.append(rt('Each axis is a model’s '), gl('percentile', 'percentile'), rt(' among MBDef’s mentions: the dots spread evenly, but the ensemble does not use these numbers, so the strict boundary is curved, not a plane.'));
  else el.append(rt('Raw scores pile up near 0 and 1, so the corners are crowded; the ensemble ranks them first, so here the cuts are bent surfaces, not planes.'));
  if (look) el.append(rt(` You are looking down the ${look[0]} axis: this is the 2D view “${look[1]}” below.`));
}

/* ------------------------------------------------------------------ the panels below the stage, each built on first open */
const RMETA = {};
function rPanels() {
  const box = rid('rl-xps'); box.textContent = '';
  const meta = (key) => (RMETA[key] = H('span'));
  box.append(
    expander({ title: 'Two models against each other', meta: meta('pair'), build: (b) => {
      b.append(H('div', { class: 'rl-pair' }, H('span', { id: 'rl2-pair' }), H('button', { class: 'btn', type: 'button', id: 'rl2-to3d', text: 'see this pair in 3D ↑',
        onclick: () => { if (RL.pair === 'c-s') return; rSetView(RL.pair); rid('rl3-box').scrollIntoView({ behavior: RRED ? 'auto' : 'smooth', block: 'center' }); } })),
      H('div', { class: 'ctl' }, H('span', { id: 'rl2-space' })),
      H('div', { class: 'chart rl-chart2', id: 'rl2-chart' }, H('div', { class: 'rl-tip', id: 'rl2-tip', hidden: '' })), H('p', { class: 'cap', id: 'rl2-cap' }));
      rSegs(); rScatter();
    } }),
    expander({ title: 'Where the models disagree most', meta: meta('dis'), build: (b) => {
      b.append(H('div', { class: 'rl-pair' }, H('span', { id: 'rl-dis-pair' }), H('span', { id: 'rl-dis' })), H('p', { class: 'cap', id: 'rl-dis-what' }), H('div', { class: 'cards rl-dcards', id: 'rl-dis-cards' }));
      rSegs(); rDis();
    } }),
    Object.assign(expander({ title: 'Mentions by percentile', meta: meta('pct'), build: (b) => {
      const q = (k, t) => H('button', { class: 'btn', type: 'button', 'data-rq': k, text: t, onclick: () => rQuick(k) });
      b.append(H('div', { class: 'rl-pair' }, H('p', { class: 'cap', id: 'rl-strip-cap' }), H('span', { id: 'rl-score' })),
        H('div', { class: 'rl-strip', id: 'rl-strip', tabindex: '0', role: 'slider', 'aria-label': 'percentile', 'aria-valuemin': '0', 'aria-valuemax': '100' }),
        H('div', { class: 'ctl' }, H('span', { class: 'mono', id: 'rl-pctv' })),
        H('div', { class: 'ctl' }, q('top', 'Most relevant'), q('bottom', 'Least relevant'), q('strict', 'Around the strict cut'), q('random', 'Random sample'), H('span', { class: 'rl-sp' }), q('down', '← 12 lower'), q('up', '12 higher →')),
        H('p', { class: 'small muted', id: 'rl-what' }), H('div', { class: 'cards', id: 'rl-cards' }));
      rSegs(); rStripWire(); rStrip(); rByPct();
    } }), { className: 'xp rl-wide' }),
    expander({ title: 'The three models and the ensemble', meta: meta('members'), build: rMembers }),
    expander({ title: 'Plane check and rounding', meta: meta('check'), build: rChecks }),
  );
}
function rMetas() {
  const N = D.m.n, ns = nStrict(), [a, b] = RL.pair.split('-'), sp = RL.space;
  metaOf(RMETA.pct, 'strict cut at the', ordn(100 * (1 - ns / N)), 'percentile');
  metaOf(RMETA.members, 'weights', `${f3(RW.c)} · ${f3(RW.b)} · ${f3(RW.j)}`);
  if (RREF.ok) metaOf(RMETA.check, '', String(RREF.misK), `of ${nf(N)} behind the cut`); else metaOf(RMETA.check, '', 'off');
  // the 2D pair: in ensemble ranks, how many the hidden member decides; otherwise how often both say 0.5 or more
  if (b !== 's' && sp === 'a' && RREF.ok) { const t = RM.find((k) => k !== a && k !== b), c = rBands(a, b, t); metaOf(RMETA.pair, '', pc((N - c.n1 - c.n2) / N), `decided by the ${RNAME[t]}`); }
  else if (b !== 's') { let both = 0; for (let i = 0; i < N; i++) both += D.score[a][i] >= 0.5 && D.score[b][i] >= 0.5; metaOf(RMETA.pair, '', pc(both / N), 'both 0.5 or more'); }
  else metaOf(RMETA.pair, '', nf(ns), 'strict above the line');
  metaOf(RMETA.dis, '', nf(rBig(a, b)), 'differ by over half');
}
function rBands(a, b, t) { const N = D.m.n, cut = D.meta.cut, bf = D.meta.best_f1_cut, wa = RW[a], wb = RW[b], wt = RW[t]; let n0 = 0, n1 = 0, n2 = 0; for (let i = 0; i < N; i++) { const v = wa * rv('a', a, i) + wb * rv('a', b, i); if (v + wt < cut) n0++; if (v >= bf) n1++; else if (v + wt < bf) n2++; } return { n0, n1, n2 }; }
function rBig(a, b) { const N = D.m.n, sp = RL.space === 'a' && !RREF.ok ? 'b' : RL.space; let big = 0; for (let i = 0; i < N; i++) if (Math.abs(rv(sp, a, i) - rv(sp, b, i)) > 0.5 + 1e-6) big++; return big; }  // the margin: 32-bit values, steps of 1e-4 at the finest
function rMembers(body) {
  const N = D.m.n;
  body.append(H('p', { class: 'cap' }, rt('Each model’s score becomes a '), gl('memberrank', 'rank r'), rt(` among its own scores on the ${nf(RREF.n || rNLab())} labelled train + dev records; the `), gl('ensemble', 'ensemble score'), rt(' is their weighted sum:')),
    H('div', { class: 'rl-formula', text: `s = ${f3(RW.c)}·r(classifier) + ${f3(RW.b)}·r(basal) + ${f3(RW.j)}·r(judge)` }));
  const cards = H('div', { class: 'rl-members' });
  for (const k of RM) {
    let hi = 0; for (let i = 0; i < N; i++) hi += D.score[k][i] >= 0.5;
    const medR = RREF.ok ? Float32Array.from(RREF.r[k]).sort()[N >> 1] : null;
    cards.append(H('div', { class: 'rl-member' }, H('span', { class: 'nm' }, gl(RNAME[k], RNAME[k])), H('span', { class: 'w', text: f3(RW[k]) }), H('span', { class: 'small muted', text: 'weight in s' }),
      H('span', { class: 'small', text: GL[RNAME[k]][1] }),
      H('span', { class: 'small muted', text: `Scores 0.5 or more on ${pc(hi / N)} of MBDef’s mentions${medR != null ? `; median rank ${f3(medR)}` : ''}.` })));
  }
  body.append(cards);
}
function rChecks(body) {
  const N = D.m.n, ns = nStrict();
  if (!RREF.ok) { body.append(H('p', { class: 'note', text: `The ensemble ranks are off, so there is no plane to check: ${RREF.why}.` })); return; }
  const rc = RREF.checks.released, pr = RREF.checks.page_recompute, n4 = (n) => (n === 1 ? '1 mention' : `${n} mentions`);
  const tile = (v, l, b) => H('div', { class: 'tile' }, H('span', { class: 'tl', text: l }), H('span', { class: 'tv', text: v }), b ? H('span', { class: 'tb', text: b }) : null);
  body.append(H('p', { class: 'cap', text: `Run in your browser on all ${nf(N)} mentions: each ensemble score recomputed from the exact ranks.` }),
    H('div', { class: 'tiles rl-checks' },
      tile(String(RREF.misK), 'behind the cut plane', `${RREF.mis32} with the positions as drawn; the data’s own check: ${rc.below_cut}`),
      tile(String(RREF.misS), 'on the wrong side of the strict plane', `${RREF.misS32} as drawn; ${nf(ns)} strict, ${nf(N - ns)} not`),
      tile(RREF.near.toFixed(6), 'closest past the cut', `largest gap to the stored score ${RREF.maxd.toExponential(1)}`),
      tile(`${RREF.mis4} · ${RREF.misS4}`, 'misplaced at 4 decimals', 'behind the cut · wrong side of strict')),
    H('p', { class: 'small', text: `Rounding: the page’s data rounds raw scores to 4 decimals, which would put ${n4(RREF.mis4)} behind the cut and ${n4(RREF.misS4)} on the wrong side of the strict cut, each within 0.00005 of it; the exact ranks put none. Ranks need the unrounded scores: ${pr ? `${pr.reference_scores_below_0_0001.judge} of the judge’s ${nf(RREF.n)} labelled scores lie below 0.0001 and ${pr.reference_scores_above_0_9999.judge} above 0.9999` : 'many labelled scores crowd near 0 and 1'}. From the 4-decimal scores alone the page can recompute the ranks of ${nf(RREF.nsame)} mentions (${pc(RREF.nsame / N)}); the worked examples are among them.` }),
    H('p', { class: 'small muted', text: `Mention order against the ranks file, by SHA-256: ${RREF.hash}.` }));
}

/* ------------------------------------------------------------------ the 3D view */
const RVS = `attribute vec3 p0; attribute vec3 col; uniform float uT, uSize, uPlane, uCut; uniform vec3 uW, uDir; varying vec3 vC; varying float vA;
void main() {
  vec3 p = mix(p0, position, uT); vC = col; vA = 1.0;
  if (uPlane > 0.5) { float g = dot(uW, uDir); if (abs(g) > 1e-6) { float L = (uCut - dot(uW, p)) / g; if (L > 0.0) { vec3 q = p + L * uDir;
    if (all(greaterThanEqual(q, vec3(-1e-4))) && all(lessThanEqual(q, vec3(1.0001)))) vA = 0.38; } } }
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0); gl_PointSize = uSize; }`;
const RFS = `uniform float uAlpha; varying vec3 vC; varying float vA;
void main() { float r = length(gl_PointCoord - 0.5) * 2.0; float a = uAlpha * vA * (1.0 - smoothstep(0.5, 1.0, r)); if (a < 0.02) discard; gl_FragColor = vec4(vC, a); }`;
const RRING_VS = 'uniform float uSize; void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_PointSize = uSize; }';
const RRING_FS = 'uniform vec3 uCol; void main() { float r = length(gl_PointCoord - 0.5) * 2.0; float a = smoothstep(0.55, 0.68, r) * (1.0 - smoothstep(0.86, 1.0, r)); if (a < 0.05) discard; gl_FragColor = vec4(uCol, a); }';
async function r3Start() {
  const load = rid('rl3-load');
  let THREE, OrbitControls;
  try { THREE = await import('three'); ({ OrbitControls } = await import('three/addons/controls/OrbitControls.js')); }
  catch (e) { load.textContent = 'The 3D view could not load its library from cdn.jsdelivr.net: ' + e.message; return; }
  const canvas = rid('rl3-canvas'), box = rid('rl3-box'), N = D.m.n, P = rperm();
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true }); } catch (e) { load.textContent = 'This view needs WebGL, which this browser refused.'; return; }
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
  const scene = new THREE.Scene(), DIST = 6; let HH = 0.98;  // half the visible height (or width, in a tall box); more room for labels on narrow screens
  const camera = new THREE.OrthographicCamera(-HH, HH, HH, -HH, 0.01, 50); camera.up.set(0, 0, 1);  // judge up; set before the controls read it
  const HOME = new THREE.Vector3(1.0, -0.78, 0.68).normalize();
  const VIEWS = { '3d': HOME, 'c-b': new THREE.Vector3(0, -1e-5, 1).normalize(), 'c-j': new THREE.Vector3(0, -1, 0), 'b-j': new THREE.Vector3(1, 0, 0) };
  camera.position.copy(HOME).multiplyScalar(DIST);
  const controls = new OrbitControls(camera, canvas);
  Object.assign(controls, { enableDamping: true, dampingFactor: 0.12, minZoom: 0.5, maxZoom: 16, zoomSpeed: 1.1 });
  const grp = new THREE.Group(); grp.position.set(-0.5, -0.5, -0.5); scene.add(grp);
  // the dots: 'position' holds the target space, 'p0' the space they come from while they move
  const pos = new Float32Array(N * 3), from = new Float32Array(N * 3), col = new Float32Array(N * 3);
  const geom = new THREE.BufferGeometry(), posA = new THREE.BufferAttribute(pos, 3), fromA = new THREE.BufferAttribute(from, 3), colA = new THREE.BufferAttribute(col, 3);
  geom.setAttribute('position', posA); geom.setAttribute('p0', fromA); geom.setAttribute('col', colA);
  const uni = { uT: { value: 1 }, uSize: { value: 3 }, uAlpha: { value: 0.62 }, uPlane: { value: 0 }, uCut: { value: D.meta.cut }, uW: { value: new THREE.Vector3(RW.c, RW.b, RW.j) }, uDir: { value: new THREE.Vector3() } };
  const pts = new THREE.Points(geom, new THREE.ShaderMaterial({ uniforms: uni, vertexShader: RVS, fragmentShader: RFS, transparent: true, depthWrite: false }));
  pts.frustumCulled = false; grp.add(pts);
  const ring = (size) => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3)); const o = new THREE.Points(g, new THREE.ShaderMaterial({ uniforms: { uSize: { value: size }, uCol: { value: new THREE.Vector3() } }, vertexShader: RRING_VS, fragmentShader: RRING_FS, transparent: true, depthTest: false })); o.frustumCulled = false; o.renderOrder = 5; o.visible = false; grp.add(o); return o; };
  const hov = ring(16), sel = ring(22);
  // the cube, its three axes (the edges through the corner where all three members score 0) and their ticks
  const C8 = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], E = [];
  for (const a of C8) for (const b of C8) if (a.reduce((t, v, k) => t + Math.abs(v - b[k]), 0) === 1 && a.join() < b.join()) E.push([a, b]);
  const segs = (list) => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(list.flat(2), 3)); return g; };
  const TV = [0, 0.25, 0.5, 0.75, 1];
  const cubeM = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.3 }), axisM = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.8 });
  const axisL = new THREE.LineSegments(segs([[[0, 0, 0], [0, 0, 0]]]), axisM);
  grp.add(new THREE.LineSegments(segs(E), cubeM), axisL);
  // each axis is drawn and labelled on whichever of its 4 parallel cube edges lies on the outline facing the reader:
  // the lowest one for an axis that runs across the screen, the leftmost for one that runs up
  const OTH = [[1, 2], [0, 2], [0, 1]], COMBOS = [[0, 0], [1, 0], [0, 1], [1, 1]];
  const pt = (k, v, cb) => { const p = [0, 0, 0]; p[k] = v; p[OTH[k][0]] = cb[0]; p[OTH[k][1]] = cb[1]; return p; };
  let edgeKey = '';
  function setAxes(ch) {
    const key = ch.map((c) => c.join()).join('|'); if (key === edgeKey) return; edgeKey = key;
    const list = [];
    for (let k = 0; k < 3; k++) { list.push([pt(k, 0, ch[k]), pt(k, 1, ch[k])]); for (const v of TV) { const p = pt(k, v, ch[k]); for (const o of OTH[k]) { const q = p.slice(); q[o] = p[o] ? 1.03 : -0.03; list.push([p, q]); } } }
    axisL.geometry.dispose(); axisL.geometry = segs(list);
  }
  // the cut plane and the strict plane: each the polygon where the plane meets the unit cube
  const wv = [RW.c, RW.b, RW.j], dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  function poly(c) {
    const out = [];
    for (const [a, b] of E) { const fa = dot3(wv, a) - c, fb = dot3(wv, b) - c; if (fa === 0) out.push(a); if (fb === 0) out.push(b); if (fa * fb < 0) { const t = fa / (fa - fb); out.push(a.map((v, k) => v + (b[k] - v) * t)); } }
    const uq = out.filter((p, k) => out.findIndex((q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]) < 1e-9) === k);
    const m = [0, 1, 2].map((k) => uq.reduce((t, p) => t + p[k], 0) / uq.length), n = new THREE.Vector3(...wv).normalize();
    const u = new THREE.Vector3(1, 0, 0).cross(n).normalize(), v2 = n.clone().cross(u);
    return uq.sort((p, q) => { const ap = Math.atan2(v2.dot(new THREE.Vector3(p[0] - m[0], p[1] - m[1], p[2] - m[2])), u.dot(new THREE.Vector3(p[0] - m[0], p[1] - m[1], p[2] - m[2]))); const aq = Math.atan2(v2.dot(new THREE.Vector3(q[0] - m[0], q[1] - m[1], q[2] - m[2])), u.dot(new THREE.Vector3(q[0] - m[0], q[1] - m[1], q[2] - m[2]))); return ap - aq; });
  }
  function plane(c, op) {
    const pl = poly(c), tri = []; for (let k = 1; k < pl.length - 1; k++) tri.push(pl[0], pl[k], pl[k + 1]);
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(tri.flat(), 3));
    const mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ transparent: true, opacity: op, side: THREE.DoubleSide, depthWrite: false })); mesh.renderOrder = -1;
    const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.Float32BufferAttribute(pl.flat(), 3));
    const line = new THREE.LineLoop(lg, new THREE.LineBasicMaterial({ transparent: true, opacity: 0.95 })); line.renderOrder = 4;
    grp.add(mesh, line); const hi = pl.reduce((a, b) => (b[2] + 0.3 * b[1] > a[2] + 0.3 * a[1] ? b : a)), mid = [0, 1, 2].map((k) => pl.reduce((t, q) => t + q[k], 0) / pl.length), top = hi.map((v, k) => v + (mid[k] - v) * 0.3);
    return { mesh, line, top };
  }
  const cutP = plane(D.meta.cut, 0.18), strP = plane(D.meta.best_f1_cut, 0.1);
  // HTML labels: axis names, tick values and the planes' names, placed by projecting their 3D anchors every frame
  const labBox = rid('rl3-labels'), LAB = [];
  const lab = (text, p, cls, axis) => { const e = H('span', { class: cls || '', text }); labBox.append(e); const o = { e, p, axis }; LAB.push(o); return o; };
  const axT = [0, 1, 2].map((k) => lab('', [0, 0, 0], 'ax', k));
  for (let k = 0; k < 3; k++) for (const v of TV) lab(String(v), [0, 0, 0], '', k).v = v;
  const cutL = lab(`cut s = ${f4(D.meta.cut)}`, cutP.top, 'pl'), strL = lab(`strict s = ${f4(D.meta.best_f1_cut)}`, strP.top, 'pl st');
  const vt = new THREE.Vector3(); let BW = 1, BH = 1;
  const proj = (p) => { vt.set(p[0] - 0.5, p[1] - 0.5, p[2] - 0.5).project(camera); return [(vt.x + 1) * 0.5 * BW, (1 - vt.y) * 0.5 * BH]; };
  const half = (w, h, d) => Math.abs(d[0]) * w / 2 + Math.abs(d[1]) * h / 2;
  function placeLabels() {
    const C = proj([0.5, 0.5, 0.5]), ch = [], out = [], len = [], zeros = [];
    for (let k = 0; k < 3; k++) {
      const a0 = proj(pt(k, 0, [0, 0])), a1 = proj(pt(k, 1, [0, 0])), dx = a1[0] - a0[0], dy = a1[1] - a0[1], across = Math.abs(dx) >= Math.abs(dy);
      len[k] = Math.hypot(dx, dy);
      let best = COMBOS[0], bv = -Infinity;
      for (const cb of COMBOS) { const m = proj(pt(k, 0.5, cb)), v = across ? m[1] : -m[0]; if (v > bv + 0.5) { bv = v; best = cb; } }
      ch[k] = best; const m = proj(pt(k, 0.5, best)), ox = m[0] - C[0], oy = m[1] - C[1], n = Math.hypot(ox, oy) || 1; out[k] = [ox / n, oy / n];
    }
    setAxes(ch);
    for (const l of LAB) {
      let hide = l.axis != null && len[l.axis] < 0.25 * Math.min(BW, BH);
      if (l === cutL) hide = !cutP.mesh.visible; if (l === strL) hide = !strP.mesh.visible;
      l.e.hidden = hide; if (hide) continue;
      if (!l.w) { l.w = l.e.offsetWidth; l.h = l.e.offsetHeight; }
      let x, y;
      if (l.axis == null) [x, y] = proj(l.p);
      else {
        const k = l.axis, o = out[k], tick = 7 + half(30, 14, o);
        [x, y] = proj(pt(k, l.v == null ? 0.5 : l.v, ch[k]));
        const d = l.v == null ? tick + half(30, 14, o) + 8 + half(l.w, l.h, o) : tick; x += o[0] * d; y += o[1] * d;
        if (l.v == null && (x - l.w / 2 < 4 || x + l.w / 2 > BW - 4 || y - l.h / 2 < 4 || y + l.h / 2 > BH - 4)) {  // no room beside the axis: put its name past the axis's far end
          const e1 = proj(pt(k, 1, ch[k])), e0 = proj(pt(k, 0, ch[k])), ux = (e1[0] - e0[0]) / (len[k] || 1), uy = (e1[1] - e0[1]) / (len[k] || 1), dd = 12 + half(l.w, l.h, [ux, uy]);
          x = e1[0] + ux * dd; y = e1[1] + uy * dd;
        }
        if (l.v == null) { x = Math.max(4 + l.w / 2, Math.min(BW - 4 - l.w / 2, x)); y = Math.max(4 + l.h / 2, Math.min(BH - 4 - l.h / 2, y)); }
        if (l.v === 0) { if (zeros.some(([zx, zy]) => Math.hypot(zx - x, zy - y) < 22)) { l.e.hidden = true; continue; } zeros.push([x, y]); }
      }
      l.e.style.transform = `translate(${x.toFixed(1)}px,${y.toFixed(1)}px) translate(-50%,-50%)`;
    }
  }
  // drawing on demand: a frame when something changes, more frames while the camera glides or the dots move
  let raf = 0, anim = null, morph = null;
  const req = () => { if (!raf) raf = requestAnimationFrame(frame); };
  const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
  function slerp(a, b, t) { const d = Math.max(-1, Math.min(1, a.dot(b))), om = Math.acos(d); if (om < 1e-6) return b.clone(); if (Math.PI - om < 1e-3) return a.clone().lerp(b, t).add(new THREE.Vector3(0, 0, 1e-3)).normalize(); const s = Math.sin(om); return a.clone().multiplyScalar(Math.sin((1 - t) * om) / s).add(b.clone().multiplyScalar(Math.sin(t * om) / s)); }
  function frame(now) {
    raf = 0;
    if (anim) { const u = Math.min(1, (now - anim.t) / anim.dur), e = ease(u); controls.target.lerpVectors(anim.tg, new THREE.Vector3(), e); camera.position.copy(controls.target).addScaledVector(slerp(anim.d0, anim.d1, e), DIST); camera.zoom = anim.z0 + (anim.z1 - anim.z0) * e; camera.updateProjectionMatrix(); if (u >= 1) anim = null; }
    if (morph) { const u = Math.min(1, (now - morph.t) / 700); uni.uT.value = ease(u); if (u >= 1) { morph = null; update(); } }
    controls.update();
    uni.uDir.value.copy(camera.position).sub(controls.target).normalize();
    uni.uSize.value = (BW < 520 ? 2.6 : 3.2) * renderer.getPixelRatio() * Math.min(3, Math.max(1, Math.sqrt(camera.zoom)));
    renderer.render(scene, camera); placeLabels();
    if (anim || morph) req();
  }
  controls.addEventListener('change', req);
  controls.addEventListener('end', () => { if (RL.view !== '3d' && !anim) { const d = camera.position.clone().sub(controls.target).normalize(); if (d.dot(VIEWS[RL.view]) < 0.9995) { RL.view = '3d'; rSegs(); rCap3(); } } });
  function resize() { const w = box.clientWidth, h = box.clientHeight; if (!w || !h) return; BW = w; BH = h; HH = w < 520 ? 1.16 : 0.98; renderer.setSize(w, h, false); const a = w / h; if (a >= 1) Object.assign(camera, { left: -HH * a, right: HH * a, top: HH, bottom: -HH }); else Object.assign(camera, { left: -HH, right: HH, top: HH / a, bottom: -HH / a }); camera.updateProjectionMatrix(); req(); }
  function update() {  // what is shown: planes, the dimming behind the cut plane, axis names
    const showCut = RL.space === 'a' && RREF.ok && RL.plane && !morph;
    cutP.mesh.visible = cutP.line.visible = showCut; strP.mesh.visible = strP.line.visible = showCut && RL.strictPlane;
    uni.uPlane.value = showCut ? 1 : 0;
    axT.forEach((l, k) => { l.e.textContent = `${RNAME[RM[k]]} ${RUNIT[RL.space]}`; l.w = 0; });
    legend(); req();
  }
  function setSpace(sp) {
    if (morph) { const t = uni.uT.value; for (let q = 0; q < pos.length; q++) from[q] = from[q] + (pos[q] - from[q]) * t; } else from.set(pos);
    for (let q = 0; q < N; q++) { const i = P[q]; pos[3 * q] = rv(sp, 'c', i); pos[3 * q + 1] = rv(sp, 'b', i); pos[3 * q + 2] = rv(sp, 'j', i); }
    posA.needsUpdate = fromA.needsUpdate = true;
    if (RRED || !R3.ready) { uni.uT.value = 1; morph = null; } else { uni.uT.value = 0; morph = { t: performance.now() }; }
    mark(); update();
  }
  function recolor() {  // the stage's colours: the same dark ground in both themes
    scene.background = new THREE.Color(R3T.bg); cubeM.color.set(R3T.muted); axisM.color.set(R3T.ink);
    for (const [p, c] of [[cutP, R3T.accent], [strP, R3T.strict]]) { p.mesh.material.color.set(c); p.line.material.color.set(c); }
    hov.material.uniforms.uCol.value.set(...rgb(R3T.muted)); sel.material.uniforms.uCol.value.set(...rgb(R3T.ink));
    const cache = {};
    for (let q = 0; q < N; q++) { const s = rcolStr(P[q], R3T); col.set(cache[s] || (cache[s] = rgb(s)), 3 * q); }
    colA.needsUpdate = true; uni.uAlpha.value = RL.color === 'score' ? 0.72 : 0.62; legend(); req();
  }
  function legend() {
    const el = rid('rl3-legend'), st = nStrict(); el.textContent = '';
    if (RL.color === 'strict') el.append(H('span', {}, H('i', { style: 'background:var(--strict)' }), `strict ${nf(st)}`), H('span', {}, H('i', { style: 'background:var(--kept)' }), `not strict ${nf(D.m.n - st)}`));
    else el.append(H('span', { style: 'display:flex;flex-direction:column;gap:2px' }, H('span', { class: 'ramp', style: `background:linear-gradient(90deg,${RVIR.map(([t, c]) => `rgb(${c.join(',')}) ${t * 100}%`).join(',')})` }, H('b', { style: `left:${100 * D.meta.cut}%` }), H('b', { style: `left:${100 * D.meta.best_f1_cut}%;background:var(--strict)` })),
      H('span', { class: 'ends' }, H('span', { text: '0' }), H('span', { text: 'ensemble score, lines = cuts' }), H('span', { text: '1' }))));
    if (cutP.mesh.visible) el.append(H('span', {}, H('i', { style: 'background:var(--accent);border-radius:2px;width:14px;height:4px;vertical-align:2px' }), `cut plane s = ${f4(D.meta.cut)}`));
    if (strP.mesh.visible) el.append(H('span', {}, H('i', { style: 'background:var(--strict);border-radius:2px;width:14px;height:4px;vertical-align:2px' }), `strict plane s = ${f4(D.meta.best_f1_cut)}`));
  }
  function goView(v, instant) {
    const d0 = camera.position.clone().sub(controls.target).normalize(), z1 = v === '3d' ? 1 : BW < 520 ? 1.3 : BW < BH * 1.05 ? 1.08 : 1.32;
    if (RRED || instant) { controls.target.set(0, 0, 0); camera.position.copy(VIEWS[v]).multiplyScalar(DIST); camera.zoom = z1; camera.updateProjectionMatrix(); controls.update(); req(); return; }
    anim = { d0, d1: VIEWS[v].clone(), z0: camera.zoom, z1, tg: controls.target.clone(), t: performance.now(), dur: 800 }; req();
  }
  // hover and click: project every dot to the screen and take the nearest (at most one search per 40 ms)
  const PV = new THREE.Matrix4();
  function pickAt(cx, cy, rad) {
    if (anim || morph) return -1;
    const r = canvas.getBoundingClientRect(), px = cx - r.left, py = cy - r.top, w = r.width, h = r.height;
    camera.updateMatrixWorld(); grp.updateMatrixWorld(); PV.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).multiply(grp.matrixWorld);
    const e = PV.elements; let best = -1, bd = rad * rad, bz = Infinity;
    for (let q = 0; q < N; q++) {
      const x = pos[3 * q], y = pos[3 * q + 1], z = pos[3 * q + 2];
      const sx = ((e[0] * x + e[4] * y + e[8] * z + e[12]) + 1) * 0.5 * w - px, sy = (1 - (e[1] * x + e[5] * y + e[9] * z + e[13])) * 0.5 * h - py, d = sx * sx + sy * sy;
      if (d < bd || (d === bd && e[2] * x + e[6] * y + e[10] * z + e[14] < bz)) { bd = d; best = q; bz = e[2] * x + e[6] * y + e[10] * z + e[14]; }
    }
    return best;
  }
  const qOf = new Uint32Array(N); for (let q = 0; q < N; q++) qOf[P[q]] = q;
  function putRing(o, q) { if (q < 0) { o.visible = false; return; } o.geometry.attributes.position.array.set(pos.subarray(3 * q, 3 * q + 3)); o.geometry.attributes.position.needsUpdate = true; o.visible = true; }
  function mark() { putRing(sel, RL.ex >= 0 ? qOf[RL.ex] : -1); req(); }
  const tip = rid('rl3-tip'); let pend = null, prq = 0, last = 0, down = null;
  function doPick() {
    prq = 0; if (!pend) return; const now = performance.now(); if (now - last < 40) { prq = requestAnimationFrame(doPick); return; } last = now;
    const e = pend; pend = null; const q = pickAt(e.clientX, e.clientY, e.pointerType === 'touch' ? 16 : 9);
    putRing(hov, q); req();
    if (q < 0) { tip.hidden = true; canvas.style.cursor = ''; return; }
    canvas.style.cursor = 'pointer'; rTip(tip, P[q], 'click to explain it'); const r = box.getBoundingClientRect(); rPlaceTip(tip, e.clientX - r.left, e.clientY - r.top, r.width, r.height);
  }
  canvas.addEventListener('pointermove', (e) => { if (e.buttons) { tip.hidden = true; hov.visible = false; return; } pend = e; if (!prq) prq = requestAnimationFrame(doPick); });
  canvas.addEventListener('pointerleave', () => { pend = null; tip.hidden = true; hov.visible = false; req(); });
  canvas.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
  canvas.addEventListener('pointerup', (e) => {  // a click (not a drag) explains the nearest dot beside the view
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return; down = null;
    const q = pickAt(e.clientX, e.clientY, e.pointerType === 'touch' ? 16 : 10); if (q < 0) return;
    RL.ex = P[q]; mark(); rExplain(); tip.hidden = true;
    const ex = rid('rl3-explain'), r = ex.getBoundingClientRect(); if (r.top > innerHeight - 80) ex.scrollIntoView({ behavior: RRED ? 'auto' : 'smooth', block: 'nearest' });
  });
  new ResizeObserver(resize).observe(box);
  function screenOf(i) { const p = pos.subarray(3 * qOf[i], 3 * qOf[i] + 3); controls.update(); camera.updateMatrixWorld(); return proj(Array.from(p)); }  // for checks from the console
  Object.assign(R3, { setSpace, recolor, update, goView, resize, mark, screenOf });
  setSpace(RL.space); R3.ready = true; recolor(); resize(); update(); load.hidden = true;
}
function rTip(el, i, hint) {
  el.textContent = ''; const sp = RL.space;
  el.append(H('b', { text: D.m.label[i] }), pill(i), H('span', { class: 'mono', text: RM.map((k) => `${RNAME[k]} ${f3(D.m[k][i] / 10000)}`).join(' · ') }));
  if (sp !== 'c') el.append(H('span', { class: 'mono', text: `${sp === 'a' ? 'ranks r' : 'percentiles'} ${RM.map((k) => f3(rv(sp, k, i))).join(' · ')}` }));
  el.append(H('span', { class: 'mono', text: `ensemble s ${f4(rs(i))}` }), H('span', { class: 'small muted', text: hint }));
}
function rPlaceTip(el, x, y, w, h) { el.hidden = false; const tw = el.offsetWidth, th = el.offsetHeight; el.style.left = Math.max(4, x + 14 + tw > w ? x - tw - 14 : x + 14) + 'px'; el.style.top = Math.max(4, y + 14 + th > h ? y - th - 10 : y + 14) + 'px'; }

/* ------------------------------------------------------------------ the 2D pair view */
function rScatter() {
  const el = rid('rl2-chart'); if (!el || !el.clientWidth) return;
  const [a, b] = RL.pair.split('-'), N = D.m.n, sp = RL.space;
  const w = Math.max(280, el.clientWidth || 600), h = Math.round(Math.min(w * 0.9, 460));
  let cv = $('canvas', el); if (!cv) { cv = H('canvas', { 'aria-label': 'two models against each other: one dot per mention; hover for its scores, click for its definitions' }); el.prepend(cv); r2Wire(cv); }
  const dpr = Math.min(2, devicePixelRatio || 1); cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.height = h + 'px';
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  const lw = 46, bot = 38, top = 10, rgt = 12, x = lin(0, 1, lw, w - rgt), y = lin(0, 1, h - bot, top);
  ctx.lineWidth = 1; ctx.strokeStyle = RTOK.grid; ctx.fillStyle = RTOK.muted; ctx.font = `11px ${css('--f-mono')}`;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    ctx.beginPath(); ctx.moveTo(x(v), top); ctx.lineTo(x(v), h - bot); ctx.moveTo(lw, y(v)); ctx.lineTo(w - rgt, y(v)); ctx.stroke();
    ctx.textAlign = 'center'; ctx.fillText(String(v), x(v), h - bot + 14); ctx.textAlign = 'right'; ctx.fillText(String(v), lw - 6, y(v) + 4);
  }
  const un = (k) => (k === 's' && sp === 'a' ? 'score s' : RUNIT[sp]);
  ctx.fillStyle = RTOK.ink2; ctx.font = `600 12.5px ${css('--f-body')}`; ctx.textAlign = 'center'; ctx.fillText(`${nm(a)} ${un(a)} →`, (lw + w - rgt) / 2, h - 6);
  ctx.save(); ctx.translate(13, (top + h - bot) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(`${nm(b)} ${un(b)} →`, 0, 0); ctx.restore();
  const P = rperm(), va = (i) => rv(sp, a, i), vb = (i) => rv(sp, b, i);
  ctx.globalAlpha = RL.color === 'score' ? 0.3 : 0.24; let last = '';
  for (let q = 0; q < N; q++) { const i = P[q], c = rcolStr(i); if (c !== last) { ctx.fillStyle = c; last = c; } ctx.fillRect(x(va(i)) - 1.1, y(vb(i)) - 1.1, 2.2, 2.2); }
  ctx.globalAlpha = 1;
  Object.assign(R2, { x, y, a, b, sp, w, h });
  // the cuts, where they are a line or a band in this view
  const dash = (pts, color, label, below) => {
    ctx.save(); ctx.beginPath(); ctx.moveTo(x(pts[0][0]), y(pts[0][1])); ctx.lineTo(x(pts[1][0]), y(pts[1][1]));
    ctx.globalAlpha = 0.8; ctx.lineWidth = 4.5; ctx.strokeStyle = RTOK.surface; ctx.stroke();  // a halo, so a line stays visible over dots of its own colour
    ctx.globalAlpha = 1; ctx.setLineDash([6, 4]); ctx.lineWidth = 1.8; ctx.strokeStyle = color; ctx.stroke(); ctx.restore();
    if (!label) return; const mx = (pts[0][0] + pts[1][0]) / 2, my = (pts[0][1] + pts[1][1]) / 2;
    ctx.font = `600 11.5px ${css('--f-body')}`; ctx.textAlign = 'left'; ctx.lineWidth = 3; ctx.strokeStyle = RTOK.surface; ctx.fillStyle = color;
    const tw = ctx.measureText(label).width, tx = Math.max(lw + 3, Math.min(w - rgt - tw - 3, below ? x(mx) - 6 - tw : x(mx) + 6)), ty = Math.max(top + 10, Math.min(h - bot - 4, y(my) + (below ? 14 : -6)));
    ctx.strokeText(label, tx, ty); ctx.fillText(label, tx, ty);
  };
  const clip = (wa, wb, v) => { const out = []; for (const [px, py] of [[0, v / wb], [1, (v - wa) / wb], [v / wa, 0], [(v - wb) / wa, 1]]) if (px >= -1e-9 && px <= 1 + 1e-9 && py >= -1e-9 && py <= 1 + 1e-9) out.push([px, py]); return out.slice(0, 2); };
  let cap = '';
  const cut = D.meta.cut, bf = D.meta.best_f1_cut, ns = nStrict(), sw = (i) => (D.strict[i] ? 'strict' : 'not strict');
  if (b === 's') {
    if (sp !== 'b') dash([[0, cut], [1, cut]], RTOK.accent, `cut ${f4(cut)}`, true);
    const ys = sp === 'b' ? (N - ns) / N : bf;
    dash([[0, ys], [1, ys]], RTOK.strict, `strict cut${sp === 'b' ? ` (${ordn(100 * ys)} percentile)` : ` ${f4(bf)}`}`);
    const ex = rBandExample('s');
    cap = `Up: the ensemble ${sp === 'b' ? 'percentile' : 'score'}; the dashed strict cut splits strict (above) from not strict${sp === 'b' ? '' : ', and no dot lies under the cut'}. Across: the classifier alone.`
      + (ex >= 0 ? ` “${D.m.label[ex]}” has its lowest ${RUNIT[sp]}, ${f3(va(ex))}, yet the other two bring s to ${f4(rs(ex))}: ${sw(ex)}.` : '');
  } else if (sp === 'a' && RREF.ok) {
    const t = RM.find((k) => k !== a && k !== b), wa = RW[a], wb = RW[b], wt = RW[t];
    const L0 = clip(wa, wb, cut - wt), L1 = clip(wa, wb, bf), L2 = clip(wa, wb, bf - wt);
    if (L0.length === 2) dash(L0, RTOK.accent, 'cut out of reach below', true);
    if (L1.length === 2) dash(L1, RTOK.strict, `strict whatever the ${RNAME[t]} says`);
    if (L2.length === 2) dash(L2, RTOK.kept, `not strict whatever the ${RNAME[t]} says`, true);
    const { n0, n1, n2 } = rBands(a, b, t), ex = rBandExample(t);
    cap = `Seen down the ${RNAME[t]} axis, each cut becomes a band: the hidden ${RNAME[t]} rank adds up to ${f3(wt)} to s. Strict whatever it says: ${nf(n1)}; not strict whatever it says: ${nf(n2)}; the ${RNAME[t]} decides the other ${nf(N - n1 - n2)}. Under the lowest line not even its top rank reaches the cut: ${nf(n0)}.`
      + (ex >= 0 ? ` E.g. “${D.m.label[ex]}”: its ${RNAME[t]} rank ${f3(rv('a', t, ex))} brings s to ${f4(RREF.s[ex])}, ${sw(ex)}.` : '');
  } else {
    let both = 0, one = 0; for (let i = 0; i < N; i++) { const p = D.score[a][i] >= 0.5, q2 = D.score[b][i] >= 0.5; if (p && q2) both++; else if (p !== q2) one++; }
    const t = RM.find((k) => k !== a && k !== b), ex = rTopDis(a, b, 'ab', 1)[0];
    cap = `${sp === 'b' ? 'Percentiles among MBDef’s mentions' : 'Raw scores'}. Both give 0.5 or more on ${nf(both)} (${pc(both / N)}), one but not the other on ${nf(one)}; strict and not strict overlap because the hidden ${RNAME[t]} also votes.`
      + (ex >= 0 ? ` Widest split: “${D.m.label[ex]}”, ${RNAME[a]} ${f3(D.score[a][ex])} against ${RNAME[b]} ${f3(D.score[b][ex])}, ${sw(ex)}.` : '');
  }
  rid('rl2-cap').textContent = cap + ' Hover a dot for its scores; click it for its definitions.';
}
function rBandExample(t) {
  // a mention in the middle of the strict band (2D view, space a), or the mention the classifier puts lowest (classifier and ensemble)
  const N = D.m.n; let best = -1, bd = Infinity;
  if (t === 's') { for (let i = 0; i < N; i++) { const v = rv(RL.space, 'c', i); if (v < bd) { bd = v; best = i; } } return best; }
  const [a, b] = RL.pair.split('-'), mid = D.meta.best_f1_cut - RW[t] / 2;
  for (let i = 0; i < N; i++) { if (!RREF.same[i]) continue; const d = Math.abs(RW[a] * RREF.r[a][i] + RW[b] * RREF.r[b][i] - mid); if (d < bd) { bd = d; best = i; } }
  return best;
}
function r2Wire(cv) {
  const tip = rid('rl2-tip'); let pend = null, prq = 0, last = 0;
  const near = (cx, cy, rad) => { const r = cv.getBoundingClientRect(), px = cx - r.left, py = cy - r.top, { x, y, a, b, sp } = R2, N = D.m.n; let best = -1, bd = rad * rad;
    for (let i = 0; i < N; i++) { const dx = x(rv(sp, a, i)) - px, dy = y(rv(sp, b, i)) - py, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i; } } return best; };
  const run = () => { prq = 0; if (!pend) return; const now = performance.now(); if (now - last < 40) { prq = requestAnimationFrame(run); return; } last = now; const e = pend; pend = null;
    const i = near(e.clientX, e.clientY, 8); if (i < 0) { tip.hidden = true; cv.style.cursor = ''; return; } cv.style.cursor = 'pointer'; rTip(tip, i, 'click for its definitions');
    const r = cv.getBoundingClientRect(); rPlaceTip(tip, e.clientX - r.left, e.clientY - r.top, r.width, r.height); };
  cv.addEventListener('pointermove', (e) => { pend = e; if (!prq) prq = requestAnimationFrame(run); });
  cv.addEventListener('pointerleave', () => { pend = null; tip.hidden = true; });
  cv.addEventListener('click', (e) => { const i = near(e.clientX, e.clientY, 10); if (i >= 0) openMention(i); });
}

/* ------------------------------------------------------------------ where two members disagree */
function rTopDis(a, b, dir, n = 12) {
  const N = D.m.n, sp = RL.space === 'a' && !RREF.ok ? 'b' : RL.space, top = [];
  for (let i = 0; i < N; i++) {
    const d = dir === 'ab' ? rv(sp, a, i) - rv(sp, b, i) : rv(sp, b, i) - rv(sp, a, i);
    if (top.length < n || d > top[top.length - 1][0]) { top.push([d, i]); top.sort((u, v) => v[0] - u[0]); if (top.length > n) top.pop(); }
  }
  return top.map((t) => t[1]);
}
function rDis() {
  const el = rid('rl-dis-what'); if (!el) return;
  const [a, b] = RL.pair.split('-'), sp = RL.space, N = D.m.n;
  const t = b === 's' ? null : RM.find((k) => k !== a && k !== b), ix = rTopDis(a, b, RL.dis);
  const hi = RL.dis === 'ab' ? a : b, lo = RL.dis === 'ab' ? b : a, big = rBig(a, b);
  el.textContent = `The 12 mentions the ${nm(hi)} puts furthest above the ${nm(lo)}, by ${RSPACE[sp]}. ${big ? `${nf(big)} of ${nf(N)} (${pc(big / N)})` : 'None'} differ by over half the scale.`;
  cardsFor(ix, rid('rl-dis-cards'), (i) => `${nm(a)} ${f2(rv(sp, a, i))} · ${nm(b)} ${f2(rv(sp, b, i))}${t ? ` · ${RNAME[t]} ${f2(rv(sp, t, i))}` : ''}`);
}

/* ------------------------------------------------------------------ the percentile browser */
function rCum(k) {
  if (RCUM[k]) return RCUM[k];
  const o = D.order[k], N = o.length, str = new Uint32Array(N + 1);
  for (let q = 0; q < N; q++) str[q + 1] = str[q] + D.strict[o[q]];
  return (RCUM[k] = { str });
}
const RSTRIP = { padL: 30, padR: 12, top: 12, ch: 70, band: 10 };
function rStrip() {
  const el = rid('rl-strip'); if (!el || !el.clientWidth) return;
  const k = RL.k, o = D.order[k], N = o.length, sc = D.score[k], { padL, padR, top, ch, band } = RSTRIP;
  const w = Math.max(260, el.clientWidth || 600), h = el.clientHeight || 96;
  let cv = $('canvas', el); if (!cv) { cv = H('canvas'); el.append(cv); }
  const dpr = Math.min(2, devicePixelRatio || 1); cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  const x = lin(0, 1, padL, w - padR), y = lin(0, 1, top + ch, top), cols = Math.max(20, Math.round(w - padL - padR)), at = (p) => sc[o[Math.min(N - 1, Math.max(0, Math.round(p * (N - 1))))]];
  ctx.beginPath(); ctx.moveTo(x(0), y(0)); for (let c = 0; c <= cols; c++) ctx.lineTo(x(c / cols), y(at(c / cols))); ctx.lineTo(x(1), y(0)); ctx.closePath();
  ctx.fillStyle = RTOK.wash; ctx.fill(); ctx.beginPath(); for (let c = 0; c <= cols; c++) ctx[c ? 'lineTo' : 'moveTo'](x(c / cols), y(at(c / cols))); ctx.strokeStyle = RTOK.accent; ctx.lineWidth = 1.6; ctx.stroke();
  // the strict band: in each sliver of percentiles, the share strict (violet) and not strict (teal)
  const cum = rCum(k), by = top + ch + 4;
  for (let c = 0; c < cols; c++) {
    const q0 = Math.floor((c / cols) * N), q1 = Math.max(q0 + 1, Math.floor(((c + 1) / cols) * N)), n = q1 - q0, s = (cum.str[q1] - cum.str[q0]) / n;
    let yy = by; for (const [f, col] of [[s, RTOK.strict], [1 - s, RTOK.kept]]) { if (f <= 0) continue; ctx.fillStyle = col; ctx.fillRect(x(c / cols), yy, (w - padL - padR) / cols + 0.6, band * f); yy += band * f; }
  }
  ctx.font = `10.5px ${css('--f-mono')}`; ctx.fillStyle = RTOK.muted; ctx.textAlign = 'right'; ctx.fillText('1', padL - 5, top + 4); ctx.fillText('0', padL - 5, top + ch + 2);
  for (const v of [0, 25, 50, 75, 100]) { ctx.textAlign = v === 0 ? 'left' : v === 100 ? 'right' : 'center'; ctx.fillText(v === 0 ? 'lowest' : v === 100 ? 'highest' : `${v}th`, x(v / 100), by + band + 13); }
  if (k === 's') for (const [v, col, nmc] of [[D.meta.cut, RTOK.accent, 'cut'], [D.meta.best_f1_cut, RTOK.strict, 'strict']]) {
    ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x(0), y(v)); ctx.lineTo(x(1), y(v)); ctx.stroke(); ctx.restore();
    ctx.fillStyle = col; ctx.textAlign = 'right'; ctx.fillText(nmc, x(1) - 3, nmc === 'cut' ? y(v) + 11 : y(v) - 4);
  }
  const px = x(RL.p), pv = at(RL.p);
  ctx.strokeStyle = RTOK.ink; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(px, top - 4); ctx.lineTo(px, by + band + 2); ctx.stroke();
  ctx.fillStyle = RTOK.surface; ctx.beginPath(); ctx.arc(px, y(pv), 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = RTOK.ink; ctx.stroke();
  el.setAttribute('aria-valuenow', (100 * RL.p).toFixed(1)); el.setAttribute('aria-valuetext', `${ordn(100 * RL.p)} percentile, ${SC[k]} score ${f3(pv)}`);
  // one line, with numbers from the data
  const med = at(0.5), cap = rid('rl-strip-cap'); cap.textContent = '';
  const share = (v) => lowerB(Float64Array.from(o, (i) => sc[i]), v) / N;
  if (k === 's') {
    const ns = cum.str[N];
    cap.append(rt(`The ensemble score at each percentile of MBDef’s ${nf(N)} mentions. The `), gl('bestf1', 'strict cut'), rt(` sits at the ${ordn(100 * (1 - ns / N))} percentile; the band shows each sliver’s share strict (violet) and not (teal). Drag the marker or use the arrow keys.`));
  } else {
    const n0 = Math.floor(0.1 * N), strictLow = cum.str[n0] / n0;
    cap.append(rt('The '), gl(SC[k], SC[k]), rt(` score at each `), gl('percentile', 'percentile'), rt(`: half score above ${med >= 0.9995 ? '0.999' : f3(med)}, ${pc(1 - share(0.5))} score 0.5 or more. Of the tenth it scores lowest, ${pc(strictLow)} are still strict: the other two outvote it.`));
  }
}
function rStripWire() {
  const el = rid('rl-strip'); let drag = false, rq = 0, cardT = 0;
  const setP = (p, cards) => { RL.p = Math.max(0, Math.min(1, p)); if (!rq) rq = requestAnimationFrame(() => { rq = 0; rStrip(); rReadout(); }); clearTimeout(cardT); cardT = setTimeout(rByPct, cards ? 0 : 160); };
  const fromX = (cx) => { const r = el.getBoundingClientRect(); return (cx - r.left - RSTRIP.padL) / (r.width - RSTRIP.padL - RSTRIP.padR); };
  el.addEventListener('pointerdown', (e) => { drag = true; try { el.setPointerCapture(e.pointerId); } catch (_) { /* a synthetic pointer */ } setP(fromX(e.clientX)); });
  el.addEventListener('pointermove', (e) => { if (drag) setP(fromX(e.clientX)); });
  el.addEventListener('pointerup', (e) => { if (!drag) return; drag = false; setP(fromX(e.clientX), true); });
  el.addEventListener('pointercancel', () => { drag = false; });
  el.addEventListener('keydown', (e) => { const st = { ArrowLeft: -0.005, ArrowRight: 0.005, ArrowDown: -0.005, ArrowUp: 0.005, PageDown: -0.05, PageUp: 0.05 }[e.key]; if (st != null) { e.preventDefault(); setP(RL.p + st); } else if (e.key === 'Home') { e.preventDefault(); setP(0, true); } else if (e.key === 'End') { e.preventDefault(); setP(1, true); } });
}
function rReadout() { const el = rid('rl-pctv'); if (!el) return; const k = RL.k, o = D.order[k], N = o.length, at = Math.round(RL.p * (N - 1)); el.textContent = `${ordn(100 * RL.p)} percentile · ${SC[k]} score ${f3(D.score[k][o[at]])} · ${nf(at)} of ${nf(N)} mentions score lower`; }
function rByPct() {
  if (!rid('rl-cards')) return;
  const k = RL.k, o = D.order[k], N = o.length, at = Math.round(RL.p * (N - 1)), ix = o.slice(Math.max(0, Math.min(N - 12, at - 6)), Math.max(12, Math.min(N, at + 6)));
  rReadout();
  rid('rl-what').textContent = `The 12 mentions around the ${ordn(100 * RL.p)} percentile of the ${SC[k]} score, lowest first.`;
  cardsFor(ix, rid('rl-cards'), k === 's' ? null : (i) => `${SC[k]} ${f3(D.score[k][i])}`);
}
function rAroundStrict() {  // by the exact ensemble score: near the strict cut the rounded one sorts RREF.misS4 mentions the other way
  const c = D.meta.best_f1_cut, near = []; for (let i = 0; i < D.m.n; i++) if (Math.abs(RREF.s[i] - c) < 0.01) near.push(i);
  near.sort((a, b) => RREF.s[a] - RREF.s[b]); const k = near.findIndex((i) => RREF.s[i] >= c); return near.slice(k - 6, k + 6);
}
function rQuick(q) {
  const N = D.m.n; let ix, what;
  if (q === 'up' || q === 'down') { RL.p = Math.max(0, Math.min(1, RL.p + (q === 'up' ? 12 : -12) / (N - 1))); rStrip(); rByPct(); return; }
  if (q === 'strict' && RL.k !== 's') { RL.k = 's'; rSegs(); }
  const o = D.order[RL.k];
  if (q === 'top') { RL.p = 1; ix = o.slice(N - 12).reverse(); what = `The 12 mentions the ${SC[RL.k]} scores highest, highest first.`; }
  else if (q === 'bottom') { RL.p = 0; ix = o.slice(0, 12); what = `The 12 mentions the ${SC[RL.k]} scores lowest, lowest first.`; }
  else if (q === 'strict') { const lo = N - nStrict(); RL.p = lo / (N - 1); ix = RREF.ok ? rAroundStrict() : o.slice(lo - 6, lo + 6); what = `The 6 highest-scoring mentions that are not strict and the 6 lowest-scoring strict ones, around the strict cut ${f4(D.meta.best_f1_cut)}.`; }
  else { ix = Array.from({ length: 12 }, () => Math.floor(Math.random() * N)); what = `12 mentions drawn at random from all ${nf(N)}.`; }
  rStrip(); rReadout();
  rid('rl-what').textContent = what; cardsFor(ix, rid('rl-cards'), RL.k === 's' ? null : (i) => `${SC[RL.k]} ${f3(D.score[RL.k][i])}`);
}
TABS.relevance = {
  init: relevance,
  onShow: () => { R3.resize && R3.resize(); },
  view: (v, instant) => rSetView(v, instant),
  screenOf: (i) => R3.screenOf && R3.screenOf(i),
  value: (sp, k, i) => rv(sp, k, i),
  space: (sp) => rSetSpace(sp, true),
  explain: (i) => { RL.ex = i; rExplain(); R3.mark && R3.mark(); },
  state: () => ({ space: RL.space, view: RL.view, pair: RL.pair, ex: RL.ex, ref_ok: RREF.ok, why: RREF.why, behind_cut: RREF.misK, behind_cut_32bit_positions: RREF.mis32, behind_cut_by_stored_4dp_s: RREF.mis4,
    strict_misplaced: RREF.misS, strict_misplaced_32bit_positions: RREF.misS32, strict_misplaced_by_stored_4dp_s: RREF.misS4, max_abs_s_diff: RREF.maxd, closest_past_cut: RREF.near, closest_to_strict_cut: RREF.nearS,
    stored_check: RREF.checks && RREF.checks.released, ids: RREF.hash, recomputable_from_4dp: RREF.nsame, pr: rPR(), n_labelled: rNLab() }),
};
