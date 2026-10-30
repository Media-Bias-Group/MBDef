/* overview tab, the landing page: registers TABS.overview = { hero(meta), init(), onShow() }. Uses the helpers of core.js.
   Three parts:
     1. the hero, on the stage: title, the four headline numbers, a search over the terms and a live 3D map of every mention in
        its topic's colour, turning slowly. core.js calls hero(meta) as soon as data/meta.json is in, so all of it paints before
        the rest of the data. The map fills in from the mention shards (dx:shard: part.xyz, x1000 ints, and part.topic); once
        every mention is in it takes the Map tab's frame (as mapFrame(): centred, 98% of the dots within radius 10) and colours
        (mapFrame()'s colour slots in the dark-ground tier, as topicPalette(mapFrame(), true) gives them for the Map tab's dark
        stage), and init() switches to mapFrame() and topicPalette() themselves without rebuilding anything.
     2. what you can do: one card per view, each with a small picture drawn from the data.
     3. the dataset in numbers: the secondary counts, then every chart and worked example in an expander, built on first open.
   Every number is counted from D (or meta, before D is in) when it is drawn; every example is picked by a rule named beside it. */

const OV = { built: false, N: 0, NT: 0, shard: 1, nShards: 0, got: 0, pct: 0, settled: false, mix: 1, mixT0: null, lastBirth: 0 };
const ovNow = () => performance.now() / 1000;
const ovMotion = () => !matchMedia('(prefers-reduced-motion: reduce)').matches;
const ovEase = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const ovK = (v) => (v >= 1e4 ? Math.round(v / 1e3) + 'k' : v >= 1e3 ? (v / 1e3).toFixed(1) + 'k' : String(v));
const ovQ = (t) => `“${t}”`;
const ovShort = (t, n = 70) => (t.length > n ? t.slice(0, n - 1).trimEnd() + '…' : t);
const ovTitle = (p) => D.p.title[p] || D.p.id[p];
const ovBtn = (text, on) => H('button', { type: 'button', class: 'linkbtn', text, onclick: on });
const ovGo = (tab) => () => { show(tab); scrollTo({ top: 0 }); };
const ovMax = (g) => D.members[g].reduce((b, i) => (b < 0 || D.score.s[i] > D.score.s[b] ? i : b), -1);   // a term's highest-scoring mention
const ovSpell = (g) => (D.merges[g] ? D.merges[g].labels.map((l) => [l[0], l[4]]) : Object.entries(D.g.labels[g])).sort((a, b) => b[1] - a[1]);   // its spellings, most used first
function ovOpenConcept(g) { return () => { show('concepts'); scrollTo({ top: 0 }); if (TABS.concepts && TABS.concepts.open) TABS.concepts.open(g); else openMention(ovMax(g)); }; }
// where topic names would overlap: the topic led by "media bias" first, then the other topics led by a bias concept (as the Map tab)
const ovBiasLed = (tps) => tps.map((tp) => (tp.top[0].toLowerCase() === 'media bias' ? 2 : /bias/i.test(tp.top[0]) ? 1 : 0));

/* ================================================================== 1. the hero */
function ovHero(meta) {
  if (OV.built) return; OV.built = true;
  const c = meta.counts, tps = meta.topics, NT = tps.length, N = c.terms;
  Object.assign(OV, { meta, N, NT, shard: meta.shard, nShards: Math.ceil(N / meta.shard), bias: ovBiasLed(tps), pal: tiered(NT, true) });
  OV.raw = new Float32Array(N * 3); OV.topic = new Uint8Array(N); OV.born = new Float32Array(N).fill(-1e6);   // -1e6: not in yet
  OV.col0 = new Uint8Array(N * 3); OV.col1 = new Uint8Array(N * 3); OV.have = new Uint8Array(OV.nShards);
  OV.dirty = { pos: false, col: false, born: false };

  // the search's examples: the lead terms of the bias-led topics, "media bias" first, of two words or more
  const leads = tps.map((tp, t) => t).filter((t) => OV.bias[t]).sort((a, b) => OV.bias[b] - OV.bias[a] || tps[b].size - tps[a].size).map((t) => tps[t].top[0]).filter((l) => /\s/.test(l)).slice(0, 2);
  const copy = H('div', { class: 'ov-copy' },
    H('p', { class: 'eyebrow' }, 'Media Bias Group · ', H('span', { class: 'ov-nc', text: 'MBDef' }), ' dataset'),
    H('h1', { class: 'ov-title', text: 'MBDef Explorer' }),
    H('p', { class: 'ov-lede', text: 'The terms that media-bias research defines, merged across papers and mapped by meaning: open any one to see where each definition comes from.' }),
    ovSearch(leads), ovTry());
  const stats = H('div', { class: 'stats ov-stats' },
    stat({ value: c.concepts, label: 'terms', sub: 'merged across papers', lead: true, onclick: ovGo('concepts') }),
    stat({ value: c.definitions, label: 'definitions', sub: `${f2(c.definitions / N)} per term mention`, onclick: ovGo('concepts') }),
    stat({ value: c.papers, label: 'papers', sub: `${nf(c.seeds.papers)} seed · ${nf(c.closure.papers)} cited`, onclick: ovGo('papers') }),
    stat({ value: NT, label: 'topics', sub: 'the map’s colours', onclick: ovGo('map') }));
  const map = H('div', { class: 'ov-map', id: 'ov-map' },
    H('canvas', { id: 'ov-canvas', tabindex: '0', role: 'button', 'aria-label': `A 3D map of the ${nf(N)} term mentions in ${NT} topics: dots close together have similar definitions. Press Enter to open the Map tab.` }),
    H('div', { class: 'ov-labs', id: 'ov-labs', 'aria-hidden': 'true' }),
    H('div', { class: 'ov-mapcap' }, H('span', { class: 'ov-mapst', id: 'ov-mapst' }), H('button', { type: 'button', class: 'ov-mapgo', text: 'Open the map →', onclick: ovGo('map') })));
  const hero = $('#ov-hero'); hero.textContent = '';
  hero.append(map, H('div', { class: 'ov-scrim', 'aria-hidden': 'true' }), copy, stats);
  ovStatus();

  document.addEventListener('dx:shard', ovShard);
  document.addEventListener('dx:progress', (e) => { OV.pct = e.detail.done / e.detail.total; ovStatus(); if (OV.search) OV.search.progress(); });
  ovCardsShell();
  $('#ov-num-sec').append(secHead('The dataset in numbers', null), H('p', { class: 'small muted ov-wait', text: 'Counting…' }));
  ovStartGL();
}
function ovStatus() {
  const el = $('#ov-mapst'); if (!el) return;
  el.textContent = '';
  if (!OV.settled) el.append(`Loading ${nf(OV.N)} mentions · ${Math.round(100 * OV.pct)}%`);
  else el.append(`${nf(OV.N)} mentions · ${OV.NT} topics`, H('span', { class: 'ov-wide', text: ` · ${matchMedia('(pointer: coarse)').matches ? 'swipe' : 'drag'} to turn` }));
}

/* ---- the mentions come in shard by shard; until all are in, the frame is estimated from those that have come */
const OV_STAR = [222, 228, 236];
const ovSoft = (t) => OV.pal[t % OV.pal.length].map((v, j) => Math.round(OV_STAR[j] + (v - OV_STAR[j]) * 0.32));   // a hint of colour while loading
function ovPaint(arr, i, rgb) { arr[i * 3] = rgb[0]; arr[i * 3 + 1] = rgb[1]; arr[i * 3 + 2] = rgb[2]; }
function ovShard(e) {
  const { s, part } = e.detail; if (!OV.have || OV.have[s]) return;
  OV.have[s] = 1; OV.got++;
  const off = s * OV.shard, n = Math.min(part.n, OV.N - off), move = ovMotion(), base = !move ? -50 : OV.gl ? ovNow() : 0;
  for (let k = 0; k < n; k++) {
    const i = off + k, t = part.topic[k];
    OV.raw[i * 3] = part.xyz[k * 3]; OV.raw[i * 3 + 1] = part.xyz[k * 3 + 1]; OV.raw[i * 3 + 2] = part.xyz[k * 3 + 2];
    OV.topic[i] = t; OV.born[i] = base + (move ? Math.random() * 0.9 : 0);   // they twinkle in over a moment
    const rgb = ovSoft(t); ovPaint(OV.col0, i, rgb); ovPaint(OV.col1, i, rgb);
  }
  OV.lastBirth = base + 0.9;
  OV.dirty.pos = OV.dirty.col = OV.dirty.born = true;
  if (OV.got === OV.nShards) ovSettle(); else ovRefitSoon();
}
let ovRefitT = 0;
function ovRefitSoon() { if (!ovRefitT) ovRefitT = setTimeout(() => { ovRefitT = 0; if (!OV.settled) ovRefit(false); }, 200); }
// the frame, as mapFrame(): centre on the mean, scale so that 98% of the dots lie within radius 10. While loading, every
// third mention that has come in; then all of them, which gives mapFrame()'s numbers exactly. ext: what the camera fits.
function ovRefit(exact) {
  const N = OV.N, P = OV.raw, B = OV.born, step = exact ? 1 : 3, m = [0, 0, 0];
  let n = 0;
  for (let i = 0; i < N; i += step) if (B[i] > -1e5) { m[0] += P[i * 3]; m[1] += P[i * 3 + 1]; m[2] += P[i * 3 + 2]; n++; }
  if (!n) return;
  for (let a = 0; a < 3; a++) m[a] /= n;
  const r = new Float32Array(n), rxz = new Float32Array(n), ys = new Float32Array(n);
  for (let i = 0, j = 0; i < N; i += step) {
    if (B[i] <= -1e5) continue;
    let s = 0; for (let a = 0; a < 3; a++) { const v = P[i * 3 + a] - m[a]; s += v * v; }
    r[j] = Math.sqrt(s); rxz[j] = Math.hypot(P[i * 3] - m[0], P[i * 3 + 2] - m[2]); ys[j] = P[i * 3 + 1] - m[1]; j++;
  }
  r.sort(); rxz.sort(); ys.sort();
  const k = 10 / (r[Math.floor(0.98 * (n - 1))] || 1), q = (a, f) => a[Math.floor(f * (n - 1))] * k;
  OV.fr = { c: m, k };
  OV.ext = { r: q(rxz, 0.97), y0: q(ys, 0.015), y1: q(ys, 0.985) };
  if (OV.gl) OV.gl.refit();
}
// every mention is in: the exact frame, the topic anchors and colour slots as mapFrame() makes them, the colours bloom
function ovSettle() {
  OV.settled = true;
  ovRefit(true);
  const N = OV.N, NT = OV.NT, P = OV.raw, { c, k } = OV.fr;
  const members = Array.from({ length: NT }, () => []);
  for (let i = 0; i < N; i++) members[OV.topic[i]].push(i);
  const anc = new Float32Array(NT * 3);   // the member nearest each topic's coordinate-wise median (raw coordinates: the same member)
  members.forEach((ix, t) => {
    if (!ix.length) return;
    const med = [0, 1, 2].map((a) => { const v = Float32Array.from(ix, (i) => P[i * 3 + a]).sort(); return v[v.length >> 1]; });
    let best = ix[0], bd = Infinity;
    for (const i of ix) { const d = (P[i * 3] - med[0]) ** 2 + (P[i * 3 + 1] - med[1]) ** 2 + (P[i * 3 + 2] - med[2]) ** 2; if (d < bd) { bd = d; best = i; } }
    anc.set(P.subarray(best * 3, best * 3 + 3), t * 3);
  });
  OV.anc = anc;
  OV.slot = ovSlots(anc.map((v, j) => (v - c[j % 3]) * k), members.map((ix) => ix.length));
  OV.tcol = Array.from(OV.slot, (s) => OV.pal[s]);
  ovFinalColours(); ovStatus();
  if (OV.gl) OV.gl.labels();
}
// mapFrame()'s colour slots: the biggest topic first; each takes the free colour farthest (in OKLab) from its 6 nearest coloured neighbours
function ovSlots(anc3, size) {
  const NT = size.length, pal = tiered(NT, false).map(okLab), order = [...size.keys()].sort((a, b) => size[b] - size[a]);
  const slot = new Int16Array(NT).fill(-1), used = new Set();
  const dist = (a, b) => Math.hypot(anc3[a * 3] - anc3[b * 3], anc3[a * 3 + 1] - anc3[b * 3 + 1], anc3[a * 3 + 2] - anc3[b * 3 + 2]);
  for (const t of order) {
    const near = order.filter((o) => slot[o] >= 0).map((o) => [dist(t, o), o]).sort((a, b) => a[0] - b[0]).slice(0, 6);
    let best = -1, bs = -1;
    for (let s = 0; s < pal.length; s++) {
      if (used.has(s)) continue;
      let sc = Infinity; near.forEach(([, o], j) => { sc = Math.min(sc, okDist(pal[s], pal[slot[o]]) * (1 + 0.12 * j)); });
      if (sc > bs + 1e-9) { bs = sc; best = s; }
    }
    slot[t] = best < 0 ? t % pal.length : best; used.add(slot[t]);
  }
  return slot;
}
// the final topic colours; what shows now becomes the start of a short fade (no fade before the 3D view runs)
function ovFinalColours() {
  const N = OV.N, now = OV.gl ? ovEase(OV.mix) : 1;
  if (OV.gl && ovMotion()) for (let j = 0; j < N * 3; j++) OV.col0[j] = Math.round(OV.col0[j] + (OV.col1[j] - OV.col0[j]) * now);
  for (let i = 0; i < N; i++) ovPaint(OV.col1, i, OV.tcol[OV.topic[i]]);
  if (OV.gl && ovMotion()) { OV.mix = 0; OV.mixT0 = null; } else { OV.col0.set(OV.col1); OV.mix = 1; }
  OV.dirty.col = true;
  if (OV.gl) OV.gl.paint();
}
// init(): every mention from D (when the hero came late), then the Map tab's own frame, anchors and slots
function ovFromData() {
  const N = OV.N, m = D.m;
  for (let s = 0; s < OV.nShards; s++) {
    if (OV.have[s]) continue;
    OV.have[s] = 1; OV.got++;
    for (let i = s * OV.shard; i < Math.min(N, (s + 1) * OV.shard); i++) {
      for (let a = 0; a < 3; a++) OV.raw[i * 3 + a] = m.xyz[i * 3 + a];
      OV.topic[i] = m.topic[i]; OV.born[i] = -50;
    }
    OV.dirty.pos = OV.dirty.born = true;
  }
  if (!OV.settled) ovSettle();
  const F = mapFrame(), { c, k } = OV.fr;
  OV.anc = Float32Array.from(F.anc3, (v, j) => v / k + c[j % 3]);
  const tcol = topicPalette(F, true);   // the Map tab's topic colours on its dark stage
  if (tcol.some((v, t) => v.some((x, j) => x !== OV.tcol[t][j]))) { OV.tcol = tcol; ovFinalColours(); }
  OV.F = F;
  if (OV.gl) OV.gl.labels();
}

/* ---- the 3D view: three.js (the import map's "three", as the Map tab), one Points cloud with a small shader */
async function ovStartGL() {
  let THREE;
  try { THREE = await import('three'); } catch (e) { ovNote('The 3D view could not load its library (three.js from cdn.jsdelivr.net).'); return; }
  try { ovBuildGL(THREE); } catch (e) { console.error(e); ovNote('The 3D view could not be drawn: ' + e.message); }
}
function ovNote(t) { const el = $('#ov-map'); if (el) el.append(H('p', { class: 'ov-mapnote', text: t })); }
function ovBuildGL(THREE) {
  const sec = $('#v-overview'), box = $('#ov-map'), canvas = $('#ov-canvas'), layer = $('#ov-labs');
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' }); }
  catch (e) { ovNote('The 3D view needs WebGL, which this browser refused.'); return; }
  THREE.ColorManagement.enabled = false;   // every colour is already sRGB, as in the Map tab
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  const DPR = Math.min(devicePixelRatio || 1, 2); renderer.setPixelRatio(DPR); renderer.setClearColor(0x000000, 0);
  const glc = renderer.getContext(), maxPx = (glc.getParameter(glc.ALIASED_POINT_SIZE_RANGE) || [1, 64])[1];
  const FOV = 34, tanV = Math.tan((FOV * Math.PI) / 360), scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 800);
  const N = OV.N, geom = new THREE.BufferGeometry();
  const A = { pos: new THREE.BufferAttribute(OV.raw, 3), c0: new THREE.BufferAttribute(OV.col0, 3, true), c1: new THREE.BufferAttribute(OV.col1, 3, true), born: new THREE.BufferAttribute(OV.born, 1) };
  for (const a of Object.values(A)) a.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute('position', A.pos); geom.setAttribute('aC0', A.c0); geom.setAttribute('aC1', A.c1); geom.setAttribute('aBorn', A.born);
  geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e9);
  const U = { uC: { value: new THREE.Vector3() }, uK: { value: 1 }, uTime: { value: 0 }, uMix: { value: 1 }, uSize: { value: 0.062 }, uScale: { value: 500 }, uPR: { value: DPR },
    uMinPx: { value: 1.25 }, uMaxPx: { value: Math.min(9, maxPx / DPR) }, uNear: { value: 20 }, uFar: { value: 40 } };
  // a round dot whose size shrinks with distance; dots far back fade into the stage; a new dot grows in over 0.8 s
  const mat = new THREE.ShaderMaterial({
    uniforms: U, transparent: true, depthTest: true, depthWrite: true,
    vertexShader: `uniform vec3 uC; uniform float uK, uTime, uMix, uSize, uScale, uPR, uMinPx, uMaxPx, uNear, uFar;
      attribute vec3 aC0; attribute vec3 aC1; attribute float aBorn; varying vec3 vCol; varying float vA;
      void main() {
        if (aBorn < -1e5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
        vec4 mv = modelViewMatrix * vec4((position - uC) * uK, 1.0);
        gl_Position = projectionMatrix * mv;
        float age = clamp((uTime - aBorn) / 0.8, 0.0, 1.0), depth = max(-mv.z, 0.001);
        float fog = clamp((depth - uNear) / max(uFar - uNear, 0.001), 0.0, 1.0);
        vCol = mix(aC0, aC1, uMix) * (1.0 - 0.3 * fog); vA = age * (1.0 - 0.6 * fog);
        gl_PointSize = clamp(uSize * uScale / depth, uMinPx, uMaxPx) * uPR * (0.35 + 0.65 * age);
      }`,
    fragmentShader: `varying vec3 vCol; varying float vA;
      void main() {
        vec2 c = gl_PointCoord * 2.0 - 1.0; float r = length(c), aa = max(fwidth(r) * 1.2, 0.04);
        float a = (1.0 - smoothstep(1.0 - aa, 1.0, r)) * vA; if (a < 0.02) discard;
        vec3 col = vCol * (1.0 - 0.18 * smoothstep(0.4, 1.0, r)) + 0.06 * (1.0 - smoothstep(0.0, 0.6, length(c - vec2(-0.3, -0.3))));
        gl_FragColor = vec4(col, a);
      }`,
  });
  const pts = new THREE.Points(geom, mat); pts.frustumCulled = false; scene.add(pts);
  const t0 = ovNow(); for (let i = 0; i < N; i++) if (OV.born[i] >= 0) OV.born[i] += t0;   // what came before three.js twinkles in from now
  if (OV.lastBirth >= 0) OV.lastBirth += t0;

  const SPIN = ovMotion() ? 0.07 : 0, coarse = matchMedia('(pointer: coarse)').matches;
  const st = { yaw: 0.55, vel: SPIN, el: 0.3, d: 0, dT: 30, ty: 0, tyT: 0, W: 1, H: 1, side: 0, ex: [], on: true, drag: null, idle: 9, last: 0, lastR: 0, needs: true, budget: 10 };
  const cur = { c: null, k: 1 };

  /* ---- where the cloud goes: the part of the stage that the title and the numbers leave free */
  function measure() {
    const b = box.getBoundingClientRect(); st.ex = []; let side = 0;
    for (const el of [$('.ov-copy'), $('.ov-stats'), $('.ov-mapcap')]) {
      const r = el && el.getBoundingClientRect(); if (!r || !r.width || r.bottom <= b.top || r.top >= b.bottom || r.right <= b.left || r.left >= b.right) continue;
      st.ex.push([r.left - b.left - 6, r.top - b.top - 6, r.right - b.left + 6, r.bottom - b.top + 6]);
      if (!el.classList.contains('ov-mapcap')) side = Math.max(side, r.right - b.left + 20);
    }
    st.side = side < st.W * 0.7 ? side : 0;
    st.budget = Math.max(5, Math.min(16, Math.round(((st.W - st.side) * st.H) / 24000)));
  }
  function fit() {
    const e = OV.ext; if (!e) return;
    const fw = Math.max(120, st.W - st.side - 8), fh = Math.max(120, st.H - 40);
    const halfV = ((e.y1 - e.y0) / 2) * Math.cos(st.el) + e.r * Math.abs(Math.sin(st.el));   // the turning cloud's half height, seen from the camera's height
    st.dT = Math.max(e.r / (tanV * (fw / st.H)), halfV / (tanV * (fh / st.H))) / 0.95 + e.r * 0.28;
    st.tyT = (e.y0 + e.y1) / 2;
    if (!st.d) { st.d = st.dT; st.ty = st.tyT; }
    st.needs = true;
  }
  function applyView() {   // centre the cloud in the free part, a little above the caption line
    const ox = -st.side / 2, oy = 14;
    camera.setViewOffset(st.W, st.H, ox, oy, st.W, st.H); camera.updateProjectionMatrix();
  }
  function resize() {
    const w = box.clientWidth, h = box.clientHeight; if (!w || !h) return;
    st.W = w; st.H = h; renderer.setSize(w, h, false); camera.aspect = w / h;
    U.uScale.value = h / (2 * tanV); U.uSize.value = 0.062 * Math.min(1.8, Math.max(1, 680 / h));
    measure(); fit(); applyView(); st.needs = true;
  }

  /* ---- topic names: one per topic at its anchor, placed without overlaps in priority order (media bias, then the other
     bias-led topics, then the biggest); a name already shown keeps its place before a new one takes it, so they do not flicker */
  const v3 = new THREE.Vector3(), labW = new Float32Array(OV.NT), shown = new Uint8Array(OV.NT);
  let labEls = null;
  function makeLabels() {
    if (labEls || !OV.anc) return;
    labEls = OV.meta.topics.map((tp) => { const e = H('span', { class: 'ov-lab' + (tp.size >= 2500 ? ' big' : ''), text: tp.top[0] }); layer.append(e); return e; });
    paint(); st.needs = true;
  }
  function paint() { if (labEls && OV.tcol) labEls.forEach((e, t) => e.style.setProperty('--lc', rgbStr(OV.tcol[t]))); }
  function placeLabels() {
    if (!labEls || !cur.c) return;
    pts.updateMatrixWorld(); camera.updateMatrixWorld();
    const W = st.W, Hh = st.H, placed = st.ex.map((r) => r.slice()), cand = [], tps = OV.meta.topics;
    for (let t = 0; t < OV.NT; t++) {
      v3.set((OV.anc[t * 3] - cur.c[0]) * cur.k, (OV.anc[t * 3 + 1] - cur.c[1]) * cur.k, (OV.anc[t * 3 + 2] - cur.c[2]) * cur.k).applyMatrix4(pts.matrixWorld);
      const near = v3.z; v3.project(camera);
      if (v3.z < -1 || v3.z > 1) continue;
      cand.push({ t, x: (v3.x * 0.5 + 0.5) * W, y: (0.5 - v3.y * 0.5) * Hh, near, pri: OV.bias[t] * 1e8 + shown[t] * 1e6 + tps[t].size });
    }
    cand.sort((a, b) => b.pri - a.pri);
    const on = new Uint8Array(OV.NT); let n = 0;
    for (const c of cand) {
      if (n >= st.budget) break;
      const el = labEls[c.t]; if (!labW[c.t]) labW[c.t] = el.offsetWidth || 90;
      const w = labW[c.t], h = 24, x = Math.min(Math.max(c.x - w / 2, 6), W - w - 6), y = Math.min(Math.max(c.y - h / 2, 6), Hh - h - 6);
      const bx = [x - 4, y - 3, x + w + 4, y + h + 3];
      if (placed.some((p) => bx[0] < p[2] && bx[2] > p[0] && bx[1] < p[3] && bx[3] > p[1])) continue;
      placed.push(bx); on[c.t] = 1; n++;
      el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
      el.style.opacity = c.near < -3 ? '0.6' : '1';   // names on the far side of the cloud are dimmer
    }
    for (let t = 0; t < OV.NT; t++) { if (!on[t]) labEls[t].style.opacity = '0'; shown[t] = on[t]; }
  }

  /* ---- drag to turn (sideways on a touch screen, so the page still scrolls), click or Enter to open the Map tab */
  const toMap = ovGo('map');
  canvas.addEventListener('pointerdown', (e) => { if (e.button !== 0) return; st.drag = { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, t: performance.now(), moved: 0, type: e.pointerType }; st.vel = 0; try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* no capture: the drag still works inside the canvas */ } });
  canvas.addEventListener('pointermove', (e) => {
    const d = st.drag; if (!d) return;
    const dx = e.clientX - d.x, dy = e.clientY - d.y, now = performance.now(), dt = Math.max(8, now - d.t) / 1000;
    d.x = e.clientX; d.y = e.clientY; d.t = now; d.moved = Math.max(d.moved, Math.hypot(e.clientX - d.x0, e.clientY - d.y0));
    const dyaw = dx * 0.006; st.yaw += dyaw; st.vel = 0.6 * st.vel + 0.4 * (dyaw / dt);
    if (d.type !== 'touch' && dy) { st.el = Math.max(-0.4, Math.min(1.1, st.el + dy * 0.004)); fit(); }
    st.idle = 0; st.needs = true;
  });
  const end = (e) => { const d = st.drag; st.drag = null; if (!d) return; if (e.type === 'pointerup' && d.moved < 6) { st.vel = SPIN; toMap(); } };
  canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toMap(); }
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); st.yaw += e.key === 'ArrowLeft' ? -0.25 : 0.25; st.idle = 0; st.needs = true; }
  });

  /* ---- the frame loop: only while the overview is shown and the hero is on screen; the slow turn alone draws at 30 fps */
  function tick(ms) {
    requestAnimationFrame(tick);
    const now = ms / 1000, dt = Math.min(0.05, now - (st.last || now)); st.last = now;
    if (sec.hidden || document.hidden || !st.on || !OV.fr) return;
    let busy = st.needs || !!st.drag;
    for (const k of ['pos', 'col', 'born']) if (OV.dirty[k]) { const a = k === 'pos' ? A.pos : k === 'born' ? A.born : null; if (a) a.needsUpdate = true; else { A.c0.needsUpdate = true; A.c1.needsUpdate = true; } OV.dirty[k] = false; busy = true; }
    if (!cur.c) { cur.c = OV.fr.c.slice(); cur.k = OV.fr.k; fit(); }
    const f = Math.min(1, dt * 4);
    let moving = Math.abs(OV.fr.k - cur.k) > 1e-6 * cur.k;
    for (let j = 0; j < 3; j++) { if (Math.abs(OV.fr.c[j] - cur.c[j]) > 0.01) moving = true; cur.c[j] += (OV.fr.c[j] - cur.c[j]) * f; }
    cur.k += (OV.fr.k - cur.k) * f;
    U.uC.value.set(cur.c[0], cur.c[1], cur.c[2]); U.uK.value = cur.k;
    if (OV.mix < 1) { if (OV.mixT0 == null) OV.mixT0 = now; OV.mix = Math.min(1, (now - OV.mixT0) / 1.1); busy = true; }
    U.uMix.value = ovEase(OV.mix);
    if (!st.drag) { st.idle += dt; st.vel += ((st.idle > 1.6 ? SPIN : 0) - st.vel) * Math.min(1, dt * 1.3); st.yaw += st.vel * dt; }
    const g = Math.min(1, dt * 3); if (Math.abs(st.dT - st.d) > 0.01 || Math.abs(st.tyT - st.ty) > 0.01) moving = true;
    st.d += (st.dT - st.d) * g; st.ty += (st.tyT - st.ty) * g;
    busy = busy || moving || Math.abs(st.vel - SPIN) > 1e-3 || ovNow() < OV.lastBirth + 0.8;
    if (!busy && (SPIN === 0 || now - st.lastR < 1 / 31)) return;
    st.lastR = now; st.needs = false;
    pts.rotation.y = st.yaw;
    camera.position.set(0, st.ty + st.d * Math.sin(st.el), st.d * Math.cos(st.el)); camera.lookAt(0, st.ty, 0);
    const R = OV.ext ? OV.ext.r : 10; U.uNear.value = st.d - R * 0.2; U.uFar.value = st.d + R * 1.3;
    U.uTime.value = ovNow();
    renderer.render(scene, camera); placeLabels();
  }

  new ResizeObserver(resize).observe(box);
  const cp = $('.ov-copy'); if (cp) new ResizeObserver(() => { measure(); fit(); applyView(); }).observe(cp);
  if ('IntersectionObserver' in window) new IntersectionObserver((es) => { st.on = es[es.length - 1].isIntersecting; st.needs = true; }).observe(box);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { labW.fill(0); measure(); st.needs = true; });
  OV.gl = {
    refit: () => { fit(); },
    labels: () => { makeLabels(); paint(); st.needs = true; },
    paint: () => { paint(); st.needs = true; },
    resize: () => { resize(); },
  };
  resize();
  if (OV.anc) makeLabels();
  requestAnimationFrame(tick);
}

/* ---- "Find a term": instant suggestions over the terms' names (and their other spellings), most mentions first */
function ovSearch(leads) {
  const inp = H('input', { type: 'search', id: 'ov-q', class: 'ov-q', placeholder: `Find a term: ${leads.join(', ')}…`, autocomplete: 'off', spellcheck: 'false',
    role: 'combobox', 'aria-expanded': 'false', 'aria-controls': 'ov-sugg', 'aria-autocomplete': 'list', 'aria-label': 'Find a term' });
  const list = H('ul', { class: 'ov-sugg', id: 'ov-sugg', role: 'listbox', 'aria-label': 'Terms', hidden: '' });
  let act = -1, res = [], closeT = 0;
  const close = () => { list.hidden = true; inp.setAttribute('aria-expanded', 'false'); inp.removeAttribute('aria-activedescendant'); act = -1; };
  const choose = (g) => { close(); inp.blur(); ovOpenConcept(g)(); };
  const hl = (name, q) => { const p = q ? name.toLowerCase().indexOf(q) : -1; return p < 0 ? [name] : [name.slice(0, p), H('mark', { text: name.slice(p, p + q.length) }), name.slice(p + q.length)]; };
  function draw() {
    clearTimeout(closeT); list.textContent = ''; act = -1; inp.removeAttribute('aria-activedescendant');
    const q = inp.value.trim().toLowerCase();
    if (!D.ready) { res = []; list.append(H('li', { class: 'h', role: 'presentation', text: `Loading the terms… ${Math.round(100 * OV.pct)}%` })); }
    else {
      res = q ? ovMatch(q) : OV.top.map((g) => [g]);
      list.append(H('li', { class: 'h', role: 'presentation' }, H('span', { text: q ? (res.length ? 'Terms' : 'No term matches') : 'Most defined terms' }), res.length ? H('span', { text: 'mentions' }) : null));
      res.forEach(([g, via], k) => {
        const li = H('li', { role: 'option', id: `ov-sg-${k}`, 'aria-selected': 'false' }, H('span', { class: 't' }, ...hl(D.g.name[g], q)), via ? H('span', { class: 'v', text: `as “${via}”` }) : null, H('span', { class: 'n', text: nf(D.g.n[g]) }));
        li.addEventListener('mousedown', (e) => e.preventDefault());
        li.addEventListener('click', () => choose(g));
        list.append(li);
      });
    }
    list.hidden = false; inp.setAttribute('aria-expanded', 'true');
  }
  function move(d) {
    if (!res.length) return;
    act = (act + d + res.length) % res.length;
    $$('[role=option]', list).forEach((li, k) => li.setAttribute('aria-selected', String(k === act)));
    const li = $(`#ov-sg-${act}`); inp.setAttribute('aria-activedescendant', li.id); li.scrollIntoView({ block: 'nearest' });
  }
  inp.addEventListener('input', draw); inp.addEventListener('focus', draw);
  inp.addEventListener('blur', () => { closeT = setTimeout(close, 200); });
  inp.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); if (list.hidden) draw(); move(1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); move(-1); }
    else if (e.key === 'Enter') { e.preventDefault(); const r = res[act >= 0 ? act : 0]; if (r) choose(r[0]); }
    else if (e.key === 'Escape' && !list.hidden) { e.stopPropagation(); close(); }
  });
  OV.search = { refresh: () => { if (document.activeElement === inp) draw(); }, progress: () => { if (!D.ready && !list.hidden) draw(); } };
  return H('div', { class: 'ov-find' }, H('span', { class: 'ov-find-i', 'aria-hidden': 'true' }), inp, list);
}
// a row of terms to start from, as on the Concepts tab; before the data is in, a click puts the term into the search box
function ovTry() {
  const open = (s) => () => {
    const g = D.ready ? D.g.name.findIndex((n) => n.toLowerCase() === s) : -1;
    if (g >= 0) return ovOpenConcept(g)();
    const q = $('#ov-q'); q.value = s; q.focus(); q.dispatchEvent(new Event('input'));
  };
  return H('div', { class: 'ov-try' }, H('span', { class: 'ov-try-l', text: 'Try' }),
    ['media bias', 'fake news', 'framing', 'echo chambers', 'confirmation bias'].map((s) => H('button', { type: 'button', class: 'ov-try-b', text: s, onclick: open(s) })));
}
function ovSearchReady() {
  const G = D.g, nG = G.n.length;
  OV.lname = G.name.map((s) => s.toLowerCase());
  OV.spell = new Map();   // a term's other spellings, from the spelling traces
  for (const gk in D.merges) { const g = +gk, sp = D.merges[gk].labels.map((l) => l[0]).filter((l) => l.toLowerCase() !== OV.lname[g]); if (sp.length) OV.spell.set(g, sp); }
  OV.top = Array.from({ length: nG }, (_, g) => g).sort((a, b) => G.n[b] - G.n[a]).slice(0, 8);
  if (OV.search) OV.search.refresh();
}
// ranks: the name itself; the name or one of its words starts with it; the name holds it; another spelling holds it. Within a rank, most mentions first
function ovMatch(q) {
  const G = D.g, L = OV.lname, K = 8, B = [[], [], [], []];
  const add = (b, g, via) => { const A = B[b]; if (A.length === K && G.n[g] <= G.n[A[K - 1][0]]) return; A.push([g, via]); A.sort((x, y) => G.n[y[0]] - G.n[x[0]]); if (A.length > K) A.pop(); };
  for (let g = 0; g < L.length; g++) {
    const s = L[g], p = s.indexOf(q);
    if (p === 0) add(s.length === q.length ? 0 : 1, g);
    else if (p > 0) add(/[\s\-(/'"“]/.test(s[p - 1]) ? 1 : 2, g);
    else { const sp = OV.spell.get(g); const hit = sp && sp.find((x) => x.toLowerCase().includes(q)); if (hit) add(3, g, hit); }
  }
  return B.flat().slice(0, K);
}

/* ================================================================== 2. what you can do: one card per view */
const OV_CARDS = [
  { id: 'spell', tab: 'concepts', go: 'Concepts', title: 'See how spellings merge', line: 'Every spelling of a term becomes one entry.', alt: 'The spellings of one term joining into one entry' },
  { id: 'rel', tab: 'relevance', go: 'Relevance', title: 'Relevance models in 3D', line: 'How three models decide what MBDef keeps.', alt: 'Every mention placed by its three model scores; the strict ones also pass the stricter cut' },
  { id: 'agree', tab: 'relations', go: 'Relations', title: 'See where papers agree', line: 'Links stated from both ends, and where papers contradict.', alt: 'How many links papers state from both ends' },
  { id: 'browse', tab: 'concepts', go: 'Concepts', title: 'Browse every term', line: 'Each term with its definitions, papers and neighbours.', alt: 'The terms that most papers define' },
  { id: 'papers', tab: 'papers', go: 'Papers', title: 'Start from a paper', line: 'Every paper with the terms it defines.', alt: 'How many terms each paper defines' },
];
function ovCardsShell() {
  const sec = $('#ov-do-sec'); sec.textContent = '';
  const grid = H('div', { class: 'ov-do' }); OV.vis = {}; OV.cardG = {};
  for (const cd of OV_CARDS) {
    const vis = H('div', { class: 'ov-vis wait', role: 'img', 'aria-label': cd.alt });
    const b = H('button', { type: 'button', class: 'ov-card' }, vis, H('span', { class: 'ov-ct', text: cd.title }), H('span', { class: 'ov-cl', text: cd.line }), H('span', { class: 'ov-cgo', text: `${cd.go} →` }));
    b.addEventListener('click', () => {
      const g = OV.cardG[cd.id]; show(cd.tab); scrollTo({ top: 0 });
      if (g == null || g < 0 || !D.ready) return;
      if (cd.id === 'spell' && TABS.concepts && TABS.concepts.open) TABS.concepts.open(g);
    });
    OV.vis[cd.id] = vis; grid.append(b);
  }
  sec.append(secHead('What you can do', null), grid);
}
function ovCardsFill() {
  const draw = (id, fn) => { const el = OV.vis[id]; el.classList.remove('wait'); el.textContent = ''; try { fn(el); } catch (e) { console.error(e); el.append(H('p', { class: 'ov-visnote', text: 'This picture could not be drawn.' })); } };
  const gs = ovSpellPick(); OV.cardG.spell = gs;
  draw('spell', (el) => ovVisSpell(el, gs));
  draw('rel', ovVisRel);
  draw('browse', ovVisBrowse);
  draw('papers', ovVisPapers);
  let asked = false;   // the picture that needs the links loads when the cards come near the screen
  const fill = () => { if (asked) return; asked = true; relations().then((rel) => draw('agree', (el) => ovVisAgree(el, ovRel(rel)))).catch((e) => { console.error(e); draw('agree', () => { throw e; }); }); };
  if ('IntersectionObserver' in window) { const io = new IntersectionObserver((es) => { if (es.some((x) => x.isIntersecting)) { io.disconnect(); fill(); } }, { rootMargin: '300px' }); io.observe($('#ov-do-sec')); }
  else fill();
}
// the spelling example: of the terms with 100 or more mentions, the one with the most spellings; ties: its most common spelling covers the smallest share
function ovSpellPick() {
  let best = -1, bk = null;
  for (const gk in D.merges) {
    const g = +gk, n = D.g.n[g]; if (n < 100) continue;
    const L = D.merges[gk].labels, k = [L.length, -Math.max(...L.map((l) => l[4])) / n];
    if (!bk || k[0] > bk[0] || (k[0] === bk[0] && k[1] > bk[1])) { best = g; bk = k; }
  }
  return best >= 0 ? best : D.g.n.reduce((b, n, g) => (n > D.g.n[b] ? g : b), 0);
}
const OV_W = 400, OV_H = 150;   // every card picture is drawn in this box and scaled to the card
const ovSvg = (el) => { const s = S('svg', { viewBox: `0 0 ${OV_W} ${OV_H}`, 'aria-hidden': 'true' }); el.append(s); return s; };
function ovVisSpell(el, g) {
  const L = ovSpell(g), n = Math.min(6, L.length), svg = ovSvg(el), cx = 252, cy = OV_H / 2, top = 15, dy = n > 1 ? (OV_H - top - 34) / (n - 1) : 0;
  for (let k = 0; k < n; k++) {
    const y = top + k * dy;
    T(svg, 14, y + 4, trunc(L[k][0], 20), 'vt');
    T(svg, 150, y + 4, `${L[k][1]}`, 'vm', 'end');
    S('path', { d: `M158,${y} C210,${y} 214,${cy} ${cx - 8},${cy}`, class: 'vline' }, svg);
  }
  if (L.length > n) T(svg, 14, OV_H - 8, `+ ${L.length - n} more spelling${L.length - n > 1 ? 's' : ''}`, 'vm');
  S('rect', { x: cx - 8, y: cy - 34, width: OV_W - cx - 6, height: 68, rx: 8, class: 'vnode' }, svg);
  T(svg, cx + 6, cy - 9, trunc(D.g.name[g], 16), 'vs');
  T(svg, cx + 6, cy + 10, `${L.length} spellings`, 'vm'); T(svg, cx + 6, cy + 24, `${nf(D.g.n[g])} mentions`, 'vm');
}
function ovVisRel(el) {
  const dpr = Math.min(devicePixelRatio || 1, 2), cv = H('canvas', { width: OV_W * dpr, height: OV_H * dpr }); el.append(cv);
  const x = cv.getContext('2d'); x.scale(dpr, dpr);
  const yaw = -0.72, pitch = 0.34, sc = 88, N = D.m.n, cs = rgbOf(css('--stage-strict')), ck = rgbOf(css('--stage-kept'));
  const P = (a, b, c) => { // classifier → x, basal → up, judge → depth; the unit cube, turned and tilted
    const px = a - 0.5, py = b - 0.5, pz = c - 0.5, x1 = px * Math.cos(yaw) + pz * Math.sin(yaw), z1 = -px * Math.sin(yaw) + pz * Math.cos(yaw);
    return [OV_W / 2 - 18 + x1 * sc * 1.3, OV_H / 2 + 12 - (py * Math.cos(pitch) - z1 * Math.sin(pitch)) * sc, py * Math.sin(pitch) + z1 * Math.cos(pitch)];
  };
  const ink = css('--stage-muted'), rule = css('--stage-rule');
  x.strokeStyle = rule; x.lineWidth = 1;
  for (const [a, b] of [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 1, 0]], [[1, 0, 0], [1, 0, 1]], [[0, 1, 0], [1, 1, 0]], [[0, 1, 0], [0, 1, 1]], [[0, 0, 1], [1, 0, 1]], [[0, 0, 1], [0, 1, 1]], [[1, 1, 0], [1, 1, 1]], [[1, 0, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]]) {
    const p = P(...a), q = P(...b); x.beginPath(); x.moveTo(p[0], p[1]); x.lineTo(q[0], q[1]); x.stroke();
  }
  const step = Math.max(1, Math.floor(N / 6000)), dots = [];
  for (let i = 0; i < N; i += step) { const p = P(D.score.c[i], D.score.b[i], D.score.j[i]); dots.push([p[2], p[0], p[1], D.strict[i]]); }
  dots.sort((a, b) => a[0] - b[0]);
  for (const [, px, py, s] of dots) { const c = s ? cs : ck; x.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},0.72)`; x.fillRect(px - 0.9, py - 0.9, 1.8, 1.8); }
  x.fillStyle = ink; x.font = `10.5px ${css('--f-mono')}`;
  const lab = (t, p, al) => { x.textAlign = al; x.fillText(t, p[0], p[1]); };
  lab('classifier', P(1.05, -0.08, 0), 'left'); lab('basal', P(0, 1.1, 0), 'center'); lab('judge', P(-0.06, -0.1, 1.04), 'right');
  const key = (c, t, yy) => { x.fillStyle = rgbStr(c); x.fillRect(OV_W - 104, yy - 7, 8, 8); x.fillStyle = ink; x.textAlign = 'left'; x.fillText(t, OV_W - 91, yy); };
  key(cs, 'strict', 18); key(ck, 'not strict', 34);
}
function ovVisBrowse(el) {
  const G = D.g, top = Array.from(G.n.keys()).sort((a, b) => G.n[b] - G.n[a]).slice(0, 5), mx = G.n[top[0]];
  el.append(H('ol', { class: 'ov-vbars' }, top.map((g) => H('li', {}, H('span', { class: 't', text: G.name[g] }), H('span', { class: 'b' }, H('i', { style: `width:${(100 * G.n[g]) / mx}%` })), H('span', { class: 'n', text: nf(G.n[g]) })))),
    H('p', { class: 'ov-vfoot', text: 'mentions: the papers that define each term' }));
}
function ovVisPapers(el) {
  const bands = [[1, 1], [2, 4], [5, 9], [10, 19], [20, 49], [50, 1e7]], P = D.p.id.length, seed = bands.map(() => 0), cited = bands.map(() => 0);
  for (let p = 0; p < P; p++) { const b = bands.findIndex(([a, z]) => D.p.n[p] >= a && D.p.n[p] <= z); if (b >= 0) (D.p.corpus[p] === 'seeds' ? seed : cited)[b]++; }
  const svg = ovSvg(el), mx = Math.max(...bands.map((_, k) => seed[k] + cited[k])), x0 = 26, bw = (OV_W - x0 - 20) / bands.length, y = lin(0, mx, OV_H - 26, 46);
  bands.forEach((b, k) => {
    const x = x0 + k * bw + bw * 0.18, w = bw * 0.64;
    S('rect', { x, y: y(seed[k]), width: w, height: OV_H - 26 - y(seed[k]), rx: 2, class: 'va' }, svg);
    S('rect', { x, y: y(seed[k] + cited[k]), width: w, height: y(seed[k]) - y(seed[k] + cited[k]) - 1, rx: 2, class: 'vb' }, svg);
    T(svg, x + w / 2, y(seed[k] + cited[k]) - 5, nf(seed[k] + cited[k]), 'vm', 'middle');
    T(svg, x + w / 2, OV_H - 11, bandLabel(b), 'vm', 'middle');
  });
  const ns = seed.reduce((a, b) => a + b, 0);
  S('rect', { x: x0, y: 9, width: 9, height: 9, rx: 2, class: 'va' }, svg); T(svg, x0 + 14, 17, `seed ${nf(ns)}`, 'vt');
  S('rect', { x: x0 + 100, y: 9, width: 9, height: 9, rx: 2, class: 'vb' }, svg); T(svg, x0 + 114, 17, `cited ${nf(P - ns)}`, 'vt');
  T(svg, OV_W - 10, 17, 'papers by terms defined', 'vm', 'end');
}
// links, pairs and the two link examples, from relations(): [from, to, field, papers], one per (term, named term, field)
function ovRel(rel) {
  if (OV.R) return OV.R;
  const fi = rel.fields, G = D.g.n.length, ROLE = ['sp', 'sb', 'sy', 'rt'], pair = new Map(), best = {};
  for (const e of rel.edges) {
    const k = e[0] * G + e[1]; let v = pair.get(k); if (!v) pair.set(k, (v = [0, 0, 0, 0])); v[ROLE.indexOf(fi[e[2]])] = e[3];
    const c = fi[e[2]], s = e[3] * 1e6 + D.g.n[e[0]] + D.g.n[e[1]]; if (!best[c] || s > best[c][1]) best[c] = [e, s];   // most stated; ties: the bigger terms
  }
  // linked from both ends: a's papers call b a supertype and b's papers call a a subtype (agree), or each calls the other the same
  let agree = 0, msp = 0, msb = 0, ex = null, exS = -1;
  for (const [k, v] of pair) {
    const a = Math.floor(k / G), b = k % G, w = pair.get(b * G + a); if (!w) continue;
    if (v[0] && w[1]) { agree++; const s = Math.min(v[0], w[1]) * 1e6 + v[0] + w[1]; if (s > exS) { exS = s; ex = [a, b, v[0], w[1]]; } }   // the example: both ends most stated
    if (a < b && v[0] && w[0]) msp++;
    if (a < b && v[1] && w[1]) msb++;
  }
  return (OV.R = { rel, pair, best, agree, msp, msb, ex });
}
function ovVisAgree(el, R) {
  const nm = (g) => H('b', { text: trunc(D.g.name[g], 22) });
  el.append(H('div', { class: 'ov-agree' },
    H('span', { class: 'ov-big num', text: nf(R.agree) }),
    H('span', { class: 'ov-ag-l', text: 'links stated from both ends, and both ends agree' }),
    R.ex ? H('span', { class: 'ov-ag-ex' }, nm(R.ex[0]), ' is a kind of ', nm(R.ex[1]), `, say ${nf(R.ex[2])} + ${nf(R.ex[3])} papers from the two ends`) : null,
    H('span', { class: 'ov-ag-c' }, H('i'), `${nf(R.msp + R.msb)} pairs contradict: each calls the other broader, or narrower`)));
}

/* ================================================================== 3. the dataset in numbers */
function ovStats() {
  const N = D.m.n, G = D.g.n.length, P = D.p.id.length, strictIn = new Uint32Array(G);
  let strictM = 0, defs = 0, oneDef = 0, single = 0, strictC = 0, seeds = 0;
  for (let i = 0; i < N; i++) { strictM += D.strict[i]; defs += D.m.ndef[i]; if (D.m.ndef[i] === 1) oneDef++; if (D.strict[i]) strictIn[D.m.group[i]]++; }
  for (let g = 0; g < G; g++) { if (D.g.n[g] === 1) single++; if (strictIn[g]) strictC++; }
  for (let p = 0; p < P; p++) if (D.p.corpus[p] === 'seeds') seeds++;
  return { N, P, concepts: G, strictM, defs, oneDef, strictIn, single, strictC, seeds };
}
// the worked examples: "fake news" (else the term with the most spellings), the not-strict term with the most mentions,
// the single-mention term with the most definitions
function ovPicks(st) {
  const G = D.g, nG = G.n.length; let fk = -1, ns = -1, sg = -1;
  for (let g = 0; g < nG; g++) {
    const n = G.n[g];
    if (G.name[g].toLowerCase() === 'fake news') fk = g;
    if (!st.strictIn[g] && (ns < 0 || n > G.n[ns])) ns = g;
    if (n === 1 && (sg < 0 || G.ndef[g] > G.ndef[sg])) sg = g;
  }
  if (fk < 0) for (let g = 0; g < nG; g++) if (fk < 0 || Object.keys(G.labels[g]).length * 1e6 + G.n[g] > Object.keys(G.labels[fk]).length * 1e6 + G.n[fk]) fk = g;
  return { fk, ns, sg };
}
const ovMed = (a) => { const s = Array.from(a).sort((x, y) => x - y); return s[Math.floor((s.length - 1) / 2)]; };
function ovNumbers(st) {
  const sec = $('#ov-num-sec'), pk = ovPicks(st), M = D.meta, fc = M.fields.counts, bl = M.baseline;
  const items = FIELDS.reduce((s, f) => s + fc[f[0]][0], 0), mb = (...k) => H('span', {}, ...k.map((x) => (typeof x === 'object' ? x : x)));
  const fig = (v, label, ...sub) => H('div', { class: 'ov-num' }, H('b', { class: 'num', text: v }), H('span', { class: 'l' }, label), sub.length ? H('span', { class: 's' }, ...sub) : null);
  const nums = H('div', { class: 'ov-nums' },
    fig(nf(st.N), gl('mention', 'term mentions'), 'one term in one paper'),
    fig(nf(st.strictM), gl('strict', 'strict mentions'), `${pc(st.strictM / st.N)} of the mentions`),
    fig(nf(st.strictC), gl('strict', 'strict terms'), `of the ${nf(st.concepts)} terms`),
    fig(nf(st.single), gl('single', 'single-mention terms'), `${pc(st.single / st.concepts)}: one paper each`),
    fig(f2(st.defs / st.N), gl('definition', 'definitions per mention'), `${pc(st.oneDef / st.N)} have one`),
    fig(nf(st.seeds), gl('seed', 'seed papers'), 'the starting collection'),
    fig(nf(st.P - st.seeds), gl('closure', 'cited papers'), 'that seed papers cite'),
    fig(nf(M.counts.dropped), gl('cut', 'mentions left out'), 'below the relevance cut'),
    fig(nf(M.fields.relations), gl('link', 'links between terms'), 'named by the papers'),
    fig(f2(bl.nearest_neighbour_cos_mean), gl('neighbours', 'nearest-mention similarity'), `${f2(bl.random_pair_cos_mean)} for two random mentions`));
  const N = st.N, cut = M.cut;
  const xps = H('div', { class: 'xps' },
    expander({ title: 'What the numbers mean', meta: mb(H('b', { text: nf(N) }), ' mentions → ', H('b', { text: nf(st.concepts) }), ' terms'), build: (b) => ovExplain(b, st, pk) }),
    expander({ title: 'Mentions per term', meta: mb(H('b', { text: pc(st.single / st.concepts) }), ' from one paper'), build: (b) => ovChartTerms(b, st) }),
    expander({ title: 'Definitions per mention', meta: mb(H('b', { text: pc(st.oneDef / N) }), ' have one'), build: (b) => ovChartDefs(b, st) }),
    expander({ title: 'Relevance scores', meta: mb('cut ', H('b', { text: f3(cut) }), ` · ${pc(st.strictM / N)} strict`), build: (b) => ovChartScores(b, st, pk) }),
    expander({ title: 'Terms per paper', meta: mb('median ', H('b', { text: nf(ovMed(D.p.n)) })), build: (b) => ovChartPapers(b) }),
    expander({ title: 'What else the papers say', meta: mb(H('b', { text: nf(items) }), ' items, 6 fields'), build: (b) => ovFieldsX(b) }));
  sec.textContent = '';
  sec.append(secHead('The dataset in numbers', null), nums, xps, H('p', { class: 'small muted', text: 'Dotted words: hover for the meaning, click for an example.' }));
}

/* ---- "What the numbers mean": the flow, the splits, three worked examples */
function ovExplain(box, st, pk) {
  const G = D.g, cut = D.meta.cut, bf = D.meta.best_f1_cut;
  const node = (cls, v, key, label, d, extra) => H('div', { class: 'ovf-n ' + cls }, H('b', { class: 'ovf-v num', text: nf(v) }), H('span', { class: 'ovf-l' }, gl(key, label)), H('span', { class: 'ovf-d', text: d }), extra || null);
  const arrow = (...kids) => H('div', { class: 'ovf-a' }, H('span', { class: 'ovf-at' }, ...kids), H('span', { class: 'ovf-ar', 'aria-hidden': 'true' }));
  box.append(H('p', { class: 'cap', text: 'Each term a paper defines is one mention; mentions with the same label merge into one term. Mention counts are before merging, term counts after.' }),
    H('div', { class: 'ovf', role: 'group', 'aria-label': 'From papers to term mentions to terms' },
      node('ovf-m', st.P, 'paper', 'papers', `${nf(st.seeds)} seed, ${nf(st.P - st.seeds)} cited`),
      arrow('score reaches the ', gl('cut', 'cut'), ` ${cut.toFixed(3)}`),
      node('ovf-k', st.N, 'mention', 'term mentions', 'one term in one paper, with all its definitions there'),
      arrow(gl('labelgroup', 'merged by label')),
      node('ovf-c', st.concepts, 'concept', 'terms', `${nf(st.single)} hold a single mention`,
        H('div', { class: 'ovf-s' }, H('b', { class: 'num', text: nf(st.strictC) }), ' ', gl('strict', 'strict'), H('span', { class: 'ovf-d', text: `a mention past the best-F1 cut ${bf.toFixed(3)}` })))));
  const split = (title, a, b, la, lb, ca, cb) => H('div', { class: 'ovs' }, H('div', { class: 'ovs-h' }, ...title),
    H('div', { class: 'ovs-bar', role: 'img', 'aria-label': `${la}: ${nf(a)}; ${lb}: ${nf(b)}` }, H('i', { class: ca, style: `flex:${a}` }), H('i', { class: cb, style: `flex:${b}` })),
    H('div', { class: 'ovs-k' }, H('span', {}, H('i', { class: ca }), `${la} ${nf(a)} (${pc(a / (a + b))})`), H('span', {}, H('i', { class: cb }), `${lb} ${nf(b)}`)));
  box.append(H('div', { class: 'ovs-wrap' },
    split([gl('mention', 'Term mentions')], st.strictM, st.N - st.strictM, 'strict', 'not strict', 'c-strict', 'c-plain'),
    split([gl('concept', 'Terms')], st.strictC, st.concepts - st.strictC, 'strict', 'not strict', 'c-strict', 'c-plain'),
    split([gl('paper', 'Papers')], st.seeds, st.P - st.seeds, 'seed', 'cited', 'c-con', 'c-plain')));
  const ex = (title, ...body) => H('div', { class: 'ov-case' }, H('h4', { text: title }), ...body);
  const fk = pk.fk, sp = ovSpell(fk), fkS = D.members[fk].reduce((s, i) => s + D.strict[i], 0);
  box.append(H('h4', { class: 'ov-cases-h', text: 'Worked examples' }), H('div', { class: 'ov-cases' },
    ex(`One term, many mentions: ${ovQ(G.name[fk])}`,
      H('p', { text: `${nf(G.n[fk])} mentions in ${nf(G.papers[fk])} papers (${nf(fkS)} strict), written ${sp.length} ways. They merge into one term: +${nf(G.n[fk])} mentions, +1 term.` }),
      H('p', { class: 'ov-spell' }, sp.map(([l, c]) => H('span', { class: 'mono' }, `${l} `, H('span', { class: 'muted', text: `×${nf(c)}` })))),
      H('p', { class: 'ctl' }, ovBtn('Open this term', ovOpenConcept(fk)), ovBtn('Read its best-scored mention', () => openMention(ovMax(fk))))),
    pk.sg >= 0 ? ex(`One mention, many definitions: ${ovQ(G.name[pk.sg])}`,
      H('p', { text: `Only ${ovQ(ovShort(ovTitle(D.m.paper[D.members[pk.sg][0]]), 80))} defines it, ${nf(G.ndef[pk.sg])} times: +${nf(G.ndef[pk.sg])} definitions, +1 mention, +1 term. No other single-mention term has more definitions.` }),
      H('p', { class: 'ctl' }, ovBtn('Open this term', ovOpenConcept(pk.sg)), ovBtn('Read its definitions', () => openMention(D.members[pk.sg][0])))) : null,
    pk.ns >= 0 ? ex(`A term, but not a strict one: ${ovQ(G.name[pk.ns])}`,
      H('p', { text: `${nf(G.n[pk.ns])} mentions; the best scores ${f3(D.score.s[ovMax(pk.ns)])}: past the cut (${cut.toFixed(3)}), short of the best-F1 cut (${bf.toFixed(3)}). No term that is not strict has more mentions.` }),
      H('p', { class: 'ctl' }, ovBtn('Open this term', ovOpenConcept(pk.ns)))) : null));
}

/* ---- a bar chart on a log scale, one or two series per band; a band is a button: it picks its examples */
function ovBars(el, cats, series, opt) {
  const w = Math.max(300, el.clientWidth || 600), h = 220, lw = 46, top = 20, bot = 40, n = cats.length;
  const mx = Math.max(...series.flatMap((s) => s.values), 1), y = (v) => lin(0, Math.log10(mx) * 1.07 + 0.02, h - bot, top)(Math.log10(Math.max(v, 1)));
  const x = lin(-0.5, n - 0.5, lw, w - 6), slot = (w - lw - 6) / n, bw = Math.min(44, (slot * 0.76) / series.length), labs = bw >= 24;
  const svg = S('svg', { viewBox: `0 0 ${w} ${h}`, role: 'group', 'aria-label': opt.label }); el.textContent = ''; el.append(svg);
  for (let v = 1; v <= mx * 1.01; v *= 10) { S('line', { x1: lw, x2: w - 6, y1: y(v), y2: y(v), class: 'gl' }, svg); T(svg, lw - 6, y(v) + 4, ovK(v), 'tx-m', 'end'); }
  const groups = cats.map((c, i) => {
    const g = S('g', { class: 'ovb', tabindex: '0', role: 'button', 'aria-label': opt.tip(i) }, svg);
    S('rect', { x: x(i) - slot / 2, y: top - 6, width: slot, height: h - top - 8, style: 'fill:transparent' }, g);
    series.forEach((s, k) => {
      const v = s.values[i], bx = x(i) - (series.length * bw) / 2 + k * bw;
      S('rect', { x: bx + 1, y: y(v), width: bw - 2, height: Math.max(0, y(1) - y(v)), rx: 2, style: s.style }, g);
      if (labs || k === 0) T(g, labs ? bx + bw / 2 : x(i), y(labs ? v : Math.max(...series.map((q) => q.values[i]))) - 5, ovK(v), 'tx-m', 'middle');
    });
    T(g, x(i), h - bot + 15, c, 'tx-m', 'middle');
    S('title', {}, g).textContent = opt.tip(i);
    g.addEventListener('click', () => pick(i));
    g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(i); } });
    return g;
  });
  T(svg, (lw + w) / 2, h - 5, opt.xlabel, 'tx-m', 'middle');
  const pick = (i) => { groups.forEach((g, j) => { g.classList.toggle('dim', j !== i); g.setAttribute('aria-pressed', String(j === i)); }); opt.pick(i); };
  el.append(H('div', { class: 'ov-legend' }, series.map((s) => H('span', {}, H('i', { class: 'sw', style: s.style.replace('fill', 'background').replace('fill-opacity', 'opacity') }), s.name)), H('span', { class: 'muted', text: 'log scale: each line is 10× the one below' })));
  return pick;
}
const OV_S1 = 'fill:var(--accent)', OV_S2 = 'fill:var(--accent);fill-opacity:.35';
const inBand = (v, [a, b]) => v >= a && v <= b;
function ovChartTerms(body, st) {
  const G = D.g, nG = G.n.length, N = D.m.n, chart = H('div', { class: 'chart' }), exEl = H('p', { class: 'cap ov-ex', 'aria-live': 'polite' });
  const big = Array.from(G.n.keys()).sort((a, b) => G.n[b] - G.n[a]).slice(0, 3);
  body.append(H('p', { class: 'cap', text: `${pc(st.single / st.concepts)} of the terms come from one paper, yet they hold only ${pc(st.single / N)} of the mentions; the largest: ${big.map((g) => `${ovQ(G.name[g])} (${nf(G.n[g])})`).join(', ')}.` }), chart, exEl);
  const cb = chart.clientWidth < 480 ? [[1, 1], [2, 2], [3, 3], [4, 4], [5, 9], [10, 49], [50, 1e7]] : [[1, 1], [2, 2], [3, 3], [4, 4], [5, 9], [10, 19], [20, 49], [50, 1e7]];
  const cnt = cb.map(() => 0), held = cb.map(() => 0);
  for (let g = 0; g < nG; g++) { const b = cb.findIndex((x) => inBand(G.n[g], x)); cnt[b]++; held[b] += G.n[g]; }
  const pick = ovBars(chart, cb.map(bandLabel), [{ name: 'terms', values: cnt, style: OV_S1 }, { name: 'their mentions', values: held, style: OV_S2 }], {
    label: 'Terms by how many mentions they have, and the mentions they hold', xlabel: 'mentions per term',
    tip: (i) => `${nf(cnt[i])} terms with ${bandLabel(cb[i])} mentions (${pc(cnt[i] / nG)}); they hold ${nf(held[i])} mentions (${pc(held[i] / N)})`,
    pick: (i) => {
      const gs = []; for (let g = 0; g < nG; g++) if (inBand(G.n[g], cb[i])) gs.push(g);
      gs.sort((a, b) => G.n[b] - G.n[a] || G.ndef[b] - G.ndef[a]);
      exEl.textContent = '';
      exEl.append(H('b', { text: `${nf(cnt[i])} terms with ${bandLabel(cb[i])} mention${cb[i][1] === 1 ? '' : 's'}` }), ', e.g. ', ...gs.slice(0, 5).flatMap((g, j) => [j ? ', ' : '', ovBtn(G.name[g], ovOpenConcept(g)), ` (${nf(G.n[g])})`]), '.');
    } });
  pick(0);
}
function ovChartDefs(body, st) {
  const N = D.m.n, chart = H('div', { class: 'chart' }), exEl = H('p', { class: 'cap ov-ex', 'aria-live': 'polite' });
  body.append(H('p', { class: 'cap', text: `${pc(st.oneDef / N)} of the mentions have one definition. A paper that defines a term more than once gives one mention with all of them, and relevance is decided for the mention as a whole.` }), chart, exEl);
  const db = [[1, 1], [2, 2], [3, 3], [4, 5], [6, 9], [10, 1e7]], cnt = db.map(() => 0), held = db.map(() => 0);
  for (let i = 0; i < N; i++) { const b = db.findIndex((x) => inBand(D.m.ndef[i], x)); cnt[b]++; held[b] += D.m.ndef[i]; }
  const pick = ovBars(chart, db.map(bandLabel), [{ name: 'mentions', values: cnt, style: OV_S1 }, { name: 'their definitions', values: held, style: OV_S2 }], {
    label: 'Mentions by how many definitions they have, and the definitions they hold', xlabel: 'definitions per mention',
    tip: (j) => `${nf(cnt[j])} mentions with ${bandLabel(db[j])} definitions (${pc(cnt[j] / N)}); they hold ${nf(held[j])} definitions (${pc(held[j] / st.defs)})`,
    pick: (j) => {
      const ms = []; for (let i = 0; i < N; i++) if (inBand(D.m.ndef[i], db[j])) ms.push(i);
      ms.sort((a, b) => D.m.ndef[b] - D.m.ndef[a] || D.score.s[b] - D.score.s[a]);
      exEl.textContent = '';
      exEl.append(H('b', { text: `${nf(cnt[j])} mentions with ${bandLabel(db[j])} definition${db[j][1] === 1 ? '' : 's'}` }), ', e.g. ', ...ms.slice(0, 3).flatMap((i, k) => [k ? '; ' : '', ovBtn(D.m.label[i], () => openMention(i)), ` in ${ovQ(ovShort(ovTitle(D.m.paper[i]), 54))} (${D.m.ndef[i]})`]), '.');
    } });
  pick(db.length - 1);
}
function ovChartPapers(body) {
  const P = D.p.id.length, chart = H('div', { class: 'chart' }), exEl = H('p', { class: 'cap ov-ex', 'aria-live': 'polite' });
  body.append(H('p', { class: 'cap', text: `Half of the ${nf(P)} papers define at most ${nf(ovMed(D.p.n))} terms with ${nf(ovMed(D.p.ndef))} definitions; a few long works define over a hundred.` }), chart, exEl);
  const pb = chart.clientWidth < 480 ? [[1, 1], [2, 4], [5, 9], [10, 19], [20, 49], [50, 1e7]] : [[1, 1], [2, 4], [5, 9], [10, 19], [20, 49], [50, 99], [100, 1e7]];
  const cnt = pb.map(() => 0), held = pb.map(() => 0);
  for (let p = 0; p < P; p++) { const b = pb.findIndex((x) => inBand(D.p.n[p], x)); cnt[b]++; held[b] += D.p.n[p]; }
  const pick = ovBars(chart, pb.map(bandLabel), [{ name: 'papers', values: cnt, style: OV_S1 }, { name: 'their mentions', values: held, style: OV_S2 }], {
    label: 'Papers by how many terms they define, and the mentions they hold', xlabel: 'term mentions per paper',
    tip: (j) => `${nf(cnt[j])} papers with ${bandLabel(pb[j])} term mentions (${pc(cnt[j] / P)}); they hold ${nf(held[j])} mentions (${pc(held[j] / D.m.n)})`,
    pick: (j) => {
      const ps = []; for (let p = 0; p < P; p++) if (inBand(D.p.n[p], pb[j])) ps.push(p);
      ps.sort((a, b) => D.p.n[b] - D.p.n[a] || D.p.ndef[b] - D.p.ndef[a]);
      exEl.textContent = '';
      exEl.append(H('b', { text: `${nf(cnt[j])} papers with ${bandLabel(pb[j])} term mention${pb[j][1] === 1 ? '' : 's'}` }), ', e.g. ', ...ps.slice(0, 3).flatMap((p, k) => [k ? '; ' : '', ovBtn(ovShort(ovTitle(p), 70), () => { show('papers'); scrollTo({ top: 0 }); if (TABS.papers && TABS.papers.open) TABS.papers.open(p); }), ` (${nf(D.p.n[p])} mentions, ${nf(D.p.ndef[p])} definitions)`]), '.');
    } });
  pick(pb.length - 1);
}
// the score distributions: one curve per score, the cut and the best-F1 cut, a readout under the pointer
function ovChartScores(body, st, pk) {
  const N = D.m.n, chart = H('div', { class: 'chart' }), rd = H('p', { class: 'small muted ov-read', 'aria-live': 'polite' }), exEl = H('p', { class: 'cap ov-ex' });
  body.append(H('p', { class: 'cap', text: `The share of the ${nf(N)} mentions scoring at most each value. The thick line, the ensemble, decides: MBDef holds only mentions past the cut, and the ${pc(st.strictM / N)} past the best-F1 cut are also strict (violet band).` }), chart, rd, exEl,
    more('Why combine three models', H('p', {}, 'The thin lines are the three models’ own scores (', gl('classifier', 'classifier'), ', ', gl('basal'), ', ', gl('judge', 'judge'), ') before they are combined. Each spreads its scores differently, which is why the ', gl('ensemble', 'ensemble'), ' combines their ranks, not their raw scores.')));
  const w = Math.max(300, chart.clientWidth || 600), h = 250, lw = 44, top = 22, bot = 34;
  const x = lin(0, 1, lw, w - 10), y = lin(0, 1, h - bot, top), cut = D.meta.cut, bf = D.meta.best_f1_cut;
  const svg = S('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': 'Share of term mentions scoring at most each value, for the ensemble and its three models' }); chart.append(svg);
  S('rect', { x: x(bf), y: top, width: x(1) - x(bf), height: h - bot - top, style: 'fill:var(--strict);opacity:.09' }, svg);
  for (const v of [0, 0.25, 0.5, 0.75, 1]) { S('line', { x1: lw, x2: w - 10, y1: y(v), y2: y(v), class: 'gl' }, svg); T(svg, lw - 6, y(v) + 4, pc(v, 0), 'tx-m', 'end'); }
  for (const v of [0, 0.2, 0.4, 0.6, 0.8, 1]) T(svg, x(v), h - bot + 15, f2(v), 'tx-m', 'middle');
  T(svg, (lw + w) / 2, h - 3, 'score', 'tx-m', 'middle');
  const STY = { s: { stroke: 'var(--accent)', 'stroke-width': 2.8 }, c: { stroke: 'var(--ink-2)', 'stroke-width': 1.4, 'stroke-dasharray': '6 3' }, b: { stroke: 'var(--ink-2)', 'stroke-width': 1.4, 'stroke-dasharray': '1.5 3' }, j: { stroke: 'var(--ink-2)', 'stroke-width': 1.4, 'stroke-dasharray': '9 3 2 3' } };
  for (const k of ['c', 'b', 'j', 's']) {
    const ord = D.order[k], pts = [];
    for (let q = 0; q <= 240; q++) { const idx = Math.min(N - 1, Math.floor((q / 240) * (N - 1))); pts.push([D.score[k][ord[idx]], (idx + 1) / N]); }
    S('path', Object.assign({ d: 'M' + pts.map(([a, b]) => `${x(a).toFixed(1)},${y(b).toFixed(1)}`).join('L'), fill: 'none', 'stroke-linejoin': 'round' }, STY[k]), svg);
  }
  S('line', { x1: x(cut), x2: x(cut), y1: top - 4, y2: h - bot, style: 'stroke:var(--ink)', 'stroke-dasharray': '4 3' }, svg);
  S('line', { x1: x(bf), x2: x(bf), y1: top - 4, y2: h - bot, style: 'stroke:var(--strict)', 'stroke-dasharray': '4 3' }, svg);
  T(svg, x(cut) - 4, top - 8, `cut ${cut.toFixed(3)}`, 'tx-m', 'end');
  T(svg, x(bf) + 4, top - 8, `best-F1 ${bf.toFixed(3)}`, 'tx-m', 'start');
  const hv = S('line', { y1: top, y2: h - bot, style: 'stroke:var(--muted);visibility:hidden' }, svg);
  const below = (k, v) => { const o = D.order[k], sc = D.score[k]; let a = 0, b = N; while (a < b) { const m = (a + b) >> 1; if (sc[o[m]] <= v) a = m + 1; else b = m; } return a; };
  const idle = 'Point at the chart, or tap it, to read the share of mentions at any score.';
  rd.textContent = idle;
  const read = (e) => {
    const r = svg.getBoundingClientRect(), v = Math.max(0, Math.min(1, ((e.clientX - r.left) * (w / r.width) - lw) / (w - 10 - lw)));
    hv.setAttribute('x1', x(v)); hv.setAttribute('x2', x(v)); hv.style.visibility = 'visible';
    const n = below('s', v), zone = v < cut ? 'below the cut' : v < bf ? 'kept, not strict' : 'kept and strict';
    rd.textContent = `Score ${f3(v)} (${zone}): ${pc(n / N)} of mentions (${nf(n)}) score at most this on the ensemble; classifier ${pc(below('c', v) / N)}, basal ${pc(below('b', v) / N)}, judge ${pc(below('j', v) / N)}.`;
  };
  svg.addEventListener('pointermove', read); svg.addEventListener('pointerdown', read);
  svg.addEventListener('pointerleave', () => { hv.style.visibility = 'hidden'; rd.textContent = idle; });
  const lg = H('div', { class: 'ov-legend' });
  for (const k of ['s', 'c', 'b', 'j']) {
    const sw = S('svg', { width: 26, height: 10, 'aria-hidden': 'true' }); S('line', Object.assign({ x1: 1, x2: 25, y1: 5, y2: 5 }, STY[k]), sw);
    lg.append(H('span', {}, sw, gl(k === 's' ? 'ensemble' : SC[k], k === 's' ? 'ensemble (decides)' : SC[k])));
  }
  lg.append(H('span', {}, H('i', { class: 'sw c-strict' }), gl('strict', 'strict')));
  chart.append(lg);
  // examples: the median-scored mention of the first worked example's term, and the lowest-scored mention of all
  const fkM = D.members[pk.fk].slice().sort((a, b) => D.score.s[a] - D.score.s[b]), mid = fkM[Math.floor(fkM.length / 2)], lo = D.order.s[0];
  const tag = (i) => (D.strict[i] ? 'strict' : 'not strict');
  exEl.append(H('b', { text: 'For example: ' }), ovBtn(D.m.label[mid], () => openMention(mid)), ` in ${ovQ(ovShort(ovTitle(D.m.paper[mid]), 54))} scores ${f3(D.score.s[mid])}, the median of its ${nf(fkM.length)} mentions (${tag(mid)}); `,
    ovBtn(D.m.label[lo], () => openMention(lo)), ` in ${ovQ(ovShort(ovTitle(D.m.paper[lo]), 54))} scores ${D.score.s[lo].toFixed(4)}, the lowest of all, at the cut (${tag(lo)}).`);
}
/* ---- what else the papers say: one tile per field, counted over every mention before merging; the most stated link of each name field */
function ovFieldsX(box) {
  const fc = D.meta.fields.counts, N = D.m.n, say = {}, tiles = H('div', { class: 'ovx-tiles' });
  box.append(H('p', { class: 'cap' }, 'Besides definitions, the extraction lists six more fields per term and paper. A name that is itself a term becomes a ', gl('link', 'link between terms'), '.'), tiles);
  for (const [c, label, sing, col] of FIELDS) {
    const [items, withAny] = fc[c];
    say[c] = H('span', { class: 'tb ovx-s' });
    tiles.append(H('div', { class: 'tile', style: `--fc:var(${col})` }, H('span', { class: 'tl' }, gl(sing.replace(' ', ''), label)), H('span', { class: 'tv num', text: nf(items) }), H('span', { class: 'tb', text: `in ${pc(withAny / N)} of the mentions` }), say[c]));
  }
  box.append(H('p', { class: 'cap' }, 'Where papers agree in ', ovBtn('Relations', ovGo('relations')), '.'));
  relations().then((rel) => {
    const R = ovRel(rel), q = (g) => ovBtn(D.g.name[g], ovOpenConcept(g));
    const say1 = { sp: (a, b) => [q(a), ' is a kind of ', q(b)], sb: (a, b) => [q(b), ' is a kind of ', q(a)], sy: (a, b) => [q(b), ' is another name for ', q(a)], rt: (a, b) => [q(b), ' relates to ', q(a)] };
    for (const c of NAMES) { if (!R.best[c]) continue; const [a, b, , n] = R.best[c][0]; say[c].append('Most stated: ', ...say1[c](a, b), ` (${nf(n)} paper${n === 1 ? '' : 's'})`); }
  }).catch((e) => console.error(e));
}

/* ================================================================== the tab */
function ovInit() {
  if (!OV.built) ovHero(D.meta);   // the page opened on another tab: the hero comes now, with every mention at once
  ovFromData();
  const st = ovStats();
  ovSearchReady();
  ovCardsFill();
  ovNumbers(st);
}
TABS.overview = { hero: ovHero, init: ovInit, onShow: () => { if (OV.gl) OV.gl.resize(); } };
