/* map tab: registers TABS.map. Uses the helpers of core.js (the colour helpers, mapFrame() and topicPalette() live there).
   The look is the poster's night sky, in both themes: one THREE.Points cloud drawn by a small shader (a round, softly
   shaded dot whose size shrinks with distance, clamped in px), in two passes (full dots with depth, faded dots blended on
   top), topic colours spaced in OKLab so that neighbouring topics differ, a faint fog towards the back, topic names at the
   topic centres, screen-space picking for the hover card, and camera flights that ease the target, swing the direction
   and scale the distance (rising a little on long trips). The frame is centred, with 98% of the dots within radius 10.
   Around it: a slim control rail (find a label, colour, view, dots) top left and the topic list top right; on a phone the
   rail is a bar at the bottom and the topics a sheet from the top. Under the stage, in expanders: how to read the map, a
   worked example and every topic in a table.
   Keys while the map is shown: / find · R reset · 2 / 3 view · N names · T topics · C controls · Esc clears. */
GL.embedding = GL.embedding || ['Embedding', 'A list of numbers that a language model computes for a text, so that texts with similar meaning get similar numbers.', 'Each mention\'s definitions are embedded with Qwen3-Embedding-8B and averaged into one vector. The similarity of two mentions is the cosine of their vectors: 1 means the same direction, 0 unrelated.'];
GL.umap = GL.umap || ['UMAP', 'A method that turns each mention\'s long embedding into 3 (or 2) map coordinates, keeping similar mentions close together.', 'Who sits next to whom is the reliable part of the map; long distances across it are only rough. The coordinates were computed once and are part of the page\'s data.'];

const MAP = { pending: -1, pendingColor: null, ready: false };

/* ------------------------------------------------------------------ small helpers */
const ICO = {
  sliders: '<path d="M2.5 4.5h11M2.5 11.5h11"/><circle cx="5.5" cy="4.5" r="1.9"/><circle cx="10.5" cy="11.5" r="1.9"/>',
  reset: '<path d="M3.1 8.6a5 5 0 1 0 1.6-4.3"/><path d="M4.4 1.6v3.1h3.1"/>',
};
function ico(n) { const s = S('svg', { viewBox: '0 0 16 16', width: 15, height: 15, 'aria-hidden': 'true', class: 'ico', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }); s.innerHTML = ICO[n]; return s; }
// the map's topic colours on a given ground: topicPalette's colour slots, on the lighter tier when the ground is dark
function mapPalette(F, dark) { const pal = tiered(F.NT, dark); return Array.from({ length: F.NT }, (_, t) => pal[F.slot[t]]); }
const stageIsDark = () => { const b = $('#mapbox'); return !b || okLab(rgbOf(getComputedStyle(b).getPropertyValue('--map-bg1')))[0] < 0.5; };
// swatches on the stage take the stage's colours; those under it (in the expanders) the reader's theme
function paintTopics(F) {
  const box = $('#mapbox'), on = mapPalette(F, stageIsDark()), off = topicPalette(F);
  for (const e of $$('[data-tc]', $('#v-map'))) e.style.setProperty('--lc', rgbStr((box.contains(e) ? on : off)[+e.dataset.tc]));
}
// the topics led by a bias concept: the one led by "media bias" first, then the others; the order of the topic list
function topicRanks(F) {
  const tps = D.meta.topics, lead = tps.map((tp) => (tp.top[0].toLowerCase() === 'media bias' ? 2 : /bias/i.test(tp.top[0]) ? 1 : 0));
  return { lead, ord: [...tps.keys()].sort((a, b) => lead[b] - lead[a] || F.size[b] - F.size[a]) };
}
function layoutStage() { // full-bleed width without the scrollbar, and the height left under the sticky header
  const box = $('#mapbox'); if (!box) return;
  box.style.setProperty('--mapw', document.documentElement.clientWidth + 'px');
  const hd = $('header.top'); box.style.setProperty('--maphdr', (hd ? hd.offsetHeight : 52) + 'px');
}
const toMap = () => scrollTo({ top: 0, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });

/* ------------------------------------------------------------------ under the stage: three expanders, every number from D */
function buildBelow(F) {
  const el = $('#map-explain'); if (!el || el.childElementCount) return;
  const { N, T, NT } = F, tps = D.meta.topics, nb = D.m.nb;
  let same = 0; for (let i = 0; i < N; i++) if (T[nb[i * 8]] === T[i]) same++;
  // the worked example: "framing", or else the most frequent label
  let lab = 'framing', ix = [];
  for (let i = 0; i < N; i++) if (D.m.label[i].toLowerCase() === lab) ix.push(i);
  if (ix.length < 5) { const c = new Map(); for (const l0 of D.m.label) { const l = l0.toLowerCase(); c.set(l, (c.get(l) || 0) + 1); } lab = [...c].sort((a, b) => b[1] - a[1])[0][0]; ix = []; for (let i = 0; i < N; i++) if (D.m.label[i].toLowerCase() === lab) ix.push(i); }
  const tc = new Map(); for (const i of ix) tc.set(T[i], (tc.get(T[i]) || 0) + 1);
  const ranked = [...tc].sort((a, b) => b[1] - a[1]);
  const { lead } = topicRanks(F), nLead = lead.filter(Boolean).length;
  el.append(
    expander({ title: 'How to read the map', meta: H('span', {}, H('b', { text: pc(same / N, 0) }), ' sit next to their own topic'), build: (b) => howTo(b, F, same) }),
    expander({ title: `Worked example: “${lab}”`, meta: H('span', {}, H('b', { text: nf(ix.length) }), ` mentions in ${ranked.length} topic${ranked.length === 1 ? '' : 's'}`), build: (b) => example(b, F, lab, ix, ranked) }),
    expander({ id: 'map-tt', title: `All ${NT} topics`, meta: H('span', {}, H('b', { text: String(nLead) }), ` led by a bias concept`), build: (b) => topicTable(b, F) }));
}
function howTo(b, F, same) {
  const bl = D.meta.baseline;
  b.append(H('div', { class: 'stats' },
    stat({ value: f2(bl.random_pair_cos_mean), label: 'two random mentions', sub: 'mean similarity of their definitions' }),
    stat({ value: f2(bl.nearest_neighbour_cos_mean), label: 'a mention and its nearest', sub: 'mean similarity: near means alike', lead: true }),
    stat({ value: pc(same / F.N, 0), label: 'nearest is in the same topic', sub: `of ${nf(F.N)} mentions` })),
  H('p', {}, 'Each dot is one ', gl('mention', 'term mention'), ': a term as one paper defines it. Mentions defined in similar words sit close together; each colour is one of ', String(F.NT), ' ', gl('topic', 'topics'), '.'),
  more('How this is made',
    H('p', {}, 'Each mention’s definitions become one ', gl('embedding'), '; ', gl('umap'), ' turns these into map positions. Who sits next to whom is reliable; long distances across the map are only rough.'),
    H('p', {}, 'Topics are found on the vectors themselves, not on the map, and are named after the concepts with the most mentions in them. Where names would overlap, the topic led by “media bias” keeps its name first, then the other topics led by a bias concept.'),
    H('p', {}, 'The other colourings use the same layout: the ', gl('ensemble', 'ensemble score'), ' and its members (', gl('classifier', 'classifier'), ', ', gl('basal'), ', ', gl('judge', 'judge'), '), ', gl('strict', 'strict'), ' or not, a ', gl('seed', 'seed'), ' or ', gl('closure', 'cited'), ' paper, or chosen ', gl('concept', 'concepts'), '.'),
    H('p', {}, 'Keys: / find · R reset · 2 or 3 view · N names · T topics · C controls · Esc clears. Double-click a dot to fly to it.')));
}
function example(b, F, lab, ix, ranked) {
  const { T } = F, tps = D.meta.topics, nb = D.m.nb, nbs = D.m.nbs;
  const sw = (t) => H('i', { class: 'tsw', 'data-tc': t });
  const papers = new Set(ix.map((i) => D.m.paper[i])).size, shown = ranked.slice(0, 4), rest = ranked.slice(4).reduce((s, [, n]) => s + n, 0);
  // the example mention of a topic: a readable paper title, typical (at least 6 of its 8 nearest mentions in its
  // topic), then the one whose neighbours carry the most different labels
  const others = (i) => { const own = D.m.label[i].toLowerCase(), seen = new Set(); for (let k = 0; k < 8; k++) { const l = D.m.label[nb[i * 8 + k]].toLowerCase(); if (l !== own) seen.add(l); } return seen.size; };
  const inTopic = (i) => { let n = 0; for (let k = 0; k < 8; k++) n += T[nb[i * 8 + k]] === T[i] ? 1 : 0; return Math.min(n, 6); };
  const readable = (i) => { const t = D.p.title[D.m.paper[i]] || ''; return t.length >= 20 && t.length <= 160 && (t.match(/[a-z]/g) || []).length > 0.5 * t.length ? 1 : 0; };
  const best = (t) => ix.filter((i) => T[i] === t).map((i) => [i, readable(i), inTopic(i), others(i), D.score.s[i]]).sort((a, c) => c[1] - a[1] || c[2] - a[2] || c[3] - a[3] || c[4] - a[4])[0][0];
  const ex = best(ranked[0][0]), ex2 = ranked.length > 1 ? best(ranked[1][0]) : -1;
  const title = (i) => `“${trunc(D.p.title[D.m.paper[i]] || D.p.id[D.m.paper[i]], 80)}”`;
  const isolate = (t) => { if (MAP.isolate) MAP.isolate(t); toMap(); };
  const goto = (j) => { TABS.map.focus(j); toMap(); };
  const bars = H('ul', { class: 'mx-bars' }), share = (n) => (n / ix.length < 0.005 ? '<1%' : pc(n / ix.length, 0));
  for (const [t, n] of shown) bars.append(H('li', {}, H('button', { type: 'button', class: 'linkbtn', title: `show only this topic on the map: ${tps[t].name}`, onclick: () => isolate(t) }, sw(t), H('span', { text: tshort(t) })), H('span', { class: 'bar' }, H('i', { 'data-tc': t, style: `width:${(100 * n) / ix.length}%` })), H('span', { class: 'small mono', text: `${nf(n)} · ${share(n)}` })));
  if (rest) bars.append(H('li', {}, H('span', { class: 'small muted', text: `${ranked.length - shown.length} other topics` }), H('span', { class: 'bar' }, H('i', { style: `width:${(100 * rest) / ix.length}%;background:var(--muted)` })), H('span', { class: 'small mono', text: `${nf(rest)} · ${share(rest)}` })));
  const chips = H('div', { class: 'mx-nb' });
  for (let k = 0; k < 8; k++) { const j = nb[ex * 8 + k]; chips.append(H('button', { type: 'button', title: `fly to this mention · topic: ${tps[T[j]].name}`, onclick: () => goto(j) }, sw(T[j]), D.m.label[j], H('span', { class: 'mono', text: f2(nbs[ex * 8 + k] / 1000) }))); }
  let nSame = 0; for (let k = 0; k < 8; k++) nSame += T[nb[ex * 8 + k]] === T[ex] ? 1 : 0;
  b.append(H('p', { class: 'small muted', text: `${nf(ix.length)} mentions from ${nf(papers)} papers carry the label “${lab}” (any capitalisation). Where they land:` }), bars,
    H('p', { class: 'small' }, 'In ', title(ex), ' (', pill(ex), `), ${nSame} of its 8 `, gl('neighbours', 'nearest mentions'), ' share its topic:'), chips);
  if (ex2 >= 0) {
    const diff = [], seen = new Set([lab.toLowerCase()]);
    for (let k = 0; k < 8 && diff.length < 3; k++) { const j = nb[ex2 * 8 + k], l = D.m.label[j].toLowerCase(); if (!seen.has(l)) { seen.add(l); diff.push(j); } }
    const links = diff.flatMap((j, k) => [k ? (k === diff.length - 1 ? ' and ' : ', ') : '', H('button', { type: 'button', class: 'linkbtn', text: D.m.label[j], onclick: () => goto(j) })]);
    b.append(H('p', { class: 'small' }, 'In ', title(ex2), ` it lands in “${tshort(T[ex2])}”`, diff.length ? ', next to ' : null, ...links, '. Same word, different company.'));
  }
  b.append(H('div', { class: 'ctl' },
    H('button', { type: 'button', class: 'btn primary', text: `Light up all ${nf(ix.length)}`, onclick: () => { if (MAP.setQuery) MAP.setQuery(`"${lab}"`, true); else $('#map-q').value = `"${lab}"`; toMap(); } }),
    H('button', { type: 'button', class: 'btn', text: 'Fly to the first', onclick: () => goto(ex) }),
    H('button', { type: 'button', class: 'btn', text: 'Read it', onclick: () => openMention(ex) })));
  paintTopics(F);
}
function topicTable(b, F) {
  const { N, T, NT } = F, tps = D.meta.topics, { lead } = topicRanks(F);
  const groups = Array.from({ length: NT }, () => new Set()), papers = Array.from({ length: NT }, () => new Set()), strict = new Int32Array(NT);
  for (let i = 0; i < N; i++) { const t = T[i]; groups[t].add(D.m.group[i]); papers[t].add(D.m.paper[i]); strict[t] += D.strict[i]; }
  const rows = tps.map((tp, t) => ({ t, name: tp.top.join(' · '), lead: lead[t], n: F.size[t], g: groups[t].size, p: papers[t].size, s: strict[t] / Math.max(1, F.size[t]) }));
  const cols = [['name', 'Topic: its largest concepts'], ['n', 'Mentions'], ['g', 'Concepts'], ['p', 'Papers'], ['s', 'Strict']];
  const st = { k: 'n', dir: -1 }, wrap = H('div', { class: 'tblwrap' }), tb = H('tbody');
  const head = H('tr', {}, cols.map(([k, t]) => H('th', { 'data-k': k }, H('button', { type: 'button', text: t, onclick: () => { st.dir = st.k === k ? -st.dir : k === 'name' ? 1 : -1; st.k = k; draw(); } }))));
  const go = (t) => { if (MAP.isolate) { MAP.isolate(t, true); if (MAP.flyTopic) MAP.flyTopic(t); } toMap(); };
  function draw() {
    for (const th of $$('th', head)) th.setAttribute('aria-sort', th.dataset.k === st.k ? (st.dir > 0 ? 'ascending' : 'descending') : 'none');
    tb.textContent = '';
    for (const r of [...rows].sort((a, c) => (st.k === 'name' ? a.name.localeCompare(c.name) : a[st.k] - c[st.k]) * st.dir)) {
      tb.append(H('tr', { class: 'row', title: 'show only this topic on the map', onclick: () => go(r.t) },
        H('td', { class: 'wrapc' }, H('i', { class: 'tsw', 'data-tc': r.t }), ' ', r.name, r.lead ? H('span', { class: 'tbias', text: 'bias' }) : null),
        H('td', { class: 'n', text: nf(r.n) }), H('td', { class: 'n', text: nf(r.g) }), H('td', { class: 'n', text: nf(r.p) }), H('td', { class: 'n', text: pc(r.s, 0) })));
    }
    paintTopics(F);
  }
  wrap.append(H('table', { class: 'mx-tt' }, H('thead', {}, head), tb)); draw();
  b.append(H('p', { class: 'small muted' }, 'Click a row to show only that topic on the map. Concepts and papers count those with at least one mention in the topic; ', gl('strict', 'strict'), ' is the share of its mentions that are strict.'), wrap);
  paintTopics(F);
}

/* ------------------------------------------------------------------ start: data first, then three.js */
async function startMap() {
  const F = mapFrame(); MAP.F = F;
  layoutStage(); addEventListener('resize', layoutStage);
  if (window.ResizeObserver && $('header.top')) new ResizeObserver(layoutStage).observe($('header.top'));
  $('#map-count').textContent = `${nf(F.N)} mentions · ${F.NT} topics`;
  try { buildBelow(F); } catch (e) { console.error(e); }
  if (matchMedia('(max-width: 1000px)').matches) $('#maptopics').open = false;
  let THREE, OrbitControls;
  try { THREE = await import('three'); ({ OrbitControls } = await import('three/addons/controls/OrbitControls.js')); }
  catch (e) { $('#mapload').textContent = 'The 3D view could not load its library from cdn.jsdelivr.net: ' + e.message; return; }
  try { buildMap(THREE, OrbitControls, F); } catch (e) { console.error(e); $('#mapload').textContent = 'The map could not be drawn: ' + e.message; }
}

function buildMap(THREE, OrbitControls, F) {
  const { N, T, NT, pos3, pos2, anc3, anc2 } = F;
  const sec = $('#v-map'), box = $('#mapbox'), canvas = $('#mapcanvas'), tip = $('#maptip'), rail = $('#maprail'), tpanel = $('#maptopics'), hint = $('#maphint');
  const tps = D.meta.topics, lower = D.m.label.map((s) => s.toLowerCase()), gLower = D.g.name.map((s) => s.toLowerCase()), seedOf = Uint8Array.from(D.m.paper, (p) => (D.p.corpus[p] === 'seeds' ? 1 : 0));
  const { lead: biasLed, ord } = topicRanks(F);
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)'), coarse = matchMedia('(pointer: coarse)').matches, narrow = matchMedia('(max-width: 700px)');

  /* ---- renderer, camera, controls */
  THREE.ColorManagement.enabled = false; // every colour below is already sRGB: pass it straight through
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' }); }
  catch (e) { $('#mapload').textContent = 'This view needs WebGL, which this browser refused.'; return; }
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  const DPR = Math.min(devicePixelRatio || 1, 2); renderer.setPixelRatio(DPR);
  const glc = renderer.getContext(), maxPx = (glc.getParameter(glc.ALIASED_POINT_SIZE_RANGE) || [1, 64])[1];
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 800);
  const controls = new OrbitControls(camera, canvas);
  Object.assign(controls, { enableDamping: true, dampingFactor: 0.08, rotateSpeed: 0.7, zoomSpeed: 1.1, panSpeed: 0.9, screenSpacePanning: true, minDistance: 0.4, maxDistance: 160, autoRotateSpeed: 0.5 });
  if ('zoomToCursor' in controls) controls.zoomToCursor = true;
  controls.listenToKeyEvents(canvas);

  /* ---- the ground: a soft radial light with a little dither against banding */
  const bgMat = new THREE.ShaderMaterial({
    uniforms: { uC0: { value: new THREE.Color() }, uC1: { value: new THREE.Color() }, uRes: { value: new THREE.Vector2(1, 1) } },
    vertexShader: 'varying vec2 vUv; void main() { vUv = position.xy * 0.5 + 0.5; gl_Position = vec4(position.xy, 0.999, 1.0); }',
    fragmentShader: `uniform vec3 uC0, uC1; uniform vec2 uRes; varying vec2 vUv;
      void main() {
        float a = uRes.x / uRes.y;
        vec2 p = (vUv - vec2(0.5, 0.52)) * vec2(a, 1.0);
        float r = length(p) / max(0.62, 0.5 * a);
        vec3 c = mix(uC0, uC1, smoothstep(0.0, 1.15, r));
        float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
        gl_FragColor = vec4(c + (n - 0.5) / 255.0, 1.0);
      }`,
    depthTest: false, depthWrite: false,
  });
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bgMat); ground.frustumCulled = false; ground.renderOrder = -1; scene.add(ground);

  /* ---- the dots */
  const FOG = 0.5;
  const U = { uSize: { value: 0.085 }, uScale: { value: 400 }, uPixelRatio: { value: DPR }, uMinPx: { value: 1.6 }, uMaxPx: { value: Math.min(15, maxPx / DPR) },
    uFogNear: { value: 20 }, uFogFar: { value: 40 }, uFogAmt: { value: FOG }, uBg: { value: new THREE.Color() } };
  const dotVertex = `
    uniform float uSize, uScale, uPixelRatio, uMinPx, uMaxPx, uPass, uFogNear, uFogFar, uFogAmt; uniform vec3 uBg;
    attribute vec3 aColor; attribute float aSize; attribute float aAlpha;
    varying vec3 vColor; varying float vAlpha;
    void main() {
      bool opaque = aAlpha > 0.995;
      if (aSize <= 0.0 || (uPass < 0.5 ? !opaque : opaque)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      float depth = max(-mv.z, 0.001);
      float fog = clamp((depth - uFogNear) / max(uFogFar - uFogNear, 0.001), 0.0, 1.0) * uFogAmt;   // dots far back sink into the ground
      vColor = mix(aColor, uBg, fog); vAlpha = aAlpha;
      gl_PointSize = clamp(aSize * uSize * uScale / depth, uMinPx, uMaxPx) * uPixelRatio;
    }`;
  const dotFragment = `
    uniform float uPass; varying vec3 vColor; varying float vAlpha;
    void main() {
      vec2 c = gl_PointCoord * 2.0 - 1.0;
      float r = length(c);
      float aa = max(fwidth(r) * 1.25, 0.03);
      float cov = 1.0 - smoothstep(1.0 - aa, 1.0, r);
      if (cov <= 0.002) discard;
      vec3 col = vColor * (1.0 - 0.2 * smoothstep(0.45, 1.0, r));                              // a slightly darker rim
      col += 0.07 * (1.0 - smoothstep(0.0, 0.55, length(c - vec2(-0.32, -0.32))));                // a faint light from the top left
      gl_FragColor = vec4(col, uPass < 0.5 ? cov : cov * vAlpha);
    }`;
  const cur = pos3.slice();
  const geom = new THREE.BufferGeometry();
  const posAttr = new THREE.BufferAttribute(cur, 3); posAttr.setUsage(THREE.DynamicDrawUsage);
  const aColor = new THREE.BufferAttribute(new Uint8Array(N * 3), 3, true), aSize = new THREE.BufferAttribute(new Float32Array(N), 1), aAlpha = new THREE.BufferAttribute(new Uint8Array(N), 1, true);
  geom.setAttribute('position', posAttr); geom.setAttribute('aColor', aColor); geom.setAttribute('aSize', aSize); geom.setAttribute('aAlpha', aAlpha);
  geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e5);
  // pass 0: full dots, depth-written, edges smoothed by alpha-to-coverage; pass 1: faded dots, blended without writing depth
  const matMain = new THREE.ShaderMaterial({ uniforms: { ...U, uPass: { value: 0 } }, vertexShader: dotVertex, fragmentShader: dotFragment, transparent: false, depthWrite: true, depthTest: true });
  matMain.alphaToCoverage = true;
  const matGhost = new THREE.ShaderMaterial({ uniforms: { ...U, uPass: { value: 1 } }, vertexShader: dotVertex, fragmentShader: dotFragment, transparent: true, depthWrite: false, depthTest: true });
  const dotsMain = new THREE.Points(geom, matMain), dotsGhost = new THREE.Points(geom, matGhost);
  dotsMain.frustumCulled = dotsGhost.frustumCulled = false; dotsGhost.renderOrder = 1;
  scene.add(dotsMain, dotsGhost);

  /* ---- rings mark the hovered and the focused dot */
  const ringGeom = new THREE.BufferGeometry();
  const ringPos = new THREE.BufferAttribute(new Float32Array(6), 3), ringSize = new THREE.BufferAttribute(new Float32Array(2), 1), ringPad = new THREE.BufferAttribute(new Float32Array(2), 1), ringThick = new THREE.BufferAttribute(new Float32Array(2), 1), ringCol = new THREE.BufferAttribute(new Uint8Array(6), 3, true);
  ringGeom.setAttribute('position', ringPos); ringGeom.setAttribute('aSize', ringSize); ringGeom.setAttribute('aPad', ringPad); ringGeom.setAttribute('aThick', ringThick); ringGeom.setAttribute('aColor', ringCol);
  ringGeom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e5); ringGeom.setDrawRange(0, 0);
  const ringMat = new THREE.ShaderMaterial({
    uniforms: { ...U, uHalo: { value: new THREE.Color() } },
    vertexShader: `uniform float uSize, uScale, uPixelRatio, uMinPx, uMaxPx; attribute float aSize, aPad, aThick; attribute vec3 aColor;
      varying float vPx, vThick; varying vec3 vInk;
      void main() { vInk = aColor; vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;
        float px = clamp(aSize * uSize * uScale / max(-mv.z, 0.001), uMinPx, uMaxPx) + aPad; vPx = px; vThick = aThick; gl_PointSize = px * uPixelRatio; }`,
    fragmentShader: `uniform vec3 uHalo; varying float vPx, vThick; varying vec3 vInk;
      void main() { float r = length(gl_PointCoord * 2.0 - 1.0); if (r > 1.0) discard;
        float d = (1.0 - r) * vPx * 0.5;
        float ink = smoothstep(0.0, 0.9, d) * (1.0 - smoothstep(vThick, vThick + 0.9, d));
        float halo = smoothstep(vThick, vThick + 0.9, d) * (1.0 - smoothstep(vThick + 1.6, vThick + 2.5, d));
        float a = max(ink, halo * 0.85); if (a < 0.01) discard;
        gl_FragColor = vec4(mix(uHalo, vInk, ink / max(ink + halo, 1e-4)), a); }`,
    transparent: true, depthTest: false, depthWrite: false,
  });
  const rings = new THREE.Points(ringGeom, ringMat); rings.frustumCulled = false; rings.renderOrder = 4; scene.add(rings);

  /* ---- state */
  const st = { dim: '3', color: MAP.pendingColor || 'topic', lastScore: 'score', only: -1, q: '', qg: -1, names: true, focus: -1, hover: -1, topics: new Set(), preview: -1, spin: false, sizeK: 1, autoK: 1 };
  MAP.pendingColor = null;
  let atHome = true, TH = null, needsRender = true, projDirty = true, flight = null, morph = null, mix = 0, pendingHover = null, lastPick = 0, down = null;
  const match = new Uint8Array(N); let nMatch = 0;
  const isScore = (m) => ['score', 'c', 'b', 'j'].includes(m);

  function readTheme() { // everything from the stage's own tokens: the stage is dark in both themes
    const cs = getComputedStyle(box), c = (v) => rgbOf(cs.getPropertyValue(v)), bg0 = c('--map-bg0'), bg1 = c('--map-bg1'), dark = okLab(bg1)[0] < 0.5;
    return { dark, bg0, bg1, ink: c('--ink'), accent: c('--accent'), muted: c('--muted'), kept: c('--kept'), strict: c('--strict'),
      topic: mapPalette(F, dark), lut: virLut(dark), seed: rgbOf(COLS[0]), cited: rgbOf(COLS[1]), concept: COLS.map(rgbOf), ghost: dark ? 34 : 30 };
  }
  function applyTheme() {
    TH = readTheme();
    const s = (c) => c.map((v) => v / 255);
    bgMat.uniforms.uC0.value.setRGB(...s(TH.bg0)); bgMat.uniforms.uC1.value.setRGB(...s(TH.bg1));
    renderer.setClearColor(new THREE.Color(...s(TH.bg1)));
    U.uBg.value.setRGB(...TH.bg0.map((v, k) => (0.45 * v + 0.55 * TH.bg1[k]) / 255));
    ringMat.uniforms.uHalo.value.setRGB(...s(TH.bg0));
    paintTopics(F); drawLegend(); refresh();
  }

  /* ---- colour, size and fade of every dot */
  function refresh() {
    const col = aColor.array, size = aSize.array, alpha = aAlpha.array, mode = st.color;
    const sel = st.preview >= 0 ? new Set([st.preview]) : st.topics, hasSel = sel.size > 0, q = !!st.q;
    const ck = new Map([...MAPSEL].map((g, k) => [g, k])), cOn = mode === 'concept' && ck.size > 0;
    const sc = mode === 'score' ? D.score.s : ['c', 'b', 'j'].includes(mode) ? D.score[mode] : null, pal = TH.topic, lut = TH.lut;
    const cls = st.only >= 0 && mode === 'strict' ? D.strict : st.only >= 0 && mode === 'corpus' ? seedOf : null;
    for (let i = 0; i < N; i++) {
      const t = T[i];
      let rgb;
      if (mode === 'topic') rgb = pal[t];
      else if (sc) rgb = lut[Math.min(255, Math.max(0, Math.round(sc[i] * 255)))];
      else if (mode === 'strict') rgb = D.strict[i] ? TH.strict : TH.kept;
      else if (mode === 'corpus') rgb = seedOf[i] ? TH.seed : TH.cited;
      else { const k = ck.get(D.m.group[i]); rgb = k === undefined ? TH.muted : TH.concept[k % 12]; }
      let on = !hasSel || sel.has(t), boost = 1;
      if (on && cls && cls[i] !== st.only) on = false;
      if (on && q) { if (match[i]) boost = 1.45; else on = false; }
      if (on && cOn) { if (ck.has(D.m.group[i])) boost *= 1.35; else on = false; }
      col[3 * i] = rgb[0]; col[3 * i + 1] = rgb[1]; col[3 * i + 2] = rgb[2];
      alpha[i] = on ? 255 : TH.ghost; size[i] = on ? boost : 0.72;
    }
    aColor.needsUpdate = aSize.needsUpdate = aAlpha.needsUpdate = true;
    projDirty = true; needsRender = true; updateRings();
  }

  /* ---- the legend of each colour mode: small, and itself a control where it can be */
  const histCache = {};
  function scoreHist(m) {
    if (histCache[m]) return histCache[m];
    const sc = m === 'score' ? D.score.s : D.score[m], nb = 40, h = new Array(nb).fill(0);
    for (let i = 0; i < N; i++) h[Math.min(nb - 1, Math.floor(sc[i] * nb))]++;
    return (histCache[m] = h);
  }
  function drawLegend() {
    const el = $('#map-legend'); el.textContent = ''; const m = st.color;
    if (m === 'topic') {
      const strip = H('div', { class: 'tstrip', role: 'group', 'aria-label': 'topics by size' });
      for (const t of [...F.size.keys()].sort((a, b) => F.size[b] - F.size[a])) {
        strip.append(H('button', { type: 'button', tabindex: '-1', 'data-tc': t, 'aria-pressed': String(st.topics.has(t)), style: `flex:${F.size[t]} 1 0`, title: `${tps[t].name} · ${nf(F.size[t])} mentions`, 'aria-label': tps[t].name,
          onclick: (e) => pickTopic(t, e), onpointerenter: (e) => { if (e.pointerType === 'mouse') preview(t); }, onpointerleave: () => preview(-1) }));
      }
      el.append(strip, H('div', { class: 'mcap' }, gl('topic', `${NT} topics`), H('span', { text: 'by size · hover or click' })));
    } else if (isScore(m)) {
      const h = scoreHist(m), W = 240, Hh = 30, nb = h.length, mx = Math.max(...h), name = m === 'score' ? 'ensemble' : SC[m];
      const svg = S('svg', { viewBox: `0 0 ${W} ${Hh + 3}`, class: 'mhist', role: 'img', 'aria-label': `how the ${name} scores of the mentions spread from 0 to 1` });
      h.forEach((v, k) => {
        const bh = v ? Math.max(1.2, (v / mx) * Hh) : 0, c = TH.lut[Math.min(255, Math.round(((k + 0.5) / nb) * 255))];
        const r = S('rect', { x: (k * W) / nb + 0.4, y: Hh - bh, width: W / nb - 0.8, height: bh, rx: 0.8, fill: rgbStr(c) }, svg);
        S('title', {}, r).textContent = `${f2(k / nb)}–${f2((k + 1) / nb)}: ${nf(v)} mentions`;
      });
      S('rect', { x: 0, y: Hh + 1, width: W, height: 2, rx: 1, fill: 'url(#mramp)' }, svg);
      const gr = S('linearGradient', { id: 'mramp' }, S('defs', {}, svg));
      [0, 0.25, 0.5, 0.75, 1].forEach((u) => S('stop', { offset: u, 'stop-color': rgbStr(TH.lut[Math.round(u * 255)]) }, gr));
      if (m === 'score') S('line', { x1: D.meta.cut * W, x2: D.meta.cut * W, y1: 0, y2: Hh + 3, class: 'mcut' }, svg);
      el.append(svg, H('div', { class: 'mcap' }, H('span', { text: '0' }), m === 'score' ? H('span', {}, gl('ensemble', 'ensemble'), ' · every dot ≥ ', gl('cut', 'cut'), ` ${D.meta.cut.toFixed(3)}`) : H('span', {}, gl(SC[m], SC[m]), ' score'), H('span', { text: '1' })));
    } else if (m === 'strict' || m === 'corpus') {
      const cl = m === 'strict' ? D.strict : seedOf; let a = 0; for (let i = 0; i < N; i++) a += cl[i];
      const keys = m === 'strict' ? [[1, 'var(--strict)', 'strict', a], [0, 'var(--kept)', 'not strict', N - a]] : [[1, COLS[0], 'seed papers', a], [0, COLS[1], 'cited papers', N - a]];
      el.append(H('div', { class: 'mkeys' }, keys.map(([v, c, t, n]) => H('button', { type: 'button', class: 'mkey', 'aria-pressed': String(st.only === v), style: `--kc:${c}`, title: `show only: ${t}`,
        onclick: () => { st.only = st.only === v ? -1 : v; drawLegend(); refresh(); } }, H('i'), H('span', { text: t }), H('b', { class: 'mono', text: nf(n) }), H('span', { class: 'mono', text: pc(n / N, 0) })))),
      H('div', { class: 'mcap' }, m === 'strict' ? H('span', {}, 'all ', gl('kept', 'kept'), '; ', gl('strict', 'strict'), ' also pass ', gl('bestf1', 'best-F1'), ` ${D.meta.best_f1_cut.toFixed(3)}`) : H('span', {}, 'the mention’s paper: ', gl('seed', 'seed'), ' or ', gl('closure', 'cited'))));
    } else if (m === 'concept') {
      const cc = H('div', { class: 'mchips' });
      [...MAPSEL].forEach((g, k) => cc.append(H('button', { class: 'mchip', type: 'button', title: 'remove from the map', onclick: () => { MAPSEL.delete(g); drawLegend(); refresh(); } }, H('i', { style: `background:${COLS[k % 12]}` }), H('span', { text: D.g.name[g] }), H('b', { class: 'mono', text: nf(D.g.n[g]) }), ' ×')));
      el.append(MAPSEL.size ? cc : H('div', { class: 'mcap' }, H('span', {}, 'None yet: pick the largest, or “Colour on the map” in ', H('button', { type: 'button', class: 'linkbtn', text: 'Concepts', onclick: () => show('concepts') }))),
        H('div', { class: 'mrow' }, H('button', { type: 'button', class: 'mbtn', text: 'The 12 largest', onclick: () => { MAPSEL.clear(); Array.from(D.g.n.keys()).sort((a, b) => D.g.n[b] - D.g.n[a]).slice(0, 12).forEach((g) => MAPSEL.add(g)); drawLegend(); refresh(); } }),
          MAPSEL.size ? H('button', { type: 'button', class: 'mbtn', text: 'Clear', onclick: () => { MAPSEL.clear(); drawLegend(); refresh(); } }) : null,
          MAPSEL.size ? H('button', { type: 'button', class: 'mbtn', text: 'Fly there', onclick: () => flyToSet([...MAPSEL].flatMap((g) => D.members[g])) }) : null));
    }
    paintTopics(F);
  }
  function setColor(c) { if (isScore(c)) st.lastScore = c; if (c !== st.color) st.only = -1; st.color = c; drawColorSeg(); drawLegend(); refresh(); }
  function drawColorSeg() {
    const top = isScore(st.color) ? 'score' : st.color;
    seg($('#seg-color'), [['topic', 'topic'], ['score', 'score'], ['strict', 'strict'], ['corpus', 'seed\u2009/\u2009cited'], ['concept', 'concepts']], top, (v) => setColor(v === 'score' ? st.lastScore : v));
    const sc = $('#seg-score'); sc.hidden = top !== 'score';
    if (top === 'score') seg(sc, [['score', 'ensemble'], ['c', 'classifier'], ['b', 'basal'], ['j', 'judge']], st.color, setColor); else sc.textContent = '';
  }

  /* ---- the topic list: bias-led topics first, then by size; hover lights one up, click shows only it */
  const tlist = $('#map-tlist'), tmax = Math.max(...F.size, 1), rows = new Array(NT), nLead = biasLed.filter(Boolean).length;
  ord.forEach((t, k) => {
    const tp = tps[t];
    const r = H('button', { type: 'button', class: 'trow', 'data-tc': t, 'aria-pressed': 'false', title: `${tp.name} · ${nf(F.size[t])} mentions. Click: only this topic (again: all); Shift-click: add it.`, style: `--p:${(100 * F.size[t]) / tmax}%` },
      H('i'), H('span', { class: 'tn', text: tshort(t) }), biasLed[t] ? H('span', { class: 'tbias', text: 'bias' }) : H('span'), H('span', { class: 'tc', text: nf(F.size[t]) }));
    r.addEventListener('click', (e) => pickTopic(t, e));
    r.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') preview(t); });
    r.addEventListener('pointerleave', () => preview(-1));
    rows[t] = r; tlist.append(r);
    if (k === nLead - 1) tlist.append(H('div', { class: 'tsep', 'aria-hidden': 'true' }));
  });
  $('#map-tq').addEventListener('input', (e) => {
    const q = e.target.value.trim().toLowerCase(); let n = 0;
    rows.forEach((r, t) => { const on = !q || tps[t].top.some((x) => x.toLowerCase().includes(q)); r.hidden = !on; n += on ? 1 : 0; });
    const sep = $('.tsep', tlist); if (sep) sep.hidden = !!q; $('#map-tnone').hidden = n > 0;
  });
  function preview(t) { if (st.preview === t) return; st.preview = t; refresh(); }
  function syncTopics() {
    const n = st.topics.size;
    rows.forEach((r, t) => r.setAttribute('aria-pressed', String(st.topics.has(t))));
    labs.forEach((l, t) => l.setAttribute('aria-pressed', String(st.topics.has(t))));
    for (const b of $$('.tstrip button', rail)) b.setAttribute('aria-pressed', String(st.topics.has(+b.dataset.tc)));
    tlist.classList.toggle('has-sel', n > 0); rail.classList.toggle('has-sel', n > 0);
    $('#map-tall').disabled = !n; $('#map-tfly').disabled = !n;
    $('#map-tsum').textContent = n ? `${n} of ${NT}` : `${NT}`;
    refresh();
  }
  function toggleTopic(t) { if (st.topics.has(t)) st.topics.delete(t); else st.topics.add(t); syncTopics(); }
  function isolate(t, keep) { st.topics = !keep && st.topics.size === 1 && st.topics.has(t) ? new Set() : new Set([t]); syncTopics(); }
  function pickTopic(t, e) { if (e && (e.shiftKey || e.ctrlKey || e.metaKey)) toggleTopic(t); else isolate(t); }
  $('#map-tall').addEventListener('click', () => { st.topics.clear(); syncTopics(); });
  $('#map-tfly').addEventListener('click', () => flyToSet([...st.topics].flatMap((t) => F.members[t])));

  /* ---- floating names (each topic's largest concept; the full name on hover) and the pin of the focused mention */
  const lw = $('#maplabels');
  const labs = tps.map((tp, t) => {
    const b = H('button', { type: 'button', class: 'tlab' + (tp.size >= 2000 ? ' big' : '') + (biasLed[t] ? ' bias' : ''), 'data-tc': t, 'aria-pressed': 'false', title: `${tp.name} · ${nf(tp.size)} mentions. Click to show only this topic; click again to show all.`, text: tp.top[0] });
    b.addEventListener('click', () => isolate(t));
    b.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') preview(t); });
    b.addEventListener('pointerleave', () => preview(-1));
    b.hidden = true; lw.append(b); return b;
  });
  const labW = new Float32Array(NT);
  const pin = H('button', { type: 'button', class: 'mpin', title: 'read this mention' }, H('span')); pin.hidden = true; let pinW = 0;
  pin.addEventListener('click', () => { if (st.focus >= 0) openMention(st.focus); });
  lw.append(pin);
  const pm = new THREE.Matrix4();
  const smooth = (a, b, x) => { const u = Math.min(1, Math.max(0, (x - a) / (b - a))); return u * u * (3 - 2 * u); };
  function placeLabels() {
    const W = box.clientWidth, Hh = box.clientHeight; if (!W) return;
    camera.updateMatrixWorld(); pm.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const m = pm.elements, dC = camera.position.distanceTo(controls.target), placed = [], br = box.getBoundingClientRect();
    for (const p of [rail, tpanel, hint.classList.contains('gone') ? null : hint]) { if (!p) continue; const r = p.getBoundingClientRect(); if (r.width) placed.push([r.left - br.left - 4, r.top - br.top - 4, r.right - br.left + 4, r.bottom - br.top + 4]); } // names stay clear of the panels
    const toScreen = (x, y, z) => { const w = m[3] * x + m[7] * y + m[11] * z + m[15]; if (w <= 0.05) return null; return [((m[0] * x + m[4] * y + m[8] * z + m[12]) / w * 0.5 + 0.5) * W, (0.5 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / w * 0.5) * Hh, w]; };
    pin.hidden = true;
    if (st.focus >= 0) {
      const i = st.focus, p = toScreen(cur[i * 3], cur[i * 3 + 1], cur[i * 3 + 2]);
      if (p && p[0] > -40 && p[0] < W + 40 && p[1] > 0 && p[1] < Hh + 40) {
        pin.hidden = false; if (!pinW) pinW = pin.offsetWidth;
        const bx = [p[0] - pinW / 2, p[1] - 44, p[0] + pinW / 2, p[1] - 14];
        pin.style.transform = `translate(${Math.round(bx[0])}px, ${Math.round(bx[1])}px)`; placed.push(bx);
      }
    }
    const sel = st.preview >= 0 ? new Set([st.preview]) : st.topics, cand = [];
    for (let t = 0; t < NT; t++) {
      const l = labs[t];
      if (!st.names) { l.hidden = true; continue; }
      const x = anc3[t * 3] + (anc2[t * 3] - anc3[t * 3]) * mix, y = anc3[t * 3 + 1] + (anc2[t * 3 + 1] - anc3[t * 3 + 1]) * mix, z = anc3[t * 3 + 2] * (1 - mix);
      const p = toScreen(x, y, z);
      if (!p || p[0] < -20 || p[1] < -10 || p[0] > W + 20 || p[1] > Hh + 10) { l.hidden = true; continue; }
      const fade = 1 - 0.72 * smooth(dC - 3, dC + 11, p[2]) * (1 - mix);
      // the label rule: a chosen topic first, then the one led by "media bias", then the other bias-led ones, then size
      cand.push({ t, sx: p[0], sy: p[1], fade, pri: (sel.has(t) ? 1e9 : 0) + biasLed[t] * 1e8 + tps[t].size * (dC / p[2]) ** 2 });
    }
    cand.sort((a, b) => b.pri - a.pri);
    for (const c of cand) {
      const l = labs[c.t], bw = labW[c.t] || tps[c.t].top[0].length * 7.2 + 22, bh = 24;
      c.sx = Math.min(Math.max(c.sx, bw / 2 + 4), W - bw / 2 - 4); // keep the whole name inside the stage
      const bx = [c.sx - bw / 2 - 3, c.sy - bh / 2 - 2, c.sx + bw / 2 + 3, c.sy + bh / 2 + 2];
      if (placed.some((p) => bx[0] < p[2] && bx[2] > p[0] && bx[1] < p[3] && bx[3] > p[1])) { l.hidden = true; continue; }
      placed.push(bx); l.hidden = false;
      if (!labW[c.t]) labW[c.t] = l.offsetWidth;
      l.style.transform = `translate(${Math.round(c.sx - labW[c.t] / 2)}px, ${Math.round(c.sy - bh / 2)}px)`;
      l.style.opacity = (sel.size && !sel.has(c.t) ? 0.38 : 1) * c.fade;
    }
  }

  /* ---- rings */
  function updateRings() {
    let r = 0;
    const put = (i, pad, thick, rgb) => { ringPos.array.set(cur.subarray(i * 3, i * 3 + 3), r * 3); ringSize.array[r] = Math.max(aSize.array[i], 1); ringPad.array[r] = pad; ringThick.array[r] = thick; ringCol.array.set(rgb, r * 3); r++; };
    if (st.focus >= 0) put(st.focus, 14, 2.4, TH.accent);
    if (st.hover >= 0 && st.hover !== st.focus) put(st.hover, 8, 1.6, TH.ink);
    ringPos.needsUpdate = ringSize.needsUpdate = ringPad.needsUpdate = ringThick.needsUpdate = ringCol.needsUpdate = true;
    ringGeom.setDrawRange(0, r); needsRender = true;
  }

  /* ---- camera: the free area between the panels, fitting, flights */
  const view = { x: 0, y: 0, tx: 0, ty: 0 };
  const railOpen = () => !rail.classList.contains('collapsed');
  function freeRect() { // the part of the stage no panel covers: on a wide screen between the panels, on a phone between the bars
    const W = Math.max(1, box.clientWidth), Hh = Math.max(1, box.clientHeight), b = box.getBoundingClientRect(); let l = 0, r = W, t = 0, bt = Hh;
    if (narrow.matches) {
      const s = $('summary', tpanel).getBoundingClientRect(), fr = $('.mr-find', rail).getBoundingClientRect();
      if (s.height) t = s.bottom - b.top + 4;
      if (fr.height) bt = fr.top - b.top - 6;
      if (bt - t < Hh * 0.5) { t = 0; bt = Hh; }
    } else if (W >= 900) {
      const rr = rail.getBoundingClientRect(), tr = tpanel.getBoundingClientRect();
      if (rr.width && railOpen()) l = rr.right - b.left + 8;
      if (tr.width && tpanel.open) r = tr.left - b.left - 8;
      if (r - l < W * 0.45) { l = 0; r = W; }
    }
    return { l, r, t, b: bt, W, H: Hh };
  }
  function applyView() {
    const W = Math.max(1, box.clientWidth), Hh = Math.max(1, box.clientHeight);
    camera.aspect = W / Hh;
    if (Math.abs(view.x) < 0.5 && Math.abs(view.y) < 0.5) camera.clearViewOffset(); else camera.setViewOffset(W, Hh, view.x, view.y, W, Hh);
    camera.updateProjectionMatrix(); projDirty = true; needsRender = true;
  }
  function updateViewTarget(now) { const f = freeRect(); view.tx = f.W / 2 - (f.l + f.r) / 2; view.ty = f.H / 2 - (f.t + f.b) / 2; if (now || reduceMotion.matches) { view.x = view.tx; view.y = view.ty; } applyView(); }
  function applySize() { U.uSize.value = 0.085 * st.sizeK * st.autoK; U.uMaxPx.value = Math.min(15 * Math.max(1, st.sizeK), maxPx / DPR); projDirty = true; needsRender = true; }
  function resize() {
    const w = box.clientWidth, h = box.clientHeight; if (!w || !h) return;
    renderer.setSize(w, h, false);
    U.uScale.value = h / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    bgMat.uniforms.uRes.value.set(w, h); updateViewTarget(true);
    // a narrow stage shows the map smaller, so its dots grow a little to keep the picture as dense as on a wide one
    const f = freeRect(); st.autoK = Math.min(1.45, Math.max(1, Math.sqrt(780 / Math.max(1, Math.min(f.r - f.l, f.b - f.t))))); applySize();
    if (atHome && !flight && !morph && mix % 1 === 0) { const v = homeView(st.dim); camera.position.copy(v.pos); controls.target.copy(v.target); controls.update(); }
  }
  // look along dir and move until the lo..hi percentile box of the set fills the free area
  function fitView(list, P, dir, lo, hi, fill, minD = 1.2) {
    const cam = camera.clone(); cam.clearViewOffset();
    const f = freeRect(), fx = (f.r - f.l) / f.W, fy = (f.b - f.t) / f.H, target = new THREE.Vector3();
    for (const i of list) { target.x += P[i * 3]; target.y += P[i * 3 + 1]; target.z += P[i * 3 + 2]; }
    target.divideScalar(Math.max(1, list.length));
    const step = Math.max(1, Math.floor(list.length / 8000)), ds = [];
    for (let k = 0; k < list.length; k += step) { const i = list[k]; ds.push(Math.hypot(P[i * 3] - target.x, P[i * 3 + 1] - target.y, P[i * 3 + 2] - target.z)); }
    ds.sort((a, b) => a - b);
    const r95 = Math.max(ds[Math.floor(0.95 * (ds.length - 1))] || 0, 0.6), v0 = THREE.MathUtils.degToRad(camera.fov) / 2;
    const dS = r95 / Math.sin(Math.atan(Math.max(0.05, Math.min(Math.tan(v0) * fy, Math.tan(v0) * (f.W / f.H) * fx))));
    let d = dS; const mm = new THREE.Matrix4(), right = new THREE.Vector3(), up = new THREE.Vector3();
    for (let it = 0; it < 4; it++) {
      cam.position.copy(target).addScaledVector(dir, d); cam.lookAt(target); cam.updateMatrixWorld();
      mm.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
      const e = mm.elements, xs = [], ys = [];
      for (let k = 0; k < list.length; k += step) {
        const i = list[k], x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2], w = e[3] * x + e[7] * y + e[11] * z + e[15];
        if (w <= 0.05) continue;
        xs.push((e[0] * x + e[4] * y + e[8] * z + e[12]) / w); ys.push((e[1] * x + e[5] * y + e[9] * z + e[13]) / w);
      }
      if (xs.length < 2) break;
      xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
      const q = (a, fr) => a[Math.min(a.length - 1, Math.floor(fr * a.length))];
      const x1 = q(xs, lo), x2 = q(xs, hi), y1 = q(ys, lo), y2 = q(ys, hi);
      const halfH = d * Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2), halfW = halfH * cam.aspect;
      right.setFromMatrixColumn(cam.matrixWorld, 0); up.setFromMatrixColumn(cam.matrixWorld, 1);
      target.addScaledVector(right, ((x1 + x2) / 2) * halfW).addScaledVector(up, ((y1 + y2) / 2) * halfH);
      const extent = Math.max((x2 - x1) / 2 / fx, (y2 - y1) / 2 / fy, 0.02);
      d = Math.min(Math.max((d * extent) / fill, minD), Math.max(1.6 * dS, minD), 300);
    }
    return { target, pos: target.clone().addScaledVector(dir, d) };
  }
  function homeView(dim) {
    const P = dim === '2' ? pos2 : pos3, az = THREE.MathUtils.degToRad(32), el = THREE.MathUtils.degToRad(18);
    const dir = dim === '2' ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
    const list = [], step = Math.max(1, Math.floor(N / 12000)); for (let i = 0; i < N; i += step) list.push(i);
    return fitView(list, P, dir, 0.006, 0.994, 0.9);
  }
  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const qId = new THREE.Quaternion(), qTmp = new THREE.Quaternion(), vTmp = new THREE.Vector3();
  // a flight eases the target along a line, turns the view direction on the sphere and scales the distance
  // geometrically; a long trip rises a little in the middle, so the reader keeps the overview while travelling
  function flyTo(toT, toP, ms = 900, home = false) {
    atHome = home;
    const fromT = controls.target.clone(), o0 = camera.position.clone().sub(fromT), o1 = toP.clone().sub(toT);
    const d0 = Math.max(o0.length(), 1e-3), d1 = Math.max(o1.length(), 1e-3), u0 = o0.divideScalar(d0), u1 = o1.divideScalar(d1);
    const travel = fromT.distanceTo(toT), lift = Math.min(0.7, travel / (d0 + d1));
    const dur = reduceMotion.matches ? 0 : ms * Math.min(1.5, Math.max(0.8, 0.8 + 0.22 * Math.log2(1 + travel / Math.min(d0, d1) + Math.abs(Math.log2(d1 / d0)))));
    flight = { t0: performance.now(), ms: dur, fromT, toT: toT.clone(), d0, d1, u0, rot: new THREE.Quaternion().setFromUnitVectors(u0, u1), lift };
    needsRender = true;
  }
  function stepFlight(now) {
    const f = flight, t = f.ms ? Math.min(1, (now - f.t0) / f.ms) : 1, k = ease(t);
    controls.target.lerpVectors(f.fromT, f.toT, k);
    qTmp.copy(qId).slerp(f.rot, k); vTmp.copy(f.u0).applyQuaternion(qTmp);
    const d = Math.exp(Math.log(f.d0) + (Math.log(f.d1) - Math.log(f.d0)) * k) * (1 + f.lift * Math.sin(Math.PI * k));
    camera.position.copy(controls.target).addScaledVector(vTmp, d);
    if (t >= 1) flight = null; needsRender = true; projDirty = true;
  }
  function flyToPoint(i, dist) {
    const p = st.dim === '2' ? new THREE.Vector3(pos2[i * 3], pos2[i * 3 + 1], 0) : new THREE.Vector3(pos3[i * 3], pos3[i * 3 + 1], pos3[i * 3 + 2]);
    const dir = camera.position.clone().sub(controls.target).normalize();
    flyTo(p, p.clone().addScaledVector(dir, dist ?? Math.min(camera.position.distanceTo(controls.target), st.dim === '2' ? 8 : 6)));
  }
  function flyToSet(ix) {
    const list = ix.filter((i) => aSize.array[i] > 0); const use = list.length ? list : ix;
    if (!use.length) return; if (use.length === 1) { flyToPoint(use[0]); return; }
    const v = fitView(use, st.dim === '2' ? pos2 : pos3, camera.position.clone().sub(controls.target).normalize(), 0.03, 0.97, 0.8, 7); flyTo(v.target, v.pos);
  }
  const firstTouch = () => { if (!hint.classList.contains('gone')) setTimeout(() => { hint.classList.add('gone'); needsRender = true; }, 2500); };
  controls.addEventListener('start', () => { flight = null; atHome = false; hideTip(); firstTouch(); });
  controls.addEventListener('change', () => { needsRender = true; projDirty = true; });

  /* ---- 3D / 2D: the dots glide between the two layouts */
  function setPositions(k) {
    mix = k; for (let j = 0; j < N * 3; j++) cur[j] = pos3[j] + (pos2[j] - pos3[j]) * k;
    posAttr.needsUpdate = true; U.uFogAmt.value = FOG * (1 - k); projDirty = true; needsRender = true; updateRings();
  }
  function configureControls(d) {
    const three = d === '3';
    controls.enableRotate = three; controls.mouseButtons.LEFT = three ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN; controls.touches.ONE = three ? THREE.TOUCH.ROTATE : THREE.TOUCH.PAN;
    controls.autoRotate = st.spin && three; $('#map-spin').disabled = !three;
    hint.textContent = coarse ? (three ? 'drag to turn · pinch to zoom · tap a dot' : 'drag to move · pinch to zoom · tap a dot')
      : three ? 'drag to turn · scroll to zoom · click a dot to read it' : 'drag to move · scroll to zoom · click a dot to read it';
  }
  function drawDimSeg() { seg($('#seg-dim'), [['3', '3D'], ['2', '2D']], st.dim, setDim); }
  function setDim(d) {
    if (d === st.dim) return; st.dim = d; configureControls(d); drawDimSeg();
    morph = { t0: performance.now(), ms: reduceMotion.matches ? 0 : 950, from: mix, to: d === '2' ? 1 : 0 };
    const h = homeView(d); flyTo(h.target, h.pos, 950, true);
  }
  function goHome() { const h = homeView(st.dim); flyTo(h.target, h.pos, 900, true); }
  function resetAll() { // the opening view: every topic, no search, no pin; the colour stays
    st.topics.clear(); st.only = -1; st.preview = -1; setFocus(-1); hideTip();
    if (st.q) setQuery(''); $('#map-tq').value = ''; $('#map-tq').dispatchEvent(new Event('input'));
    drawLegend(); syncTopics(); goHome();
  }

  /* ---- picking in screen space: every dot projected once per camera move, then the nearest disc under the pointer */
  const proj = new Float32Array(N * 4);
  function project() {
    camera.updateMatrixWorld(); pm.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const m = pm.elements, W = box.clientWidth, Hh = box.clientHeight, sz = aSize.array, us = U.uSize.value * U.uScale.value, mn = U.uMinPx.value, mxp = U.uMaxPx.value;
    for (let i = 0; i < N; i++) {
      const o = i * 4; if (sz[i] === 0) { proj[o + 2] = -1; continue; }
      const x = cur[i * 3], y = cur[i * 3 + 1], z = cur[i * 3 + 2], w = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (w <= 0.01) { proj[o + 2] = -1; continue; }
      const sx = ((m[0] * x + m[4] * y + m[8] * z + m[12]) / w * 0.5 + 0.5) * W, sy = (0.5 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / w * 0.5) * Hh;
      if (sx < -30 || sy < -30 || sx > W + 30 || sy > Hh + 30) { proj[o + 2] = -1; continue; }
      proj[o] = sx; proj[o + 1] = sy; proj[o + 2] = w; proj[o + 3] = Math.min(Math.max((sz[i] * us) / w, mn), mxp) / 2;
    }
    projDirty = false;
  }
  function pickAt(x, y, tol) { // full dots before faded ones, then the one nearest the camera
    if (projDirty) project();
    const al = aAlpha.array; let best = -1, bestD = Infinity, bestStrong = false;
    for (let i = 0; i < N; i++) {
      const o = i * 4, d = proj[o + 2]; if (d < 0) continue;
      const dx = proj[o] - x, dy = proj[o + 1] - y, r = Math.max(proj[o + 3], 2.5) + tol;
      if (dx * dx + dy * dy > r * r) continue;
      const strong = al[i] === 255;
      if ((strong && !bestStrong) || (strong === bestStrong && d < bestD)) { best = i; bestD = d; bestStrong = strong; }
    }
    return best;
  }
  const localXY = (e) => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };

  /* ---- hover card */
  function showTip(i, x, y) {
    tip.textContent = ''; const t = T[i];
    tip.append(H('div', { class: 'tt-t', text: D.m.label[i] }),
      H('div', { class: 'tt-r' }, H('span', {}, H('i', { class: 'tsw', style: `--lc:${rgbStr(TH.topic[t])}` }), tshort(t)), pill(i), H('span', { class: 'mono', text: `ensemble ${f3(D.score.s[i])}` })),
      H('div', { class: 'tt-p', text: trunc(D.p.title[D.m.paper[i]] || D.p.id[D.m.paper[i]] || '', 120) }));
    tip.hidden = false;
    const W = box.clientWidth, Hh = box.clientHeight, w = tip.offsetWidth, h = tip.offsetHeight;
    let left = x + 16, top = y + 16;
    if (left + w > W - 8) left = Math.max(8, x - w - 16);
    if (top + h > Hh - 8) top = Math.max(8, y - h - 16);
    tip.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  }
  function hideTip() { tip.hidden = true; }
  function setHover(i, x, y) {
    canvas.classList.toggle('over-dot', i >= 0);
    if (i >= 0) showTip(i, x, y); else hideTip();
    if (i !== st.hover) { st.hover = i; updateRings(); }
  }
  canvas.addEventListener('pointerdown', (e) => {
    down = { x: e.clientX, y: e.clientY, t: performance.now(), type: e.pointerType, button: e.button };
    if (narrow.matches) { if (tpanel.open) tpanel.open = false; if (railOpen()) setRail(false); } // on a phone the map comes first: a touch folds the sheets away
  });
  canvas.addEventListener('pointermove', (e) => {
    if (e.pointerType === 'touch') return;
    if (e.buttons) { if (st.hover >= 0) setHover(-1); return; }
    pendingHover = localXY(e);
  });
  canvas.addEventListener('pointerleave', () => { pendingHover = null; setHover(-1); });
  canvas.addEventListener('pointerup', (e) => {
    const d = down; down = null; if (!d) return;
    const touch = d.type === 'touch';
    if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > (touch ? 10 : 5) || performance.now() - d.t > 600 || d.button !== 0) return;
    const [x, y] = localXY(e), i = pickAt(x, y, touch ? 9 : 3);
    if (i >= 0) { hideTip(); setFocus(i); openMention(i); } else if (st.focus >= 0) setFocus(-1);
  });
  canvas.addEventListener('dblclick', (e) => { const [x, y] = localXY(e), i = pickAt(x, y, 3); if (i >= 0) { setFocus(i); flyToPoint(i); } });
  function setFocus(i) { st.focus = i; pinW = 0; if (i >= 0) { pin.firstChild.textContent = D.m.label[i]; pin.title = `${D.m.label[i]}: click to read it`; } updateRings(); }
  function focusOn(i) { if (!(i >= 0 && i < N)) return; setFocus(i); flyToPoint(i); }

  /* ---- find a label: a substring lights up every mention whose label holds it ("…" = the exact label); the list under
     the box offers the concepts whose names hold it, largest first, and lights up one concept's mentions */
  const qIn = $('#map-q'), sug = $('#map-sugg');
  let qTimer = 0, sugg = [], sAt = -1;
  function writeQn(text, g) {
    const qn = $('#map-qn'); qn.textContent = ''; if (!text) return;
    qn.append(H('span', {}, H('span', { text }), nMatch ? ' · ' : null, nMatch ? H('button', { class: 'linkbtn', type: 'button', text: 'fly there', onclick: flyToMatches }) : null,
      g >= 0 ? ' · ' : null, g >= 0 ? H('button', { class: 'linkbtn', type: 'button', text: 'open the concept', onclick: () => { show('concepts'); if (TABS.concepts && TABS.concepts.open) TABS.concepts.open(g); } }) : null));
  }
  function setQuery(v, fly) {
    qIn.value = v; closeSugg();
    let q = v.trim().toLowerCase(); const exact = q.length > 2 && q[0] === '"' && q[q.length - 1] === '"'; if (exact) q = q.slice(1, -1).trim();
    st.q = q; st.qg = -1; nMatch = 0;
    for (let i = 0; i < N; i++) { const m = q ? (exact ? lower[i] === q : lower[i].includes(q)) : false; match[i] = m ? 1 : 0; if (m) nMatch++; }
    writeQn(q ? `${nf(nMatch)} mention${nMatch === 1 ? '' : 's'} ${exact ? 'with exactly this label' : 'match'}` : '', -1);
    refresh();
    if (fly && nMatch) flyToMatches();
  }
  function pickConcept(g) {
    qIn.value = D.g.name[g]; closeSugg();
    st.q = 'concept'; st.qg = g; match.fill(0); nMatch = 0; for (const i of D.members[g]) { match[i] = 1; nMatch++; }
    writeQn(`${nf(nMatch)} mention${nMatch === 1 ? '' : 's'} of the concept`, g);
    refresh(); flyToMatches();
  }
  function closeSugg() { sug.hidden = true; sug.textContent = ''; sugg = []; sAt = -1; qIn.setAttribute('aria-expanded', 'false'); qIn.removeAttribute('aria-activedescendant'); }
  function suggest(v) {
    closeSugg(); const q = v.trim().toLowerCase();
    if (q.length < 2 || q[0] === '"') return;
    const pre = [], mid = [];
    for (let g = 0; g < gLower.length; g++) { const k = gLower[g].indexOf(q); if (k === 0) pre.push(g); else if (k > 0) mid.push(g); }
    const by = (a, b) => D.g.n[b] - D.g.n[a] || gLower[a].length - gLower[b].length;
    sugg = [...pre.sort(by).slice(0, 6), ...mid.sort(by).slice(0, 6)].slice(0, 6);
    sugg.forEach((g, k) => sug.append(H('li', { role: 'option', id: `map-sugg-${k}`, 'aria-selected': 'false', onpointerdown: (e) => e.preventDefault(), onclick: () => pickConcept(g) },
      H('span', { text: D.g.name[g] }), H('b', { class: 'mono', text: nf(D.g.n[g]) }))));
    sug.hidden = !sugg.length; qIn.setAttribute('aria-expanded', String(!!sugg.length));
  }
  function markSugg(k) { sAt = k; $$('li', sug).forEach((li, j) => li.setAttribute('aria-selected', String(j === k))); if (k >= 0) qIn.setAttribute('aria-activedescendant', `map-sugg-${k}`); else qIn.removeAttribute('aria-activedescendant'); }
  function flyToMatches() { const ix = []; for (let i = 0; i < N; i++) if (match[i]) ix.push(i); flyToSet(ix); }
  qIn.addEventListener('input', () => { clearTimeout(qTimer); qTimer = setTimeout(() => { const v = qIn.value; setQuery(v); suggest(v); }, 140); });
  qIn.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' && sugg.length) { e.preventDefault(); markSugg((sAt + 1) % sugg.length); }
    else if (e.key === 'ArrowUp' && sugg.length) { e.preventDefault(); markSugg(sAt <= 0 ? sugg.length - 1 : sAt - 1); }
    else if (e.key === 'Enter') { clearTimeout(qTimer); if (sAt >= 0) pickConcept(sugg[sAt]); else setQuery(qIn.value, true); }
    else if (e.key === 'Escape') { if (!sug.hidden) { e.preventDefault(); e.stopPropagation(); closeSugg(); } }
  });
  qIn.addEventListener('blur', () => setTimeout(closeSugg, 120));

  /* ---- the rail and the other controls */
  function setRail(open) {
    rail.classList.toggle('collapsed', !open); $('#map-rail-tog').setAttribute('aria-expanded', String(open));
    if (open && narrow.matches) tpanel.open = false;
    updateViewTarget(false); needsRender = true;
  }
  $('#map-rail-tog').append(ico('sliders')); $('#map-home').append(ico('reset'));
  $('#map-rail-tog').addEventListener('click', () => setRail(!railOpen()));
  $('#map-home').addEventListener('click', resetAll);
  drawDimSeg(); drawColorSeg();
  $('#map-spin').addEventListener('click', (e) => { st.spin = !st.spin; e.currentTarget.setAttribute('aria-pressed', String(st.spin)); controls.autoRotate = st.spin && st.dim === '3'; needsRender = true; });
  const setNames = (on) => { st.names = on; $('#map-names').setAttribute('aria-pressed', String(on)); needsRender = true; };
  $('#map-names').addEventListener('click', () => setNames(!st.names));
  $('#map-size').addEventListener('input', (e) => { st.sizeK = +e.target.value; $('#map-sizeo').textContent = st.sizeK.toFixed(1) + '×'; applySize(); });
  tpanel.addEventListener('toggle', () => { if (tpanel.open && narrow.matches && railOpen()) setRail(false); updateViewTarget(false); needsRender = true; });
  if (narrow.matches) setRail(false);

  /* ---- keys, while the map is shown and no box has the focus */
  addEventListener('keydown', (e) => {
    if (sec.hidden || e.ctrlKey || e.metaKey || e.altKey || !$('#drawer').hidden || !popEl.hidden) return;
    const tg = e.target; if (tg && (/^(INPUT|TEXTAREA|SELECT)$/.test(tg.tagName) || tg.isContentEditable)) return;
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (k === '/') { e.preventDefault(); qIn.focus(); qIn.select(); }
    else if (k === 'r') resetAll();
    else if (k === '2' || k === '3') setDim(k);
    else if (k === 'n') setNames(!st.names);
    else if (k === 't') tpanel.open = !tpanel.open;
    else if (k === 'c') setRail(!railOpen());
    else if (k === 'Escape' && (st.q || st.topics.size || st.only >= 0)) { if (st.q) setQuery(''); st.topics.clear(); st.only = -1; drawLegend(); syncTopics(); }
  }, true);

  /* ---- theme changes (the stage stays dark; the swatches under it follow the theme) */
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => setTimeout(applyTheme, 0));
  new MutationObserver(() => setTimeout(applyTheme, 0)).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class'] });

  /* ---- the frame loop: draws only when something changed, and only while the tab is shown */
  function tick(now) {
    requestAnimationFrame(tick);
    if (sec.hidden || document.hidden) return;
    if (morph) {
      const t = morph.ms ? Math.min(1, (now - morph.t0) / morph.ms) : 1;
      setPositions(morph.from + (morph.to - morph.from) * ease(t)); if (t >= 1) morph = null;
    }
    if (flight) stepFlight(now);
    if (view.x !== view.tx || view.y !== view.ty) {
      view.x += (view.tx - view.x) * 0.2; view.y += (view.ty - view.y) * 0.2;
      if (Math.abs(view.tx - view.x) < 0.5 && Math.abs(view.ty - view.y) < 0.5) { view.x = view.tx; view.y = view.ty; }
      applyView();
    }
    if (controls.update()) { needsRender = true; projDirty = true; }
    if (needsRender) {
      const dC = camera.position.distanceTo(controls.target); U.uFogNear.value = dC - 4; U.uFogFar.value = dC + 13;
      renderer.render(scene, camera); needsRender = false; placeLabels();
    }
    if (pendingHover && !down && now - lastPick > 32) {
      lastPick = now; const [x, y] = pendingHover; pendingHover = null;
      const i = pickAt(x, y, 2); if (i >= 0) setHover(i, x, y); else if (st.hover >= 0 || !tip.hidden) setHover(-1);
    }
  }

  /* ---- go */
  TH = readTheme(); applyTheme(); configureControls('3');
  new ResizeObserver(resize).observe(box); resize();
  const h = homeView('3'); camera.position.copy(h.pos); controls.target.copy(h.target); controls.update();
  if (qIn.value) setQuery(qIn.value, true);
  Object.assign(MAP, {
    ready: true, setColor, isolate, setQuery,
    focus: focusOn,
    flyTopic: (t) => flyToSet(F.members[t]),
    resize: () => { layoutStage(); resize(); },
    show: () => { layoutStage(); resize(); drawLegend(); refresh(); },
    flyToLastConcept: () => { const g = [...MAPSEL].pop(); if (g != null) flyToSet(D.members[g]); },
  });
  $('#mapload').hidden = true; syncTopics();
  if (st.color === 'concept') MAP.flyToLastConcept();
  if (MAP.pending >= 0) { const i = MAP.pending; MAP.pending = -1; focusOn(i); }
  requestAnimationFrame(tick);
}

TABS.map = {
  init: () => { startMap().catch((e) => { console.error(e); $('#mapload').textContent = 'The map could not be drawn: ' + e.message; }); },
  onShow: () => { if (MAP.show) MAP.show(); else layoutStage(); },
  focus: (i) => { if (MAP.ready) MAP.focus(i); else MAP.pending = i; },
  showConcepts: () => { if (MAP.ready) { MAP.setColor('concept'); MAP.flyToLastConcept(); } else MAP.pendingColor = 'concept'; },
};
