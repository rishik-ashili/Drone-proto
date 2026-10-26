// Ground dashboard client: 3D twin (three.js), 2D map, charts and controls.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const COL = { true: css('--true'), est: css('--est'), dr: css('--dr'), err: css('--err'), warn: css('--warn') };
const FLOW = ['IDLE', 'PREFLIGHT', 'TAKEOFF', 'GOTO_WAYPOINT', 'SEARCH_LANDMARK', 'ALIGN', 'NEXT_WAYPOINT',
  'RETURN_TO_START', 'PRECISION_LAND', 'LAND', 'LANDED'];

let world = null;
let snap = null;
let trail = [];
let lastTrailT = -1;
let lastSnapT = -1;
let arenaImg = null;

// NED (x north, y east, z down) -> three.js (x east, y up, z south)
const v3 = (n, e, d) => new THREE.Vector3(e, -d, -n);

async function api(cmd, extra = {}) {
  const r = await fetch('/api/command', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cmd, ...extra }) });
  const j = await r.json();
  if (!j.ok) console.warn(j.error);
  return j;
}

async function loadWorld() {
  world = await (await fetch('/api/world')).json();
  arenaImg = new Image();
  arenaImg.src = '/api/arena.jpg?' + Date.now();
  $('detector').textContent = world.detector;
  const sel = $('mission');
  sel.innerHTML = world.missions.map((m) => `<option ${m === world.mission.name ? 'selected' : ''}>${m}</option>`).join('');
  buildFaultButtons();
  buildScene();
}

// ------------------------------------------------------------------ UI wiring
$('btnStart').onclick = () => api('start');
$('btnPause').onclick = () => {
  const paused = snap && snap.paused;
  api(paused ? 'resume' : 'pause');
};
$('btnReset').onclick = () => resetAll('reset');
$('speed').onchange = (e) => api('speed', { value: parseFloat(e.target.value) });
$('mission').onchange = (e) => resetAll('mission', { value: e.target.value });
$('correction').onchange = (e) => resetAll('correction', { value: e.target.checked });

async function resetAll(cmd, extra = {}) {
  await api(cmd, extra);
  trail = []; lastTrailT = -1;
  await loadWorld();
}

function buildFaultButtons() {
  const box = $('faults');
  box.innerHTML = '';
  for (const [key, label] of Object.entries(world.faults)) {
    const b = document.createElement('button');
    b.textContent = label;
    b.dataset.fault = key;
    b.onclick = () => api(b.classList.contains('on') ? 'clear' : 'inject', { fault: key });
    box.appendChild(b);
  }
}

// ------------------------------------------------------------------ 3D scene
let renderer, scene, camera, controls, drone, rotors = [], estGhost, drGhost, footprint, depthLines;
let trailLines = {};
let staticGroup;
const MAXPTS = 40000;

function makeLine(color, opacity = 1) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAXPTS * 3), 3));
  g.setDrawRange(0, 0);
  const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }));
  l.frustumCulled = false;
  return l;
}

function textSprite(text, color = '#e6edf3', scale = 1) {
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d');
  ctx.font = '600 44px system-ui, sans-serif';
  const w = Math.ceil(ctx.measureText(text).width) + 28;
  c.width = w; c.height = 64;
  ctx.font = '600 44px system-ui, sans-serif';
  ctx.fillStyle = 'rgba(10,14,19,0.78)';
  ctx.beginPath(); ctx.roundRect(0, 0, w, 64, 14); ctx.fill();
  ctx.fillStyle = color; ctx.textBaseline = 'middle';
  ctx.fillText(text, 14, 34);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
  s.scale.set(0.42 * w / 64 * scale, 0.42 * scale, 1);
  s.renderOrder = 10;
  return s;
}

function buildDrone() {
  const g = new THREE.Group();
  const dark = new THREE.MeshStandardMaterial({ color: 0x1d252e, metalness: 0.4, roughness: 0.5 });
  const accent = new THREE.MeshStandardMaterial({ color: 0x39b6ff, emissive: 0x0b3550, metalness: 0.2, roughness: 0.4 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.07, 0.24), dark);
  g.add(body);
  const top = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.03, 0.14), accent);
  top.position.y = 0.05; g.add(top);
  const L = 0.23;
  for (const [sx, sz] of [[1, -1], [-1, -1], [1, 1], [-1, 1]]) {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.025, 0.02, Math.hypot(L, L)), dark);
    arm.position.set(sx * L / 2, 0, sz * L / 2);
    arm.rotation.y = Math.atan2(sx, sz);
    g.add(arm);
    const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.04, 16), dark);
    motor.position.set(sx * L, 0.02, sz * L); g.add(motor);
    const rotor = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.004, 32),
      new THREE.MeshBasicMaterial({ color: sz < 0 ? 0x7fd3ff : 0xb9c7d6, transparent: true, opacity: 0.35 }));
    rotor.position.set(sx * L, 0.045, sz * L); g.add(rotor); rotors.push(rotor);
    const led = new THREE.Mesh(new THREE.SphereGeometry(0.015, 8, 8),
      new THREE.MeshBasicMaterial({ color: sz < 0 ? 0x3ddc84 : 0xff5d5d }));
    led.position.set(sx * L, -0.02, sz * L); g.add(led);
  }
  const cam = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.03, 12), accent);
  cam.position.set(0, -0.05, 0); g.add(cam);
  g.scale.setScalar(1.8);
  return g;
}

function initRenderer() {
  const el = $('three');
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  el.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d131a);
  scene.fog = new THREE.Fog(0x0d131a, 30, 70);
  camera = new THREE.PerspectiveCamera(50, 1, 0.05, 200);
  camera.position.set(-6, 12, 12);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.49;
  scene.add(new THREE.HemisphereLight(0xcfe8ff, 0x1a222b, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(8, 20, 6); scene.add(sun);
  drone = buildDrone(); scene.add(drone);
  estGhost = new THREE.Mesh(new THREE.SphereGeometry(0.09, 20, 20),
    new THREE.MeshBasicMaterial({ color: COL.est, transparent: true, opacity: 0.85 }));
  drGhost = new THREE.Mesh(new THREE.SphereGeometry(0.09, 20, 20),
    new THREE.MeshBasicMaterial({ color: COL.dr, transparent: true, opacity: 0.85 }));
  scene.add(estGhost, drGhost);
  trailLines = { true: makeLine(COL.true), est: makeLine(COL.est, 0.9), dr: makeLine(COL.dr, 0.75) };
  Object.values(trailLines).forEach((l) => scene.add(l));
  const fg = new THREE.BufferGeometry();
  fg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(5 * 3), 3));
  footprint = new THREE.Line(fg, new THREE.LineBasicMaterial({ color: 0x39b6ff, transparent: true, opacity: 0.8 }));
  footprint.frustumCulled = false;
  scene.add(footprint);
  const dg = new THREE.BufferGeometry();
  dg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(64 * 2 * 3), 3));
  depthLines = new THREE.LineSegments(dg, new THREE.LineBasicMaterial({ color: 0xff5d5d, transparent: true, opacity: 0.55 }));
  depthLines.frustumCulled = false;
  scene.add(depthLines);
  new ResizeObserver(resize).observe(el);
  resize();
}

function resize() {
  const el = $('three');
  const w = el.clientWidth, h = el.clientHeight;
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function buildScene() {
  if (!renderer) initRenderer();
  if (staticGroup) scene.remove(staticGroup);
  staticGroup = new THREE.Group();
  const [xmin, xmax, ymin, ymax] = world.world.bounds;
  const tex = new THREE.TextureLoader().load('/api/arena.jpg?' + Date.now());
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(ymax - ymin, xmax - xmin),
    new THREE.MeshStandardMaterial({ map: tex, roughness: 1 }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.copy(v3((xmin + xmax) / 2, (ymin + ymax) / 2, 0));
  staticGroup.add(ground);
  // arena walls (outline)
  const edge = [v3(xmin, ymin, 0), v3(xmin, ymax, 0), v3(xmax, ymax, 0), v3(xmax, ymin, 0), v3(xmin, ymin, 0)];
  staticGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(edge),
    new THREE.LineBasicMaterial({ color: 0x33475a })));
  for (const o of world.world.obstacles) {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(o.radius, o.radius, o.height, 32),
      new THREE.MeshStandardMaterial({ color: 0x6b737c, roughness: 0.8 }));
    m.position.copy(v3(o.x, o.y, -o.height / 2));
    staticGroup.add(m);
    const s = textSprite(o.name, '#b9c7d6', 0.8);
    s.position.copy(v3(o.x, o.y, -o.height - 0.3));
    staticGroup.add(s);
  }
  const alt = world.mission.altitude_m;
  const route = [v3(0, 0, -alt)];
  world.mission.waypoints.forEach((w) => {
    route.push(v3(w.x, w.y, -alt));
    const pole = new THREE.Line(new THREE.BufferGeometry().setFromPoints([v3(w.x, w.y, 0), v3(w.x, w.y, -alt)]),
      new THREE.LineDashedMaterial({ color: 0xe6edf3, dashSize: 0.15, gapSize: 0.1, transparent: true, opacity: 0.6 }));
    pole.computeLineDistances();
    staticGroup.add(pole);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.35, 0.02, 8, 40),
      new THREE.MeshBasicMaterial({ color: 0xe6edf3, transparent: true, opacity: 0.7 }));
    ring.rotation.x = Math.PI / 2;
    ring.position.copy(v3(w.x, w.y, -alt));
    staticGroup.add(ring);
    const s = textSprite(`WP${w.id}${w.landmark ? ' · ' + w.landmark : ''}`);
    s.position.copy(v3(w.x, w.y, -alt - 0.55));
    staticGroup.add(s);
  });
  if (world.mission.on_finish === 'return_to_start') route.push(v3(0, 0, -alt));
  const rl = new THREE.Line(new THREE.BufferGeometry().setFromPoints(route),
    new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.3, gapSize: 0.2, transparent: true, opacity: 0.35 }));
  rl.computeLineDistances();
  staticGroup.add(rl);
  const home = textSprite('HOME', '#9ff0c3');
  home.position.copy(v3(0, 0, -0.6));
  staticGroup.add(home);
  scene.add(staticGroup);
  for (const l of Object.values(trailLines)) l.geometry.setDrawRange(0, 0);
}

function setTrailLine(line, idx) {
  const arr = line.geometry.attributes.position.array;
  const n = Math.min(trail.length, MAXPTS);
  const off = trail.length - n;
  for (let i = 0; i < n; i++) {
    const r = trail[off + i];
    const p = v3(r[idx], r[idx + 1], idx === 7 ? -(r[3]) : -r[idx + 2]);
    arr[i * 3] = p.x; arr[i * 3 + 1] = p.y; arr[i * 3 + 2] = p.z;
  }
  line.geometry.attributes.position.needsUpdate = true;
  line.geometry.setDrawRange(0, n);
}

function rotNED(roll, pitch, yaw) {
  const cr = Math.cos(roll), sr = Math.sin(roll), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  return [
    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
    [-sp, cp * sr, cp * cr]];
}

function cameraFootprint(pos, att) {
  // project the 4 image corners onto the ground (same camera model as the simulator)
  const c = world.camera;
  const f = (c.width / 2) / Math.tan((c.hfov_deg * Math.PI / 180) / 2);
  const R = rotNED(att[0], att[1], att[2]);
  const pts = [];
  for (const [u, v] of [[0, 0], [c.width, 0], [c.width, c.height], [0, c.height], [0, 0]]) {
    const dc = [(u - c.width / 2) / f, (v - c.height / 2) / f, 1];
    const db = [-dc[1], dc[0], dc[2]]; // camera -> body (FRD)
    const dw = [0, 1, 2].map((i) => R[i][0] * db[0] + R[i][1] * db[1] + R[i][2] * db[2]);
    if (dw[2] <= 0.05) return null;
    const s = -pos[2] / dw[2];
    pts.push([pos[0] + s * dw[0], pos[1] + s * dw[1]]);
  }
  return pts;
}

function update3D() {
  if (!snap || !renderer) return;
  const tp = snap.true ? snap.true.pos : snap.est.pos;
  const ta = snap.true ? snap.true.att : snap.est.att;
  drone.position.copy(v3(tp[0], tp[1], tp[2]));
  drone.rotation.set(ta[1], -ta[2], -ta[0], 'YXZ');
  estGhost.position.copy(v3(...snap.est.pos));
  if (snap.dr) drGhost.position.copy(v3(...snap.dr.pos));
  drGhost.visible = $('showDr').checked && !!snap.dr && snap.state !== 'IDLE';
  trailLines.dr.visible = $('showDr').checked;
  estGhost.visible = snap.state !== 'IDLE';
  const fp = -tp[2] > 0.2 ? cameraFootprint(tp, ta) : null;
  footprint.visible = !!fp;
  if (fp) {
    const a = footprint.geometry.attributes.position.array;
    fp.forEach((p, i) => { const q = v3(p[0], p[1], -0.02); a[i * 3] = q.x; a[i * 3 + 1] = q.y; a[i * 3 + 2] = q.z; });
    footprint.geometry.attributes.position.needsUpdate = true;
  }
  const da = depthLines.geometry.attributes.position.array;
  let n = 0;
  if (snap.depth && snap.planner.active) {
    snap.depth.angles.forEach((ang, i) => {
      const r = snap.depth.ranges[i];
      if (r >= 5.9) return;
      const th = ta[2] + ang;
      const a = v3(tp[0], tp[1], tp[2]), b = v3(tp[0] + r * Math.cos(th), tp[1] + r * Math.sin(th), tp[2]);
      da.set([a.x, a.y, a.z, b.x, b.y, b.z], n * 6); n++;
    });
  }
  depthLines.geometry.setDrawRange(0, n * 2);
  depthLines.geometry.attributes.position.needsUpdate = true;
}

let lastFrame = performance.now();
function animate(now) {
  const dt = (now - lastFrame) / 1000; lastFrame = now;
  if (snap && snap.armed) rotors.forEach((r, i) => { r.rotation.y += (i % 2 ? 1 : -1) * dt * 40; });
  if ($('follow').checked && snap) {
    const target = drone.position.clone();
    const delta = target.clone().sub(controls.target);
    controls.target.add(delta.multiplyScalar(0.08));
    camera.position.add(delta);
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

// ------------------------------------------------------------------ 2D map
function drawMap() {
  const cv = $('map');
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!world) return;
  const [xmin, xmax, ymin, ymax] = world.world.bounds;
  const s = Math.min(W / (ymax - ymin), H / (xmax - xmin));
  const ox = (W - s * (ymax - ymin)) / 2, oy = (H - s * (xmax - xmin)) / 2;
  const P = (n, e) => [ox + (e - ymin) * s, oy + (xmax - n) * s];
  if (arenaImg && arenaImg.complete) {
    ctx.globalAlpha = 0.85;
    ctx.drawImage(arenaImg, ox, oy, s * (ymax - ymin), s * (xmax - xmin));
    ctx.globalAlpha = 1;
  }
  // planned route
  ctx.setLineDash([6, 5]); ctx.strokeStyle = 'rgba(255,255,255,.5)'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(...P(0, 0));
  world.mission.waypoints.forEach((w) => ctx.lineTo(...P(w.x, w.y)));
  if (world.mission.on_finish === 'return_to_start') ctx.lineTo(...P(0, 0));
  ctx.stroke(); ctx.setLineDash([]);
  // trails
  const drawTrail = (i, color, width) => {
    if (trail.length < 2) return;
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
    trail.forEach((r, k) => { const p = P(r[i], r[i + 1]); k ? ctx.lineTo(...p) : ctx.moveTo(...p); });
    ctx.stroke();
  };
  if ($('showDr').checked) drawTrail(7, COL.dr, 1.5);
  drawTrail(4, COL.est, 1.5);
  drawTrail(1, COL.true, 2);
  // waypoints
  ctx.font = '600 11px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  world.mission.waypoints.forEach((w, i) => {
    const [x, y] = P(w.x, w.y);
    const done = snap && i < snap.wp_index;
    ctx.fillStyle = done ? 'rgba(61,220,132,.9)' : (snap && i === snap.wp_index ? 'rgba(57,182,255,.95)' : 'rgba(10,14,19,.85)');
    ctx.strokeStyle = '#e6edf3'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = done ? '#06130c' : '#e6edf3'; ctx.fillText(String(w.id), x, y + 0.5);
  });
  if (!snap) return;
  if (snap.true) {
    const fp = cameraFootprint(snap.true.pos, snap.true.att);
    if (fp) {
      ctx.strokeStyle = 'rgba(57,182,255,.8)'; ctx.fillStyle = 'rgba(57,182,255,.08)'; ctx.lineWidth = 1;
      ctx.beginPath(); fp.forEach((p, k) => { const q = P(p[0], p[1]); k ? ctx.lineTo(...q) : ctx.moveTo(...q); });
      ctx.fill(); ctx.stroke();
    }
  }
  const dot = (pos, color, r) => { const [x, y] = P(pos[0], pos[1]); ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); };
  if (snap.dr && $('showDr').checked) dot(snap.dr.pos, COL.dr, 4.5);
  dot(snap.est.pos, COL.est, 4.5);
  const tp = snap.true ? snap.true.pos : snap.est.pos, yaw = snap.true ? snap.true.att[2] : snap.est.att[2];
  const [x, y] = P(tp[0], tp[1]);
  ctx.save(); ctx.translate(x, y); ctx.rotate(yaw);
  ctx.fillStyle = COL.true; ctx.strokeStyle = '#06130c'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(0, -11); ctx.lineTo(7, 8); ctx.lineTo(0, 4); ctx.lineTo(-7, 8); ctx.closePath();
  ctx.fill(); ctx.stroke(); ctx.restore();
  // scale bar
  ctx.fillStyle = '#e6edf3'; ctx.fillRect(ox + 10, oy + s * (xmax - xmin) - 14, 2 * s, 3);
  ctx.textAlign = 'left'; ctx.fillText('2 m', ox + 14 + 2 * s, oy + s * (xmax - xmin) - 12);
  ctx.textAlign = 'center'; ctx.fillText('N ↑', ox + s * (ymax - ymin) - 18, oy + 14);
}

// ------------------------------------------------------------------ error chart
function drawChart() {
  const cv = $('errChart');
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const pad = { l: 36, r: 8, t: 8, b: 18 };
  const pts = trail;
  const tmax = Math.max(20, pts.length ? pts[pts.length - 1][0] : 0);
  let ymax = 0.5;
  for (const r of pts) ymax = Math.max(ymax, r[9], $('showDr').checked ? r[10] : 0);
  ymax = Math.ceil(ymax * 4) / 4;
  const X = (t) => pad.l + (W - pad.l - pad.r) * t / tmax;
  const Y = (v) => H - pad.b - (H - pad.t - pad.b) * v / ymax;
  ctx.strokeStyle = '#212c38'; ctx.fillStyle = '#8a9aab'; ctx.font = '10px system-ui'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = ymax * i / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(v.toFixed(2) + ' m', pad.l - 4, y + 3);
  }
  ctx.textAlign = 'center';
  for (let t = 0; t <= tmax; t += tmax > 120 ? 30 : 10) ctx.fillText(t + 's', X(t), H - 4);
  const line = (i, color) => {
    if (pts.length < 2) return;
    ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.beginPath();
    pts.forEach((r, k) => (k ? ctx.lineTo(X(r[0]), Y(r[i])) : ctx.moveTo(X(r[0]), Y(r[i]))));
    ctx.stroke();
  };
  if ($('showDr').checked) line(10, COL.dr);
  line(9, COL.est);
}

// ------------------------------------------------------------------ panels
const fmt = (v, d = 2, u = ' m') => (v === null || v === undefined ? '—' : Number(v).toFixed(d) + u);

function updatePanels() {
  const s = snap;
  $('simTime').textContent = s.t.toFixed(1) + ' s';
  $('px4Mode').textContent = s.mode + (s.armed ? '' : '');
  const badge = $('stateBadge');
  badge.textContent = s.state.replaceAll('_', ' ');
  badge.className = 'state-badge' + (['FAILSAFE_HOLD', 'ABORTED'].includes(s.state) ? ' alert' : s.state === 'LANDED' ? ' done' : '');
  $('btnPause').textContent = s.paused ? 'Resume' : 'Pause';
  $('speed').value = String(s.speed);
  $('correction').checked = s.correction;
  const lastState = [...s.events].reverse().find((e) => e.type === 'state');
  if (lastState && s.state !== 'IDLE') $('caption').innerHTML = captionFor(s, lastState);

  // perception
  const p = s.perception;
  if (p) {
    $('pFps').textContent = p.fps.toFixed(1) + ' FPS';
    $('pMs').textContent = p.ms.toFixed(1) + ' ms';
    $('detector').textContent = p.detector;
    $('dets').innerHTML = p.observations.map((o) =>
      `<li class="${o.accepted === false ? 'rej' : o.confirmed ? 'conf' : ''}">${o.name} · ${o.src}` +
      ` · ${o.confirmed ? (o.accepted === false ? 'rejected by EKF gate' : 'fix ±' + (o.std * 100).toFixed(0) + ' cm') : 'tracking'}</li>`).join('') ||
      (p.detections.length ? p.detections.map((d) => `<li>${d.cls} · ${d.src} ${d.conf.toFixed(2)}</li>`).join('') : '<li class="muted">no landmark in view</li>');
  }
  const m = s.metrics;
  $('pLat').textContent = fmt(m.latency_ms_mean, 0, ' ms');
  $('pFix').textContent = `${m.fixes_accepted} / ${m.fixes_rejected}`;

  // accuracy KPIs
  $('kEst').textContent = fmt(m.est_error_now_m);
  $('kEstMax').textContent = m.est_error_mean_m != null ? `mean ${fmt(m.est_error_mean_m)} · max ${fmt(m.est_error_max_m)}` : '';
  $('kDr').textContent = fmt(m.dr_error_now_m);
  $('kDrMax').textContent = `max ${fmt(m.dr_error_max_m)}`;
  $('kReach').textContent = fmt(m.reach_error_mean_m);
  $('kReachMax').textContent = m.reach_error_max_m != null ? `max ${fmt(m.reach_error_max_m)} · ${m.waypoints_done}/${m.waypoints_total} WPs` : `${m.waypoints_done}/${m.waypoints_total} WPs`;
  $('kRts').textContent = fmt(m.rts_error_m);
  $('kRtsPct').textContent = m.rts_error_pct != null ? `${m.rts_error_pct.toFixed(2)} % of path` : '';
  $('kDist').textContent = fmt(m.distance_m, 1);
  $('kBat').textContent = m.battery_pct != null ? `battery ${m.battery_pct.toFixed(0)} %` + (m.collisions ? ` · ${m.collisions} collision(s)` : '') : '';
  const pl = $('planner');
  pl.textContent = s.planner.active ? `avoiding obstacle · ${fmt(s.planner.closest, 1)}` : 'avoidance idle';
  pl.className = 'pill' + (s.planner.active ? ' on' : '');

  // state machine
  const idx = FLOW.indexOf(s.state);
  $('states').innerHTML = FLOW.map((st, i) => `<span class="${st === s.state ? 'active' : i < idx ? 'done' : ''}">${st.replaceAll('_', ' ')}</span>`).join('') +
    `<span class="fs ${s.state === 'FAILSAFE_HOLD' ? 'active' : ''}">FAILSAFE HOLD</span>`;
  $('wpBody').innerHTML = world.mission.waypoints.map((w, i) => {
    const r = s.wp_results[i];
    let st = '<span class="muted">pending</span>';
    if (r) st = r.found ? '<span class="ok">✓ aligned</span>' : (w.landmark ? '<span class="warn">✗ not found</span>' : '<span class="ok">✓ reached</span>');
    else if (i === s.wp_index && !['IDLE', 'PREFLIGHT', 'TAKEOFF'].includes(s.state)) st = `<span>${s.state.replaceAll('_', ' ').toLowerCase()}</span>`;
    return `<tr class="${i === s.wp_index ? 'active' : ''}"><td>${w.id}</td><td>${w.landmark || '—'}</td><td>${st}</td><td>${r && r.reach_error_m != null ? r.reach_error_m.toFixed(2) + ' m' : ''}</td></tr>`;
  }).join('');

  // health
  const f = Object.fromEntries(s.faults.map((x) => [x.name, x]));
  const lvl = (name) => (f[name] ? (['hold', 'rts', 'land'].includes(f[name].action) ? 'bad' : 'warn') : '');
  const rows = [
    ['Battery', `${s.battery.toFixed(0)} % · ${s.voltage.toFixed(1)} V`, lvl('battery')],
    ['Companion link', f.link ? f.link.msg : 'OK', lvl('link')],
    ['Camera', f.camera ? f.camera.msg : (p ? `OK · ${p.fps.toFixed(0)} FPS` : '—'), lvl('camera')],
    ['Position σ (EKF)', fmt(s.est.std), lvl('position_quality')],
    ['Optical flow quality', `${s.flow_quality} / 255`, s.flow_quality < 60 && s.armed ? 'warn' : ''],
    ['Geofence', f.geofence ? 'BREACH' : 'inside', lvl('geofence')],
    ['Rangefinder', fmt(s.range), ''],
    ['PX4 failsafe', f.vehicle_failsafe ? f.vehicle_failsafe.msg.replace('PX4 failsafe: ', '') : 'none', lvl('vehicle_failsafe')],
  ];
  $('health').innerHTML = rows.map(([k, v, l]) => `<div class="hrow"><i class="dot ${l}"></i><div><span>${k}</span><b title="${v}">${v}</b></div></div>`).join('');

  for (const b of $('faults').querySelectorAll('button')) b.classList.toggle('on', s.injections.includes(b.dataset.fault));

  $('events').innerHTML = [...s.events].reverse().map((e) =>
    `<li class="${e.type}"><time>${e.t.toFixed(1)}s</time><span>${e.msg}</span></li>`).join('');
}

function captionFor(s, ev) {
  const wp = world.mission.waypoints[s.wp_index];
  const lm = wp && wp.landmark ? `<b>${wp.landmark}</b>` : 'the waypoint';
  const map = {
    PREFLIGHT: 'Pre-flight checks: estimator ready, streaming setpoints, arming…',
    TAKEOFF: 'Taking off in Offboard mode — position from optical flow + IMU, <b>no GPS</b>.',
    GOTO_WAYPOINT: `Flying to WP${wp ? wp.id : ''} on onboard estimate only; avoiding obstacles with the depth sensor.`,
    SEARCH_LANDMARK: `Searching for ${lm} with an expanding square pattern.`,
    ALIGN: `AI detected ${lm} — visual servoing over it; each fix corrects accumulated drift.`,
    NEXT_WAYPOINT: 'Waypoint confirmed. Logging reach error, moving on.',
    RETURN_TO_START: 'Mission complete — returning to the takeoff point in the local frame.',
    PRECISION_LAND: 'Home pad detected — precision landing using vision.',
    FAILSAFE_HOLD: `<span class="bad">Failsafe:</span> ${ev.msg.split('(').slice(1).join('(').replace(/\)$/, '') || 'holding position'}`,
    LAND: 'Landing.',
    LANDED: `Landed. Return-to-start error <b>${fmt(s.metrics.rts_error_m)}</b> after ${fmt(s.metrics.distance_m, 1)} flown.`,
    ABORTED: `<span class="bad">Aborted:</span> ${ev.msg}`,
  };
  return map[s.state] || ev.msg;
}

// ------------------------------------------------------------------ data feeds
async function pollTrail() {
  try {
    const j = await (await fetch('/api/trail?since=' + lastTrailT)).json();
    if (j.points.length) {
      trail.push(...j.points);
      lastTrailT = trail[trail.length - 1][0];
      setTrailLine(trailLines.true, 1);
      setTrailLine(trailLines.est, 4);
      setTrailLine(trailLines.dr, 7);
    }
  } catch (e) { /* server restarting */ }
}

function connect() {
  const es = new EventSource('/api/stream');
  es.onmessage = (ev) => {
    const s = JSON.parse(ev.data);
    if (!s.est) return;
    if (s.t + 0.5 < lastSnapT) { trail = []; lastTrailT = -1; } // sim was reset
    lastSnapT = s.t;
    snap = s;
    updatePanels();
    update3D();
  };
  es.onerror = () => { $('caption').textContent = 'Connection to the simulator lost — retrying…'; };
}

setInterval(pollTrail, 300);
setInterval(() => { drawMap(); drawChart(); }, 150);
await loadWorld();
connect();
requestAnimationFrame(animate);
