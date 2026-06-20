// Animated hero for the course overview: particles sampled from a Gaussian are
// transported along straight conditional paths into structured matter (a crystal
// lattice, then a molecule), dissolve back to noise, and loop. This is the
// flow-matching picture that the whole course is built on.
(function () {
  const root = document.getElementById("overview-fig");
  if (!root) return;
  const canvas = root.querySelector("canvas");
  const ctx = canvas.getContext("2d");

  const ACCENT = [179, 19, 42];
  const GRAY = [99, 99, 110];
  const N = 720;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- target geometries (world coords, roughly [-1.2, 1.2]) ----------
  function triangularLattice() {
    const pts = [], a = 0.34, rows = 5, cols = 7;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = (c - (cols - 1) / 2) * a + (r % 2 ? a / 2 : 0);
        const y = (r - (rows - 1) / 2) * a * 0.866;
        if (x * x + (y * 1.15) * (y * 1.15) < 1.15) pts.push([x, y]);
      }
    }
    const bonds = [];
    for (let i = 0; i < pts.length; i++)
      for (let j = i + 1; j < pts.length; j++) {
        const dx = pts[i][0] - pts[j][0], dy = pts[i][1] - pts[j][1];
        if (Math.hypot(dx, dy) < a * 1.08) bonds.push([i, j]);
      }
    return { points: pts, bonds, kind: "lattice" };
  }

  function molecule() {
    // a six-membered ring with outward substituent atoms
    const pts = [], bonds = [], rIn = 0.46, rOut = 0.86;
    for (let i = 0; i < 6; i++) {
      const t = (i / 6) * 2 * Math.PI + Math.PI / 6;
      pts.push([rIn * Math.cos(t), rIn * Math.sin(t)]);
    }
    for (let i = 0; i < 6; i++) {
      const t = (i / 6) * 2 * Math.PI + Math.PI / 6;
      pts.push([rOut * Math.cos(t), rOut * Math.sin(t)]);
    }
    for (let i = 0; i < 6; i++) { bonds.push([i, (i + 1) % 6]); bonds.push([i, i + 6]); }
    return { points: pts, bonds, kind: "molecule" };
  }

  const LATTICE = triangularLattice();
  const MOL = molecule();
  const NOISE = { points: null, bonds: [], kind: "noise" };
  const SCENES = [NOISE, LATTICE, NOISE, MOL];

  // ---------- particle buffers ----------
  const x0 = new Float32Array(N * 2);
  const x1 = new Float32Array(N * 2);

  function gauss() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function fillKeyframe(buf, scene) {
    if (scene.kind === "noise") {
      for (let i = 0; i < N; i++) { buf[2 * i] = 0.62 * gauss(); buf[2 * i + 1] = 0.62 * gauss(); }
    } else {
      const P = scene.points, M = P.length;
      for (let i = 0; i < N; i++) {
        const p = P[i % M];
        buf[2 * i] = p[0] + 0.018 * gauss();
        buf[2 * i + 1] = p[1] + 0.018 * gauss();
      }
    }
  }

  // ---------- canvas sizing ----------
  let W = 0, H = 0, scale = 1;
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    W = root.clientWidth;
    H = Math.round(Math.max(260, Math.min(360, W * 0.34)));
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    scale = Math.min(W, H) / (2 * 1.25);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, W, H);
  }
  function px(x, y) { return [W / 2 + x * scale, H / 2 - y * scale]; }
  window.addEventListener("resize", resize);

  function smoother(s) { return s * s * s * (s * (6 * s - 15) + 10); }
  function lerpCol(a, b, t) {
    return `rgb(${a[0] + (b[0] - a[0]) * t | 0},${a[1] + (b[1] - a[1]) * t | 0},${a[2] + (b[2] - a[2]) * t | 0})`;
  }

  // ---------- static frame for reduced motion / no animation ----------
  function drawStatic(scene) {
    resize();
    fillKeyframe(x1, scene);
    drawBonds(scene, 0.5);
    ctx.fillStyle = lerpCol(GRAY, ACCENT, 0.5);
    for (let i = 0; i < N; i++) { const [cx, cy] = px(x1[2 * i], x1[2 * i + 1]); ctx.fillRect(cx, cy, 1.8, 1.8); }
  }
  function drawBonds(scene, alpha) {
    if (scene.kind === "noise" || alpha <= 0.01) return;
    ctx.strokeStyle = `rgba(${ACCENT[0]},${ACCENT[1]},${ACCENT[2]},${0.5 * alpha})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const [a, b] of scene.bonds) {
      const [ax, ay] = px(scene.points[a][0], scene.points[a][1]);
      const [bx, by] = px(scene.points[b][0], scene.points[b][1]);
      ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
    }
    ctx.stroke();
  }

  if (reduce) { drawStatic(MOL); return; }

  // ---------- animation state machine ----------
  let k = 0;                       // index of the "from" scene in SCENES
  let s = 0;                       // morph progress 0..1
  let holding = 0;                 // ms remaining in hold
  let last = null;
  fillKeyframe(x0, SCENES[0]);
  fillKeyframe(x1, SCENES[1]);

  const MORPH = 2000, HOLD_STRUCT = 1500, HOLD_NOISE = 550;
  let visible = true;
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; }, { threshold: 0 }).observe(root);
  }

  function advance() {
    k = (k + 1) % SCENES.length;
    for (let i = 0; i < N * 2; i++) x0[i] = x1[i];
    fillKeyframe(x1, SCENES[(k + 1) % SCENES.length]);
    s = 0;
  }

  function frame(now) {
    requestAnimationFrame(frame);
    if (last === null) last = now;
    let dt = now - last; last = now;
    if (!visible) return;
    if (dt > 60) dt = 60;

    const fromScene = SCENES[k];
    const toScene = SCENES[(k + 1) % SCENES.length];

    if (holding > 0) {
      holding -= dt;
      if (holding <= 0) advance();
    } else {
      s += dt / MORPH;
      if (s >= 1) { s = 1; holding = toScene.kind === "noise" ? HOLD_NOISE : HOLD_STRUCT; }
    }

    const e = smoother(s);
    // structuredness: how close the displayed config is to a structured scene
    let struct = 0, structScene = null;
    if (toScene.kind !== "noise") { struct = e; structScene = toScene; }
    else if (fromScene.kind !== "noise") { struct = 1 - e; structScene = fromScene; }

    // motion trails: clear with a translucent white wash
    ctx.fillStyle = "rgba(255,255,255,0.30)";
    ctx.fillRect(0, 0, W, H);

    if (structScene) drawBonds(structScene, struct);

    ctx.fillStyle = lerpCol(GRAY, ACCENT, 0.55 * struct);
    const sz = 1.7 + 0.4 * struct;
    for (let i = 0; i < N; i++) {
      const ax = x0[2 * i], ay = x0[2 * i + 1];
      const bx = x1[2 * i], by = x1[2 * i + 1];
      const [cx, cy] = px(ax + (bx - ax) * e, ay + (by - ay) * e);
      ctx.fillRect(cx, cy, sz, sz);
    }
  }

  resize();
  requestAnimationFrame(frame);
})();
