// Interactive reverse-diffusion playground for the Week 2 score-model lab.
// Vanilla JS, no dependencies. A multi-well target (a mixture of Gaussians, laid
// out like a small molecule) has a closed-form perturbed score, so the reverse
// process shown here is exact, not learned. Toggle the probability-flow ODE
// (deterministic) against the reverse SDE (stochastic): same target, different
// paths. This is the live picture the coding lab reproduces with a trained net.
(function () {
  const root = document.getElementById("diffusion-playground");
  if (!root) return;

  const ACCENT = "179, 19, 42";
  const GRAY = "120, 120, 130";
  const N = 1400;
  const WORLD = 3.4;
  const S0 = 0.16;                       // clean per-mode std ("atoms")
  const SIG_MAX = 3.0;
  const STEPS = 150;                     // reverse steps from SIG_MAX to ~0

  // three modes at the vertices of a triangle, a triatomic-molecule layout
  const MODES = [[0.0, 1.25], [-1.15, -0.7], [1.15, -0.7]];

  let sampler = "ode";                   // "ode" or "sde"
  let showScore = false;
  let running = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- noise schedule (EDM-style, rho = 7) ----------
  const RHO = 7, SIG_MIN = 0.02;
  const sched = [];
  for (let i = 0; i < STEPS; i++) {
    const t = i / (STEPS - 1);
    sched.push(Math.pow(Math.pow(SIG_MAX, 1 / RHO) +
      t * (Math.pow(SIG_MIN, 1 / RHO) - Math.pow(SIG_MAX, 1 / RHO)), RHO));
  }
  sched.push(0.0);

  // ---------- mixture score at noise level sigma (exact) ----------
  function scoreAt(x, y, sigma) {
    const varr = S0 * S0 + sigma * sigma;
    let w = [0, 0, 0], wsum = 0, mx = -Infinity;
    const q = [];
    for (let k = 0; k < 3; k++) {
      const dx = x - MODES[k][0], dy = y - MODES[k][1];
      const l = -(dx * dx + dy * dy) / (2 * varr);
      q.push(l); if (l > mx) mx = l;
    }
    for (let k = 0; k < 3; k++) { w[k] = Math.exp(q[k] - mx); wsum += w[k]; }
    let sx = 0, sy = 0;
    for (let k = 0; k < 3; k++) {
      const a = w[k] / wsum;
      sx += a * (MODES[k][0] - x) / varr;
      sy += a * (MODES[k][1] - y) / varr;
    }
    return [sx, sy];
  }

  // ---------- DOM ----------
  const controls = document.createElement("div");
  controls.className = "pg-controls";
  const samplerBtns = {};
  for (const [key, label] of [["ode", "probability-flow ODE"], ["sde", "reverse SDE"]]) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = label;
    b.addEventListener("click", () => { sampler = key; restart(); syncButtons(); });
    controls.appendChild(b); samplerBtns[key] = b;
  }
  const sep = document.createElement("span"); sep.className = "pg-sep"; controls.appendChild(sep);

  const playBtn = document.createElement("button");
  playBtn.type = "button";
  playBtn.addEventListener("click", () => { running = !running; syncButtons(); });
  controls.appendChild(playBtn);

  const restartBtn = document.createElement("button");
  restartBtn.type = "button"; restartBtn.textContent = "restart";
  restartBtn.addEventListener("click", () => { restart(); });
  controls.appendChild(restartBtn);

  const scoreBtn = document.createElement("button");
  scoreBtn.type = "button";
  scoreBtn.addEventListener("click", () => { showScore = !showScore; syncButtons(); });
  controls.appendChild(scoreBtn);

  const hint = document.createElement("span");
  hint.className = "pg-note";
  controls.appendChild(hint);

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  root.appendChild(canvas);
  root.appendChild(controls);

  function syncButtons() {
    for (const [k, b] of Object.entries(samplerBtns)) b.classList.toggle("on", k === sampler);
    playBtn.textContent = running ? "pause" : "play";
    scoreBtn.textContent = showScore ? "score field: on" : "score field: off";
    scoreBtn.classList.toggle("on", showScore);
    hint.textContent = "step " + Math.min(idx, STEPS) + " / " + STEPS +
      "   sigma = " + sched[Math.min(idx, STEPS)].toFixed(2);
  }

  // ---------- canvas sizing ----------
  let W = 0, H = 0, scale = 1;
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    W = root.clientWidth - 2;
    H = Math.min(440, Math.max(320, Math.round(W * 0.6)));
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    scale = Math.min(W, H) / (2 * WORLD);
  }
  function toPx(x, y) { return [W / 2 + x * scale, H / 2 - y * scale]; }
  window.addEventListener("resize", resize);

  // ---------- particles ----------
  const px = new Float32Array(N);
  const py = new Float32Array(N);
  let idx = 0, holdFrames = 0;

  function gauss() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function restart() {
    for (let i = 0; i < N; i++) { px[i] = SIG_MAX * gauss(); py[i] = SIG_MAX * gauss(); }
    idx = 0; holdFrames = 0;
  }

  function step() {
    if (idx >= STEPS) { if (++holdFrames > 70) restart(); return; }
    const s = sched[idx], sNext = sched[idx + 1], dsig = sNext - s;   // negative
    for (let i = 0; i < N; i++) {
      const [gx, gy] = scoreAt(px[i], py[i], s);
      if (sampler === "ode") {
        px[i] += -s * gx * dsig;
        py[i] += -s * gy * dsig;
      } else {
        px[i] += -2 * s * gx * dsig;
        py[i] += -2 * s * gy * dsig;
        if (sNext > 0) {
          const a = 0.6 * Math.sqrt(2 * s * (-dsig));   // tempered Brownian term
          px[i] += a * gauss(); py[i] += a * gauss();
        }
      }
    }
    idx++;
  }

  function draw() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);

    if (showScore) {
      const s = sched[Math.min(idx, STEPS - 1)] || 0.05;
      ctx.strokeStyle = "rgba(" + GRAY + ",0.5)";
      ctx.lineWidth = 1;
      const G = 15;
      for (let a = 0; a < G; a++) for (let b = 0; b < G; b++) {
        const wx = -2.7 + (5.4 * a) / (G - 1), wy = -2.7 + (5.4 * b) / (G - 1);
        const [sx, sy] = scoreAt(wx, wy, Math.max(s, 0.25));
        const n = Math.hypot(sx, sy) + 1e-9, L = 0.22;
        const [x0, y0] = toPx(wx, wy);
        const [x1, y1] = toPx(wx + L * sx / n, wy + L * sy / n);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      }
    }

    // target modes
    ctx.fillStyle = "rgba(" + GRAY + ",0.9)";
    for (const m of MODES) {
      const [cx, cy] = toPx(m[0], m[1]);
      ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 2 * Math.PI); ctx.fill();
    }

    // particles
    ctx.fillStyle = "rgba(" + ACCENT + ",0.5)";
    for (let i = 0; i < N; i++) {
      const [cx, cy] = toPx(px[i], py[i]);
      ctx.fillRect(cx, cy, 1.8, 1.8);
    }
  }

  function loop() {
    if (running) { step(); }
    draw(); syncButtons();
    requestAnimationFrame(loop);
  }

  restart();
  resize();
  syncButtons();
  draw();
  requestAnimationFrame(loop);
})();
