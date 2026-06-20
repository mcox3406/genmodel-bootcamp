// Interactive probability-transport playground for the Week 1 sandbox lab.
// Vanilla JS, no dependencies. Particles drawn from N(0, I) are advected by
// the lab's velocity fields with a Heun integrator.
(function () {
  const root = document.getElementById("flow-playground");
  if (!root) return;

  const ACCENT = "179, 19, 42";
  const N = 1600;
  const WORLD = 3.4;          // world half-width shown on canvas
  const DT = 0.012;

  const FIELDS = {
    "contraction": { f: (x, y) => [-x, -y], hint: "u(x) = -x" },
    "translation": { f: () => [1.0, 0.45], hint: "u(x) = c" },
    "rotation": { f: (x, y) => [-y, x], hint: "u(x) = (-x2, x1)" },
    "swirl": {
      f: (x, y) => { const e = 3 * Math.exp(-(x * x + y * y)); return [-y * e, x * e]; },
      hint: "u(x) = a (-x2, x1) exp(-|x|^2)",
    },
    "double well": {
      f: (x, y) => [-4 * x * (x * x - 1), -2 * y],
      hint: "u(x) = -grad V,  V = (x1^2 - 1)^2 + x2^2",
    },
  };

  let fieldName = "contraction";
  let trails = true;
  let running = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- DOM ----------
  const controls = document.createElement("div");
  controls.className = "pg-controls";
  const fieldBtns = {};
  for (const name of Object.keys(FIELDS)) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = name;
    b.addEventListener("click", () => { fieldName = name; syncButtons(); });
    controls.appendChild(b);
    fieldBtns[name] = b;
  }
  const sep = document.createElement("span");
  sep.className = "pg-sep";
  controls.appendChild(sep);

  const playBtn = document.createElement("button");
  playBtn.type = "button";
  playBtn.addEventListener("click", () => { running = !running; syncButtons(); });
  controls.appendChild(playBtn);

  const resetBtn = document.createElement("button");
  resetBtn.type = "button";
  resetBtn.textContent = "resample";
  resetBtn.addEventListener("click", () => { resample(); clearCanvas(); });
  controls.appendChild(resetBtn);

  const trailBtn = document.createElement("button");
  trailBtn.type = "button";
  trailBtn.addEventListener("click", () => { trails = !trails; clearCanvas(); syncButtons(); });
  controls.appendChild(trailBtn);

  const hint = document.createElement("span");
  hint.className = "pg-note";
  controls.appendChild(hint);

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  root.appendChild(canvas);
  root.appendChild(controls);

  function syncButtons() {
    for (const [n, b] of Object.entries(fieldBtns)) b.classList.toggle("on", n === fieldName);
    playBtn.textContent = running ? "pause" : "play";
    trailBtn.textContent = trails ? "trails: on" : "trails: off";
    trailBtn.classList.toggle("on", trails);
    hint.textContent = FIELDS[fieldName].hint;
  }

  // ---------- canvas sizing ----------
  let W = 0, H = 0, scale = 1;
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    W = root.clientWidth - 2;
    H = Math.min(420, Math.max(300, Math.round(W * 0.55)));
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    scale = Math.min(W, H) / (2 * WORLD);
    clearCanvas();
  }
  function toPx(x, y) { return [W / 2 + x * scale, H / 2 - y * scale]; }
  function clearCanvas() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);
  }
  window.addEventListener("resize", resize);

  // ---------- particles ----------
  const px = new Float32Array(N);
  const py = new Float32Array(N);
  function gauss() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function resample() {
    for (let i = 0; i < N; i++) { px[i] = gauss(); py[i] = gauss(); }
  }

  function step() {
    const f = FIELDS[fieldName].f;
    const xmax = (W / 2) / scale + 0.6, ymax = (H / 2) / scale + 0.6;
    for (let i = 0; i < N; i++) {
      let x = px[i], y = py[i];
      const [u1, v1] = f(x, y);
      const [u2, v2] = f(x + DT * u1, y + DT * v1);
      x += DT * 0.5 * (u1 + u2);
      y += DT * 0.5 * (v1 + v2);
      // respawn particles that drift far off-canvas (translation never ends)
      if (x > xmax || x < -xmax || y > ymax || y < -ymax) { x = gauss(); y = gauss(); }
      px[i] = x; py[i] = y;
    }
  }

  function draw() {
    if (trails) {
      ctx.fillStyle = "rgba(255,255,255,0.085)";
      ctx.fillRect(0, 0, W, H);
    } else {
      clearCanvas();
    }
    ctx.fillStyle = "rgba(" + ACCENT + ",0.55)";
    for (let i = 0; i < N; i++) {
      const [cx, cy] = toPx(px[i], py[i]);
      ctx.fillRect(cx, cy, 1.7, 1.7);
    }
  }

  function loop() {
    if (running) { step(); draw(); }
    requestAnimationFrame(loop);
  }

  resample();
  resize();
  syncButtons();
  draw();
  requestAnimationFrame(loop);
})();
