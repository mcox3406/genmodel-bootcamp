/* Builds the book-style sidebar table of contents, highlights the current
   page, handles the mobile drawer, and runs the on-this-page scroll-spy.
   The course structure lives in the TOC constant below — edit here once
   and every page picks it up. */
(function () {
  'use strict';

  var PARTS = [
    { label: 'Part I · Learning the dynamics', weeks: [1, 2, 3, 4] },
    { label: 'Part II · The design space', weeks: [5, 6, 7] },
    { label: 'Part III · Conditioning & control', weeks: [8, 9] },
    { label: 'Part IV · Applications & evaluation', weeks: [10, 11, 12] }
  ];

  var CHAPTERS = {
    1: {
      title: 'Probability transport & state spaces',
      notes: 'Probability transport and flow maps',
      problems: [
        'Data distributions, representations, and symmetries',
        'Analytic flow maps and the continuity equation',
        'Change of variables and log-likelihoods',
        'Lab: build a probability-flow sandbox',
        'Lab: learn a field from Gaussian to two moons',
        'Lab: symmetry and periodicity sanity checks'
      ]
    },
    2: {
      title: 'Diffusion & score models',
      notes: 'Diffusion and score-based models',
      problems: [
        'From Boltzmann distributions to score matching',
        'The EDM design space',
        'Lab: score-based sampling of a free-energy surface',
        'Lab: equivariant diffusion for 3D molecules'
      ]
    },
    3: {
      title: 'Flow matching & rectified flows',
      notes: 'Flow matching and rectified flows',
      problems: [
        'Conditional flow matching and the marginal field',
        'Straightness, curvature, and rectified flow',
        'Lab: independent vs optimal-transport couplings',
        'Lab: Riemannian flow matching on the torus'
      ]
    },
    4: { title: 'Stochastic interpolants' },
    5: { title: 'Couplings & optimal transport' },
    6: { title: 'One-step & few-step generation' },
    7: { title: 'Discrete & hybrid generation' },
    8: { title: 'Guidance, alignment & inverse design' },
    9: { title: 'Schrodinger bridges & control' },
    10: { title: 'Molecular & protein generation' },
    11: { title: 'Crystalline materials generation' },
    12: { title: 'Failure modes & evaluation' }
  };

  var host = document.querySelector('aside.booknav');
  if (!host) return;
  var root = host.getAttribute('data-root') || '.';

  /* Where are we? -> { week, page } with page one of 'index'|'notes'|'problem-K'|null */
  var path = location.pathname;
  var current = { week: null, page: null };
  var m = path.match(/weeks\/week(\d+)\.html$/);
  if (m) current = { week: +m[1], page: 'index' };
  m = path.match(/weeks\/week(\d+)\/notes\.html$/);
  if (m) current = { week: +m[1], page: 'notes' };
  m = path.match(/weeks\/week(\d+)\/problem-(\d+)\.html$/);
  if (m) current = { week: +m[1], page: 'problem-' + m[2] };
  var onOverview = current.week === null && !/docs\//.test(path);

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text) n.textContent = text;
    return n;
  }
  function link(href, cls, text) {
    var a = el('a', cls, text);
    a.href = href;
    return a;
  }

  var inner = el('div', 'booknav-inner');
  var brand = link(root + '/index.html', 'bn-brand', 'Generative Modeling for Molecules & Materials');
  inner.appendChild(brand);
  inner.appendChild(el('div', 'bn-tagline', 'Reading, derivation & coding group'));

  var top = el('div', 'bn-toplinks');
  var overview = link(root + '/index.html', onOverview ? 'active' : '', 'Overview');
  top.appendChild(overview);
  top.appendChild(link(root + '/docs/syllabus.pdf', '', 'Syllabus (PDF)'));
  inner.appendChild(top);

  var nav = el('nav');
  nav.setAttribute('aria-label', 'Chapters');
  PARTS.forEach(function (part) {
    nav.appendChild(el('div', 'bn-part', part.label));
    part.weeks.forEach(function (w) {
      var ch = CHAPTERS[w];
      var a = link(root + '/weeks/week' + w + '.html', 'bn-chapter' + (current.week === w ? ' active' : ''), '');
      a.appendChild(el('span', 'bn-num', String(w)));
      a.appendChild(el('span', '', ch.title));
      nav.appendChild(a);
      /* Expand notes + problems only for the chapter being read */
      if (current.week === w && (ch.notes || ch.problems)) {
        var sub = el('div', 'bn-sub');
        var base = root + '/weeks/week' + w + '/';
        if (ch.notes) {
          sub.appendChild(link(base + 'notes.html', current.page === 'notes' ? 'active' : '', 'Lecture notes'));
        }
        if ((ch.problems || []).length) {
          sub.appendChild(el('div', 'bn-sub-label', 'Problems'));
        }
        (ch.problems || []).forEach(function (title, i) {
          var k = i + 1;
          sub.appendChild(link(base + 'problem-' + k + '.html',
            current.page === 'problem-' + k ? 'active' : '',
            k + ' · ' + title));
        });
        nav.appendChild(sub);
      }
    });
  });
  inner.appendChild(nav);
  host.appendChild(inner);

  /* Mobile drawer */
  var toggle = el('button', 'booknav-toggle');
  toggle.type = 'button';
  toggle.setAttribute('aria-label', 'Toggle table of contents');
  toggle.innerHTML = '&#9776;&nbsp; Contents';
  var scrim = el('div', 'booknav-scrim');
  document.body.appendChild(toggle);
  document.body.appendChild(scrim);
  toggle.addEventListener('click', function () { document.body.classList.toggle('nav-open'); });
  scrim.addEventListener('click', function () { document.body.classList.remove('nav-open'); });
  host.addEventListener('click', function (e) {
    if (e.target.closest('a')) document.body.classList.remove('nav-open');
  });

  /* Scroll-spy for the right-hand "On this page" rail */
  var side = document.querySelector('aside.side');
  if (side) {
    var pairs = [];
    side.querySelectorAll('a[href^="#"]').forEach(function (a) {
      var target = document.getElementById(a.getAttribute('href').slice(1));
      if (target) pairs.push({ a: a, target: target });
    });
    if (pairs.length) {
      var setActive = function () {
        var line = window.scrollY + 110;
        var active = pairs[0];
        pairs.forEach(function (p) {
          var top = p.target.getBoundingClientRect().top + window.scrollY;
          if (top <= line) active = p;
        });
        pairs.forEach(function (p) { p.a.classList.toggle('active', p === active); });
      };
      var ticking = false;
      window.addEventListener('scroll', function () {
        if (!ticking) {
          ticking = true;
          requestAnimationFrame(function () { setActive(); ticking = false; });
        }
      }, { passive: true });
      setActive();
    }
  }
})();
