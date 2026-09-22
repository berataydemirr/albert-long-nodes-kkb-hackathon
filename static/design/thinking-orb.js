// <thinking-orb state="searching" size="20|64" paused> — a framework-free
// port of thinking-orbs' <ThinkingOrb> (v0.3.1, MIT, vendor/). The page
// has no bundler and runs React from a global, so we drive the package's
// own canvas engine from a custom element instead of its React wrapper.
import { resolvePreset, MODE_DRAWS } from './vendor/thinking-orbs-engine.js';

const LABELS = {
  working: 'Working…', searching: 'Searching…', solving: 'Solving…',
  listening: 'Listening…', connecting: 'Connecting…', weaving: 'Weaving…',
  composing: 'Composing…', breathing: 'Thinking…', shaping: 'Shaping…'
};

// Same resolution order as the package's theme="auto": nearest
// data-theme / .dark / .light ancestor, then the OS preference.
function isDark(el) {
  for (let e = el; e; e = e.parentElement) {
    const t = e.getAttribute && e.getAttribute('data-theme');
    if (t === 'dark') return true;
    if (t === 'light') return false;
    if (e.classList && e.classList.contains('dark')) return true;
    if (e.classList && e.classList.contains('light')) return false;
  }
  return matchMedia('(prefers-color-scheme: dark)').matches;
}

const darkMq = matchMedia('(prefers-color-scheme: dark)');
const motionMq = matchMedia('(prefers-reduced-motion: reduce)');

class ThinkingOrbElement extends HTMLElement {
  static observedAttributes = ['state', 'size', 'speed', 'paused'];

  connectedCallback() {
    if (!this.canvas) {
      this.canvas = document.createElement('canvas');
      this.canvas.setAttribute('role', 'img');
      this.canvas.style.display = 'block';
      this.appendChild(this.canvas);
    }
    this.style.display = this.style.display || 'inline-block';
    this.onEnv = () => this.restart();
    darkMq.addEventListener('change', this.onEnv);
    motionMq.addEventListener('change', this.onEnv);
    this.mo = new MutationObserver(this.onEnv);
    this.mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme'] });
    this.io = new IntersectionObserver(([e]) => { this.visible = e.isIntersecting; this.sync(); });
    this.io.observe(this.canvas);
    this.onVis = () => this.sync();
    document.addEventListener('visibilitychange', this.onVis);
    this.visible = true;
    this.restart();
  }

  disconnectedCallback() {
    this.stop();
    darkMq.removeEventListener('change', this.onEnv);
    motionMq.removeEventListener('change', this.onEnv);
    document.removeEventListener('visibilitychange', this.onVis);
    this.mo && this.mo.disconnect();
    this.io && this.io.disconnect();
  }

  attributeChangedCallback() { if (this.isConnected && this.canvas) this.restart(); }

  restart() {
    this.stop();
    const state = this.getAttribute('state') || 'working';
    const size = Number(this.getAttribute('size')) === 20 ? 20 : 64;
    const dpr = Math.min(2, devicePixelRatio || 1);
    const c = this.canvas;
    c.width = Math.round(size * dpr);
    c.height = Math.round(size * dpr);
    c.style.width = c.style.height = size + 'px';
    if (!this.hasAttribute('aria-label')) c.setAttribute('aria-label', LABELS[state] || LABELS.working);
    const ctx = c.getContext('2d');
    const { mode, speed, opts } = resolvePreset(state, size);
    const draw = MODE_DRAWS[mode];
    const dark = isDark(this);
    const mult = (speed || 1) * (Number(this.getAttribute('speed')) || 1);
    this.paint = (t) => {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size, size);
      draw(ctx, size, t, dark, opts);
    };
    this.tick = () => {
      this.paint(performance.now() / 1000 * mult);
      if (this.running) this.raf = requestAnimationFrame(this.tick);
    };
    this.still = motionMq.matches;
    if (this.still) { this.paint(0.6); return; }
    this.paint(performance.now() / 1000 * mult);
    this.sync();
  }

  sync() {
    const want = !this.still && !this.hasAttribute('paused') && this.visible && document.visibilityState !== 'hidden';
    if (want && !this.running && this.tick) { this.running = true; this.raf = requestAnimationFrame(this.tick); }
    else if (!want) this.stop();
  }

  stop() { this.running = false; cancelAnimationFrame(this.raf); }
}

if (!customElements.get('thinking-orb')) customElements.define('thinking-orb', ThinkingOrbElement);
