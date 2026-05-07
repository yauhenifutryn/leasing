"""Frontend regression tests for chat widget session wiring."""
from __future__ import annotations

import subprocess
from pathlib import Path


def test_monitor_iframe_session_matches_text_turn_session_after_intake():
    """Run the real widget JS and verify iframe + POST use one session_id."""
    repo_root = Path(__file__).resolve().parents[2]
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.attributes = {};
    this.listeners = {};
    this.style = {};
    this.className = '';
    this.classList = {
      add: (...names) => {
        const values = new Set((this.className || '').split(/\s+/).filter(Boolean));
        for (const name of names) values.add(name);
        this.className = Array.from(values).join(' ');
      },
      remove: (...names) => {
        const remove = new Set(names);
        this.className = (this.className || '').split(/\s+/).filter((name) => !remove.has(name)).join(' ');
      },
    };
    this.textContent = '';
    this.value = '';
    this.disabled = false;
    this.isConnected = false;
  }
  setAttribute(k, v) {
    this.attributes[k] = String(v);
    if (k === 'class') this.className = String(v);
    if (k === 'disabled') this.disabled = true;
  }
  appendChild(child) {
    this.children.push(child);
    child.parentNode = this;
    setConnected(child, this.isConnected);
    return child;
  }
  removeChild(child) {
    this.children = this.children.filter((c) => c !== child);
    setConnected(child, false);
    return child;
  }
  remove() {
    if (this.parentNode) this.parentNode.removeChild(this);
  }
  addEventListener(type, fn) {
    this.listeners[type] = this.listeners[type] || [];
    this.listeners[type].push(fn);
  }
  dispatch(type, event = {}) {
    for (const fn of this.listeners[type] || []) fn.call(this, event);
  }
  focus() {}
  querySelector(selector) {
    return find(this, selector);
  }
}

function setConnected(node, value) {
  node.isConnected = value;
  for (const child of node.children || []) setConnected(child, value);
}

function matches(node, selector) {
  if (selector[0] === '#') return node.attributes.id === selector.slice(1);
  if (selector[0] === '.') return (node.className || '').split(/\s+/).includes(selector.slice(1));
  if (/^[a-z]+$/i.test(selector)) return node.tagName.toLowerCase() === selector.toLowerCase();
  return false;
}

function find(node, selector) {
  if (matches(node, selector)) return node;
  for (const child of node.children || []) {
    const hit = find(child, selector);
    if (hit) return hit;
  }
  return null;
}

function findByText(node, tag, text) {
  if (node.tagName.toLowerCase() === tag && node.textContent === text) return node;
  for (const child of node.children || []) {
    const hit = findByText(child, tag, text);
    if (hit) return hit;
  }
  return null;
}

const storage = new Map();
const root = new Element('div');
root.setAttribute('id', 'mlc-host');
setConnected(root, true);
const document = {
  currentScript: { src: 'http://127.0.0.1:8000/chat_widget.js', dataset: { host: '#mlc-host', api: '' } },
  head: new Element('head'),
  body: new Element('body'),
  createElement: (tag) => new Element(tag),
  querySelector: (selector) => selector === '#mlc-host' ? root : null,
  addEventListener: (type, fn) => {
    if (type === 'DOMContentLoaded') fn();
  },
};
setConnected(document.head, true);
setConnected(document.body, true);

let postedSession = null;
let uuidCounter = 0;
const context = {
  console,
  document,
  window: {},
  localStorage: { getItem: () => null, setItem: () => {} },
  sessionStorage: {
    getItem: (k) => storage.has(k) ? storage.get(k) : null,
    setItem: (k, v) => storage.set(k, String(v)),
    removeItem: (k) => storage.delete(k),
  },
  crypto: { randomUUID: () => `${++uuidCounter}1111111-1111-4111-8111-111111111111` },
  Blob: function Blob() {},
  URL: { createObjectURL: () => 'blob:test', revokeObjectURL: () => {} },
  Date,
  setTimeout: (fn) => { fn(); return 1; },
  confirm: () => true,
  fetch: async (_url, opts) => {
    postedSession = JSON.parse(opts.body).session_id;
    return { json: async () => ({ ok: true, reply: 'ok', action: 'Noop' }) };
  },
  location: { reload: () => {} },
};
context.window = context;

vm.createContext(context);
vm.runInContext(fs.readFileSync('rag_demo_system/frontend/chat_widget.js', 'utf8'), context);

const iframe = find(root, 'iframe');
const iframeSession = new URL('http://127.0.0.1:8000' + iframe.attributes.src).searchParams.get('session');
findByText(root, 'button', 'Начать').dispatch('click');
const textarea = find(root, 'textarea');
textarea.value = 'Здравствуйте';
find(root, '.mlc-send').dispatch('click');

setImmediate(() => {
  if (iframeSession !== postedSession) {
    console.error(`iframe session ${iframeSession} did not match posted session ${postedSession}`);
    process.exit(1);
  }
});
"""
    result = subprocess.run(
        ["node", "-e", script],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
