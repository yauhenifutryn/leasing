(function() {
  'use strict';
  if (window.__mlcMounted) return;
  window.__mlcMounted = true;

  var SCRIPT = document.currentScript;
  var SRC = SCRIPT && SCRIPT.src ? SCRIPT.src : '';
  var BASE = SRC.replace(/\/chat_widget\.js.*$/, '');
  var API = (SCRIPT && SCRIPT.dataset.api) || (BASE || '');

  function el(tag, props, children) {
    var n = document.createElement(tag);
    if (props) for (var k in props) {
      if (k === 'class') n.className = props[k];
      else if (k === 'text') n.textContent = props[k];
      else if (k.indexOf('on') === 0 && typeof props[k] === 'function')
        n.addEventListener(k.slice(2), props[k]);
      else n.setAttribute(k, props[k]);
    }
    if (children) for (var i = 0; i < children.length; i++) {
      if (children[i]) n.appendChild(children[i]);
    }
    return n;
  }

  function loadCss() {
    if (document.querySelector('link[data-mlc-css]')) return;
    var link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = (BASE || '') + '/chat_widget.css';
    link.setAttribute('data-mlc-css', '1');
    document.head.appendChild(link);
  }

  // ── Sounds (Web Audio API tones; no asset downloads) ──
  // Mute persists in localStorage so the choice survives reload.
  var MUTE_KEY = 'mlc_muted';
  function isMuted() { return localStorage.getItem(MUTE_KEY) === '1'; }
  function setMuted(v) { localStorage.setItem(MUTE_KEY, v ? '1' : '0'); }

  var _audioCtx = null;
  function audioCtx() {
    if (_audioCtx) return _audioCtx;
    var Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    try { _audioCtx = new Ctx(); } catch (e) { return null; }
    return _audioCtx;
  }

  function playTick(freqStart, freqEnd, durMs) {
    if (isMuted()) return;
    var ctx = audioCtx();
    if (!ctx) return;
    // Some browsers suspend AudioContext until first user gesture; try to resume.
    if (ctx.state === 'suspended') { try { ctx.resume(); } catch (e) {} }
    var t0 = ctx.currentTime;
    var t1 = t0 + (durMs / 1000);
    var osc = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freqStart, t0);
    if (freqEnd !== freqStart) osc.frequency.exponentialRampToValueAtTime(freqEnd, t1);
    // Gentle envelope: quick attack, exponential decay (avoids click).
    gain.gain.setValueAtTime(0.0001, t0);
    gain.gain.exponentialRampToValueAtTime(0.18, t0 + 0.005);
    gain.gain.exponentialRampToValueAtTime(0.0001, t1);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t0);
    osc.stop(t1 + 0.02);
  }

  function playSendTick() { playTick(880, 880, 60); }       // bright single beep
  function playReceiveTick() { playTick(1320, 880, 110); }  // descending two-tone

  function genSessionId() {
    var key = 'mlc_session_id';
    var sid = sessionStorage.getItem(key);
    if (!sid) {
      sid = 'chat-' + (crypto.randomUUID
        ? crypto.randomUUID().slice(0, 12)
        : Math.random().toString(36).slice(2, 14));
      sessionStorage.setItem(key, sid);
    }
    return sid;
  }

  function buildPanel(opts) {
    opts = opts || {};
    var sessionId = opts.sessionId || genSessionId();
    var name = '';
    var phone = '';
    var messages = [];   // {role: 'user'|'bot'|'chip', text, ts (ms epoch)}

    var transcript = el('div', { class: 'mlc-transcript' });
    var composerInput = el('textarea', {
      placeholder: 'Сначала примите согласие выше',
      rows: '1',
      'aria-label': 'Сообщение',
      disabled: 'true',
      onkeydown: function(e) {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
      },
    });
    var sendBtn = el('button', {
      class: 'mlc-send',
      text: '→',
      'aria-label': 'Отправить',
      disabled: 'true',
      onclick: send,
    });
    var composer = el('div', { class: 'mlc-composer' }, [composerInput, sendBtn]);

    var statusDot = el('span', {
      class: 'mlc-status-dot',
      role: 'status',
      'aria-label': 'Подключено',
    });
    var headerTitle = el('div', { class: 'mlc-header-title', text: 'Микро Лизинг' });
    var headerMeta = el('div', { class: 'mlc-header-meta', text: 'Аноним' });
    var downloadBtn = el('button', {
      class: 'mlc-download',
      text: '↓',
      'aria-label': 'Скачать переписку',
      title: 'Скачать переписку (.md)',
      onclick: function() { downloadMarkdown(); },
      disabled: 'true',
    });
    var muteBtn = el('button', {
      class: 'mlc-mute',
      text: isMuted() ? '🔇' : '🔊',
      'aria-label': isMuted() ? 'Включить звук' : 'Выключить звук',
      title: isMuted() ? 'Включить звук' : 'Выключить звук',
      onclick: function() {
        var nowMuted = !isMuted();
        setMuted(nowMuted);
        muteBtn.textContent = nowMuted ? '🔇' : '🔊';
        var label = nowMuted ? 'Включить звук' : 'Выключить звук';
        muteBtn.setAttribute('aria-label', label);
        muteBtn.setAttribute('title', label);
        // First click also unlocks AudioContext on browsers that gate it.
        if (!nowMuted) { var c = audioCtx(); if (c && c.state === 'suspended') { try { c.resume(); } catch (e) {} } }
      },
    });
    var header = el('div', { class: 'mlc-header' }, [statusDot, headerTitle, headerMeta, muteBtn, downloadBtn]);

    var intake = buildIntake(function(n, p) {
      name = n; phone = p;
      headerMeta.textContent = name || 'Аноним';
      intake.remove();
      // #2 fix: composer was disabled before consent submit; enable now.
      composerInput.disabled = false;
      composerInput.placeholder = 'Сообщение...';
      sendBtn.disabled = false;
      composerInput.focus();
      // #1 fix: seed the conversational record with the user's name so the
      // classifier + LLM see it on the very first turn. Voice does this
      // implicitly through the name-capture turn ("Меня зовут <X>" → bot
      // ack); chat replicates that explicit pair here. The pair is also
      // appended to the on-screen transcript so the user sees the same
      // greeting voice callers hear ("Здравствуйте, <name>!").
      var hello = name
        ? ('Здравствуйте, ' + name + '! Я Ксения, помощница Микро Лизинг. Чем могу помочь?')
        : 'Здравствуйте! Я Ксения, помощница Микро Лизинг. Чем могу помочь?';
      setTimeout(function() {
        if (name) append('user', 'Меня зовут ' + name + '.');
        append('bot', hello);
      }, 200);
    });
    transcript.appendChild(intake);

    var panel = el('div', { class: 'mlc-panel mlc-root' }, [header, transcript, composer]);

    function append(role, text) {
      if (!transcript.isConnected) return null;  // panel closed mid-fetch
      messages.push({ role: role, text: text, ts: Date.now() });
      if (downloadBtn) downloadBtn.disabled = false;
      var b = el('div', { class: 'mlc-bubble mlc-' + role, text: text });
      transcript.appendChild(b);
      transcript.scrollTop = transcript.scrollHeight;
      return b;
    }

    function appendChip(text) {
      if (!transcript.isConnected) return;
      messages.push({ role: 'chip', text: text, ts: Date.now() });
      if (downloadBtn) downloadBtn.disabled = false;
      var c = el('div', { class: 'mlc-action-chip', text: text });
      transcript.appendChild(c);
      transcript.scrollTop = transcript.scrollHeight;
    }

    function downloadMarkdown() {
      if (!messages.length) return;
      var lines = [];
      lines.push('# Чат с Микро Лизинг');
      lines.push('');
      var hdrParts = [];
      hdrParts.push('**Сессия:** `' + sessionId + '`');
      if (name) hdrParts.push('**Имя:** ' + name);
      if (phone) hdrParts.push('**Телефон:** ' + phone);
      hdrParts.push('**Экспортировано:** ' + new Date().toISOString().replace('T', ' ').slice(0, 19));
      lines.push(hdrParts.join('  \n'));
      lines.push('');
      lines.push('---');
      lines.push('');
      for (var i = 0; i < messages.length; i++) {
        var m = messages[i];
        var t = new Date(m.ts).toLocaleTimeString('ru-RU', { hour12: false });
        if (m.role === 'chip') {
          lines.push('> _[' + t + '] ' + m.text + '_');
        } else if (m.role === 'user') {
          lines.push('**[' + t + '] Вы:** ' + m.text);
        } else if (m.role === 'bot') {
          lines.push('**[' + t + '] Бот:** ' + m.text);
        }
        lines.push('');
      }
      var md = lines.join('\n');
      var blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = 'chat-' + sessionId.replace(/^chat-/, '') + '-' + new Date().toISOString().slice(0, 10) + '.md';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(function() { URL.revokeObjectURL(url); }, 1000);
    }

    async function send() {
      var msg = composerInput.value.trim();
      if (!msg) return;
      var prior = composerInput.value;          // capture for restore-on-failure
      composerInput.value = '';
      composerInput.style.height = 'auto';
      sendBtn.disabled = true;
      append('user', msg);
      playSendTick();
      try {
        var resp = await fetch((API || '') + '/api/text-turn', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: msg,
            session_id: sessionId,
            name: name,
            phone: phone,
          }),
        });
        var data = await resp.json();
        if (!data.ok) {
          append('bot', '[ошибка: ' + (data.error || 'unknown') + ']');
        } else {
          if (data.reply) { append('bot', data.reply); playReceiveTick(); }
          var a = data.action || '';
          if (a === 'FireCalc') appendChip('Расчёт готов');
          else if (a === 'FireSMS') appendChip('SMS отправлено');
          else if (a === 'EndCall') {
            appendChip('Разговор завершён');
            composerInput.disabled = true;
            sendBtn.disabled = true;
            statusDot.classList.add('mlc-disconnected');
            statusDot.setAttribute('aria-label', 'Соединение завершено');
            return;
          }
        }
      } catch (e) {
        append('bot', '[нет связи с сервером]');
        composerInput.value = prior;           // restore for retry
      } finally {
        sendBtn.disabled = false;
      }
    }

    return { panel: panel, sessionId: sessionId };
  }

  function buildIntake(onSubmit) {
    var nameInput = el('input', { placeholder: 'Имя (необязательно)', type: 'text' });
    var phoneInput = el('input', { placeholder: 'Телефон (необязательно)', type: 'tel' });
    var startBtn = el('button', {
      class: 'mlc-btn-primary',
      text: 'Начать',
      onclick: function() { onSubmit(nameInput.value.trim(), phoneInput.value.trim()); },
    });
    // #2 fix: Skip button removed — Начать with empty fields is the same outcome.
    return el('div', { class: 'mlc-intake' }, [
      el('h2', { class: 'mlc-display', text: 'Прежде чем начать' }),
      el('p', { text: 'Если оставите телефон, мы отправим расчёт в SMS и сможем перезвонить.' }),
      el('div', { class: 'mlc-fields' }, [nameInput, phoneInput]),
      el('div', { class: 'mlc-intake-actions' }, [startBtn]),
      el('p', {
        class: 'mlc-consent',
        text: 'Нажимая «Начать», вы соглашаетесь на обработку персональных данных. Поля имени и телефона необязательны.'
      }),
    ]);
  }

  function mountEmbed() {
    loadCss();
    var fab = el('button', {
      class: 'mlc-fab mlc-root',
      text: '💬',
      'aria-label': 'Открыть чат',
    });
    var current = null;

    function dismiss() {
      if (current) {
        current.remove();
        current = null;
      }
      fab.style.display = 'grid';
    }

    fab.addEventListener('click', function() {
      if (current) { dismiss(); return; }
      var built = buildPanel();
      built.panel.classList.add('mlc-embed-panel');
      // Inject a close button into the header (embed mode only).
      var closeBtn = el('button', {
        class: 'mlc-close',
        text: '×',
        'aria-label': 'Закрыть чат',
        onclick: dismiss,
      });
      var headerEl = built.panel.querySelector('.mlc-header');
      if (headerEl) headerEl.appendChild(closeBtn);
      document.body.appendChild(built.panel);
      current = built.panel;
      fab.style.display = 'none';
    });
    document.body.appendChild(fab);
  }

  function mountHost(rootSelector) {
    loadCss();
    var root = document.querySelector(rootSelector);
    if (!root) return;
    var built = buildPanel();
    // #6 fix: pass the chat session_id to the monitor iframe so it can
    // filter to events for THIS chat session. Operator's main monitor
    // (sip_monitor.html opened directly) keeps showing everything.
    var monitor = el('div', { class: 'mlc-monitor mlc-root' }, [
      el('iframe', { src: '/sip_monitor.html?session=' + encodeURIComponent(built.sessionId) }),
    ]);
    var host = el('div', { class: 'mlc-host mlc-root' }, [built.panel, monitor]);
    root.appendChild(host);
  }

  document.addEventListener('DOMContentLoaded', function() {
    var hostSel = SCRIPT && SCRIPT.dataset.host;
    if (hostSel) mountHost(hostSel);
    else mountEmbed();
  });

  window.MikroLeasingChat = {
    mountHost: mountHost,
    mountEmbed: mountEmbed,
    version: '0.1.0',
  };
})();
