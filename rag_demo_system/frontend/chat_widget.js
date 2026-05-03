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
      placeholder: 'Сообщение...',
      rows: '1',
      'aria-label': 'Сообщение',
      onkeydown: function(e) {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
      },
    });
    var sendBtn = el('button', {
      class: 'mlc-send',
      text: '→',
      'aria-label': 'Отправить',
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
    var header = el('div', { class: 'mlc-header' }, [statusDot, headerTitle, headerMeta, downloadBtn]);

    var intake = buildIntake(function(n, p) {
      name = n; phone = p;
      headerMeta.textContent = name || 'Аноним';
      intake.remove();
      composerInput.focus();
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
          if (data.reply) append('bot', data.reply);
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
    var skipBtn = el('button', {
      class: 'mlc-btn-link',
      text: 'Пропустить',
      onclick: function() { onSubmit('', ''); },
    });
    return el('div', { class: 'mlc-intake' }, [
      el('h2', { class: 'mlc-display', text: 'Прежде чем начать' }),
      el('p', { text: 'Если оставите телефон, мы отправим расчёт в SMS и сможем перезвонить.' }),
      el('div', { class: 'mlc-fields' }, [nameInput, phoneInput]),
      el('div', { class: 'mlc-intake-actions' }, [startBtn, skipBtn]),
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
    var monitor = el('div', { class: 'mlc-monitor mlc-root' }, [
      el('iframe', { src: '/sip_monitor.html?user=chat' }),
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
