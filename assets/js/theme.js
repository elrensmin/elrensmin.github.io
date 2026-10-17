(function() {
  const scrollTop = document.getElementById('scroll-top');
  if (scrollTop) {
    window.addEventListener('scroll', function() {
      scrollTop.hidden = window.scrollY < window.innerHeight * 2;
    });
    scrollTop.addEventListener('click', function() {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  const toc = document.getElementById('toc');
  const content = document.querySelector('.log-content');
  if (content) {
    content.querySelectorAll('a').forEach(function(a) {
      if (a.hostname && a.hostname !== location.hostname) {
        a.target = '_blank';
        a.rel = 'noopener';
      }
    });
    content.addEventListener('click', function(e) {
      if (e.target.tagName !== 'IMG') return;
      const box = document.createElement('div');
      box.className = 'lightbox';
      const img = document.createElement('img');
      img.src = e.target.src;
      img.alt = e.target.alt;
      box.appendChild(img);
      box.addEventListener('click', function() { box.remove(); });
      document.body.appendChild(box);
    });
  }
  if (toc && content) {
    const headings = content.querySelectorAll('h2');
    if (headings.length > 1) {
      const root = document.createElement('ul');
      const items = [];
      headings.forEach(function(h) {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = '#' + h.id;
        a.textContent = h.textContent;
        li.appendChild(a);
        root.appendChild(li);
        items.push(li);
      });
      toc.appendChild(root);

      const spy = function() {
        let active = null;
        headings.forEach(function(h) {
          if (h.getBoundingClientRect().top <= 80) active = h;
        });
        items.forEach(function(li) {
          li.classList.toggle('active', li.querySelector('a').getAttribute('href') === '#' + (active ? active.id : ''));
        });
      };
      window.addEventListener('scroll', spy, { passive: true });
      spy();
    }
  }

  /* --- Popup notes ------------------------------------------------------
     A [*] marker opens the contents of the .popup it controls. The popup's
     nodes are MOVED into the panel (not cloned), so the MathJax output
     typeset at load stays valid and is never rendered twice.               */
  (function() {
    const triggers = document.querySelectorAll('.popup-trigger');
    if (!triggers.length) return;

    let overlay = null;
    let openTrigger = null;

    const typeset = function(el) {
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([el]).catch(function() {});
      }
    };

    const close = function(restoreHash) {
      if (!overlay || !openTrigger) return;
      const content = overlay.querySelector('.popup-panel > .popup-body');
      const target = document.getElementById(openTrigger.getAttribute('aria-controls'));
      if (content && target) target.appendChild(content);
      overlay.remove();
      overlay = null;
      document.body.style.overflow = '';
      openTrigger.setAttribute('aria-expanded', 'false');
      if (restoreHash && location.hash && /^#popup-/.test(location.hash)) {
        history.pushState(null, '', location.pathname + location.search);
      }
      openTrigger = null;
    };

    const open = function(trigger) {
      if (overlay) close(false);
      const source = document.getElementById(trigger.getAttribute('aria-controls'));
      if (!source) return;

      overlay = document.createElement('div');
      overlay.className = 'popup-overlay';
      overlay.setAttribute('role', 'dialog');
      overlay.setAttribute('aria-modal', 'true');

      const panel = document.createElement('div');
      panel.className = 'popup-panel';

      const closeBtn = document.createElement('button');
      closeBtn.type = 'button';
      closeBtn.className = 'popup-close';
      closeBtn.setAttribute('aria-label', 'Close');
      closeBtn.textContent = '\u00d7';

      const label = document.createElement('span');
      label.className = 'popup-label';
      label.textContent = 'Note';

      const body = document.createElement('div');
      body.className = 'popup-body';
      while (source.firstChild) body.appendChild(source.firstChild);

      panel.appendChild(closeBtn);
      panel.appendChild(label);
      panel.appendChild(body);
      overlay.appendChild(panel);
      document.body.appendChild(overlay);
      document.body.style.overflow = 'hidden';

      openTrigger = trigger;
      trigger.setAttribute('aria-expanded', 'true');

      overlay.addEventListener('click', function(e) {
        if (e.target === overlay) close(true);   // click outside the panel
      });
      closeBtn.addEventListener('click', function() { close(true); });

      typeset(body);
      closeBtn.focus();
    };

    triggers.forEach(function(trigger) {
      trigger.setAttribute('aria-expanded', 'false');
      trigger.addEventListener('click', function() { open(trigger); });
      trigger.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          open(trigger);
        }
      });
    });

    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') close(true);
    });

    // Deep link: /writing/<slug>/#popup-<id> opens it directly.
    if (location.hash && /^#popup-/.test(location.hash)) {
      const target = document.getElementById(location.hash.slice(1));
      if (target) {
        const trigger = document.querySelector('.popup-trigger[aria-controls="' + target.id + '"]');
        if (trigger) open(trigger);
      }
    }
  })();
})();
