(function() {
  const toggle = document.getElementById('theme-toggle');
  if (!toggle) return;

  const stored = localStorage.getItem('theme');
  if (stored === 'light') {
    document.body.classList.add('light');
  }

  toggle.addEventListener('click', function() {
    document.body.classList.toggle('light');
    const isLight = document.body.classList.contains('light');
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
  });

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
})();
