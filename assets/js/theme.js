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
})();
