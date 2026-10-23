// shared tab behavior: activates the clicked tab, remembers the last
// selection per page in localStorage, and restores it on load so that
// coming back to /i/ or /o/ from a post shows the previous tab.
(function () {
  var KEY = 'tabs-state:' + location.pathname;
  var tabs = document.querySelectorAll('.tabs .tab');
  if (!tabs.length) return;

  function select(tab) {
    tabs.forEach(function (t) {
      var on = t === tab;
      t.classList.toggle('active', on);
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      var panel = document.getElementById('tab-' + t.getAttribute('data-tab'));
      if (panel) panel.hidden = !on;
    });
  }

  tabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      select(tab);
      try {
        localStorage.setItem(KEY, tab.getAttribute('data-tab'));
      } catch (e) { /* private mode etc: state just won't persist */ }
    });
  });

  var saved = null;
  try {
    saved = localStorage.getItem(KEY);
  } catch (e) { /* ignore */ }
  if (!saved) return;

  var match = Array.prototype.filter.call(tabs, function (t) {
    return t.getAttribute('data-tab') === saved;
  })[0];
  if (match) select(match);
})();