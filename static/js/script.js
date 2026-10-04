// Rice Yield Prediction — Main JS
'use strict';

// Auto-dismiss alerts after 6 seconds
document.querySelectorAll('.alert-dismissible').forEach(el => {
  setTimeout(() => {
    const btn = el.querySelector('.btn-close');
    if (btn) btn.click();
  }, 6000);
});

// Add loading spinner to any form with data-loading
document.querySelectorAll('form[data-loading]').forEach(form => {
  form.addEventListener('submit', function() {
    const btn = this.querySelector('[type=submit]');
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing…';
    }
  });
});

// Tooltip initialisation
document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
  new bootstrap.Tooltip(el);
});
