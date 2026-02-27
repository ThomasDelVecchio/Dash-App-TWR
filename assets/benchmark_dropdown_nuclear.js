(function () {
  'use strict';

  const MAX_WIDTH = 1400;
  var _menuPollTimer = null;

  function isDark() {
    return !!document.querySelector('[data-theme="dark"]');
  }

  function shouldForce() {
    return isDark() && window.innerWidth <= MAX_WIDTH;
  }

  function paint(el, bg) {
    if (!el) return;
    el.style.setProperty('background-color', bg, 'important');
    el.style.setProperty('color', '#ffffff', 'important');
    el.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
    el.style.setProperty('border-color', 'rgba(255,255,255,0.16)', 'important');
    el.style.setProperty('-webkit-appearance', 'none', 'important');
    el.style.setProperty('appearance', 'none', 'important');
    el.style.setProperty('box-shadow', 'none', 'important');
  }

  /* ----------------------------------------------------------------
     OVERFLOW FIX: walk every ancestor of the strategy-preset dropdown
     and nuke overflow:hidden / overflow:auto so the menu is visible.
     Also push z-index on the menu itself.
     Runs ALWAYS (ungated) so it can never be missed.
     ---------------------------------------------------------------- */
  function forcePresetOverflow() {
    var preset = document.getElementById('strategy-preset-checklist');
    if (!preset) return;

    // Walk ALL ancestors up to body and force overflow visible
    var el = preset.parentElement;
    while (el && el !== document.body) {
      var ov = window.getComputedStyle(el).overflow;
      if (ov === 'hidden' || ov === 'auto' || ov === 'scroll') {
        el.style.setProperty('overflow', 'visible', 'important');
      }
      // Also check overflow-x / overflow-y
      var ovx = window.getComputedStyle(el).overflowX;
      var ovy = window.getComputedStyle(el).overflowY;
      if (ovx === 'hidden' || ovx === 'auto') {
        el.style.setProperty('overflow-x', 'visible', 'important');
      }
      if (ovy === 'hidden' || ovy === 'auto') {
        el.style.setProperty('overflow-y', 'visible', 'important');
      }
      el = el.parentElement;
    }

    // Push z-index on the menu itself
    var menuOuter = preset.querySelector('.Select-menu-outer');
    if (menuOuter) {
      menuOuter.style.setProperty('z-index', '10001', 'important');
      menuOuter.style.setProperty('position', 'absolute', 'important');
      menuOuter.style.setProperty('overflow', 'visible', 'important');
    }
  }

  /* ----------------------------------------------------------------
     NUCLEAR DARK PAINT: find EVERY .Select-menu-outer in the doc
     (portaled or not) and force-dark the menu, inputs, and options.
     Only gated on dark theme, NOT on width — iPad/mobile must work.
     ---------------------------------------------------------------- */
  function nukeAllOpenMenus() {
    if (!isDark()) return;

    // Find every open menu in the DOM — react-select can portal these
    var menus = document.querySelectorAll(
      '.Select-menu-outer, .Select-menu, [class*="MenuList"], [class*="menu-outer"]'
    );
    if (!menus.length) return;

    menus.forEach(function (menu) {
      // Dark background on the menu container itself
      paint(menu, '#151c24');
      menu.style.setProperty('color-scheme', 'dark', 'important');
      menu.style.setProperty('z-index', '10001', 'important');

      // Dark the inner scroll wrapper
      menu.querySelectorAll('.Select-menu, [class*="MenuList"]').forEach(function (inner) {
        paint(inner, '#151c24');
      });

      // Dark ALL child elements (the truly nuclear approach)
      menu.querySelectorAll('*').forEach(function (child) {
        var tag = child.tagName;
        // For inputs, special treatment
        if (tag === 'INPUT' || tag === 'TEXTAREA') {
          child.style.setProperty('background-color', '#1a2332', 'important');
          child.style.setProperty('color', '#ffffff', 'important');
          child.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
          child.style.setProperty('border-color', 'rgba(255,255,255,0.2)', 'important');
          child.style.setProperty('-webkit-appearance', 'none', 'important');
          child.style.setProperty('caret-color', '#ffffff', 'important');
        } else {
          child.style.setProperty('background-color', '#151c24', 'important');
          child.style.setProperty('color', '#ffffff', 'important');
          child.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
        }
      });

      // Now apply the checked vs unchecked distinction on top
      menu.querySelectorAll(
        '.Select-option, .VirtualizedSelectOption, [role="option"]'
      ).forEach(function (opt) {
        var isSelected =
          opt.classList.contains('is-selected') ||
          opt.classList.contains('VirtualizedSelectSelectedOption') ||
          (opt.getAttribute('aria-selected') === 'true');
        var isFocused =
          opt.classList.contains('is-focused') ||
          opt.classList.contains('VirtualizedSelectFocusedOption');

        if (isSelected) {
          opt.style.setProperty('background-color', 'rgba(0,212,255,0.22)', 'important');
          opt.style.setProperty('color', '#ffffff', 'important');
          opt.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
          opt.style.setProperty('border-left', '3px solid #00d4ff', 'important');
          opt.style.setProperty('padding-left', '9px', 'important');
        } else if (isFocused) {
          opt.style.setProperty('background-color', '#253345', 'important');
          opt.style.setProperty('color', '#ffffff', 'important');
          opt.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
          opt.style.setProperty('border-left', '3px solid transparent', 'important');
          opt.style.setProperty('padding-left', '9px', 'important');
        } else {
          opt.style.setProperty('background-color', '#151c24', 'important');
          opt.style.setProperty('color', 'rgba(244,248,255,0.7)', 'important');
          opt.style.setProperty('-webkit-text-fill-color', 'rgba(244,248,255,0.7)', 'important');
          opt.style.setProperty('border-left', '3px solid transparent', 'important');
          opt.style.setProperty('padding-left', '9px', 'important');
        }
      });
    });
  }

  /* ----------------------------------------------------------------
     Start/stop a rapid 60ms interval while a menu is open to
     keep re-painting (react-select re-renders on scroll/filter).
     ---------------------------------------------------------------- */
  function startMenuPoll() {
    if (_menuPollTimer) return;
    _menuPollTimer = setInterval(function () {
      var open = document.querySelector('.Select-menu-outer, .Select.is-open');
      if (!open) {
        clearInterval(_menuPollTimer);
        _menuPollTimer = null;
        return;
      }
      forcePresetOverflow();
      nukeAllOpenMenus();
    }, 60);
  }

  function styleBenchmarkRoot() {
    if (!shouldForce()) return;

    forcePresetOverflow();

    const roots = [
      document.getElementById('benchmark-dropdown'),
      document.getElementById('strategy-preset-checklist'),
      document.getElementById('growth-asset-class-filter')
    ].filter(Boolean);
    if (!roots.length) return;

    const selectors = [
      '.Select-control',
      '.Select-menu-outer',
      '.Select-option',
      '.Select-value',
      '.Select-value-label',
      '.Select-placeholder',
      '.Select-arrow-zone',
      '.Select-clear-zone',
      '.Select-input input',
      '.Select-multi-value-wrapper',
      '[role="combobox"]',
      '[role="listbox"]',
      '[role="option"]',
      'input[type="search"]',
      'input[role="combobox"]',
      '[class*="control"]',
      '[class*="menu"]',
      '[class*="option"]',
      '[class*="singleValue"]',
      '[class*="multiValue"]',
      '[class*="placeholder"]',
      '[class*="indicator"]',
      '[class*="valueContainer"]'
    ];

    roots.forEach((root) => {
      paint(root, '#151c24');
      root.style.setProperty('color-scheme', 'dark', 'important');

      root.querySelectorAll(selectors.join(',')).forEach((node) => {
        paint(node, '#151c24');
      });
    });
  }

  function styleLikelyOpenMenu() {
    if (!shouldForce()) return;

    const menuCandidates = document.querySelectorAll(
      '.Select-menu-outer, [role="listbox"], [class*="menu"], [class*="Menu"]'
    );

    menuCandidates.forEach((menu) => {
      paint(menu, '#151c24');
      menu.querySelectorAll('[role="option"], .Select-option, .VirtualizedSelectOption, [class*="option"], [class*="Option"]').forEach((opt) => {
        paint(opt, '#151c24');
      });
      menu.querySelectorAll('input, label, span, div').forEach((txt) => {
        txt.style.setProperty('color', '#ffffff', 'important');
        txt.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
      });
    });
  }

  function styleProjectionSliders() {
    if (!shouldForce()) return;

    const sliderRoots = [
      document.getElementById('proj-return-slider'),
      document.getElementById('proj-contrib-slider')
    ].filter(Boolean);

    if (!sliderRoots.length) return;

    sliderRoots.forEach((root) => {
      root.style.setProperty('background-color', 'transparent', 'important');
      root.style.setProperty('color-scheme', 'dark', 'important');

      root.querySelectorAll('.rc-slider-rail').forEach((node) => {
        node.style.setProperty('background-color', '#344055', 'important');
        node.style.setProperty('opacity', '1', 'important');
      });

      root.querySelectorAll('.rc-slider-track').forEach((node) => {
        node.style.setProperty('background-color', '#8b5cf6', 'important');
        node.style.setProperty('opacity', '1', 'important');
      });

      root.querySelectorAll('.rc-slider-handle').forEach((node) => {
        node.style.setProperty('background-color', '#8b5cf6', 'important');
        node.style.setProperty('border-color', '#d8b4fe', 'important');
        node.style.setProperty('border-width', '2px', 'important');
        node.style.setProperty('border-style', 'solid', 'important');
        node.style.setProperty('border-radius', '999px', 'important');
        node.style.setProperty('width', '16px', 'important');
        node.style.setProperty('height', '16px', 'important');
        node.style.setProperty('opacity', '1', 'important');
        node.style.setProperty('box-shadow', '0 0 0 2px rgba(139,92,246,0.35)', 'important');
        node.style.setProperty('-webkit-appearance', 'none', 'important');
        node.style.setProperty('appearance', 'none', 'important');
      });

      root.querySelectorAll('.rc-slider-dot').forEach((node) => {
        node.style.setProperty('background-color', '#151c24', 'important');
        node.style.setProperty('border-color', '#64748b', 'important');
        node.style.setProperty('border-radius', '999px', 'important');
      });

      root.querySelectorAll('.rc-slider-mark-text').forEach((node) => {
        node.style.setProperty('color', '#cbd5e1', 'important');
        node.style.setProperty('-webkit-text-fill-color', '#cbd5e1', 'important');
      });

      root.querySelectorAll('.rc-slider-tooltip, .rc-slider-tooltip-content').forEach((node) => {
        node.style.setProperty('background-color', 'transparent', 'important');
      });

      root.querySelectorAll('.rc-slider-tooltip-inner').forEach((node) => {
        node.style.setProperty('background-color', '#1e293b', 'important');
        node.style.setProperty('color', '#ffffff', 'important');
        node.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
        node.style.setProperty('border-color', '#64748b', 'important');
        node.style.setProperty('border-style', 'solid', 'important');
        node.style.setProperty('border-width', '1px', 'important');
        node.style.setProperty('border-radius', '6px', 'important');
        node.style.setProperty('box-shadow', '0 2px 8px rgba(0,0,0,0.35)', 'important');
        node.style.setProperty('opacity', '1', 'important');
      });

      root.querySelectorAll('.rc-slider-tooltip-arrow').forEach((node) => {
        node.style.setProperty('border-top-color', '#64748b', 'important');
        node.style.setProperty('border-bottom-color', '#64748b', 'important');
      });
    });

    document.querySelectorAll('#proj-return-slider + div, #proj-contrib-slider + div').forEach((node) => {
      node.style.setProperty('color', '#ffffff', 'important');
      node.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
    });

    document.querySelectorAll('.rc-slider-tooltip-inner, .rc-tooltip-inner').forEach((node) => {
      node.style.setProperty('background-color', '#1e293b', 'important');
      node.style.setProperty('color', '#ffffff', 'important');
      node.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
      node.style.setProperty('border', '1px solid #64748b', 'important');
      node.style.setProperty('border-radius', '6px', 'important');
      node.style.setProperty('box-shadow', '0 2px 8px rgba(0,0,0,0.35)', 'important');
    });

    document.querySelectorAll('.rc-slider-tooltip-content, .rc-tooltip-content').forEach((node) => {
      node.style.setProperty('background-color', '#1e293b', 'important');
      node.style.setProperty('border', '1px solid #64748b', 'important');
      node.style.setProperty('border-radius', '6px', 'important');
    });

    document.querySelectorAll('.rc-slider-tooltip-arrow, .rc-tooltip-arrow').forEach((node) => {
      node.style.setProperty('border-top-color', '#64748b', 'important');
      node.style.setProperty('border-bottom-color', '#64748b', 'important');
    });

    // Final fallback: any tooltip-like element inside the projection slider band
    // gets force-dark styling to kill iPad white boxes regardless of class variant.
    const projRects = sliderRoots
      .map((root) => root.getBoundingClientRect())
      .filter((rect) => rect && rect.width > 0 && rect.height > 0);

    if (projRects.length) {
      const top = Math.min(...projRects.map((r) => r.top)) - 80;
      const bottom = Math.max(...projRects.map((r) => r.bottom)) + 120;

      const tooltipCandidates = document.querySelectorAll(
        '.rc-slider-tooltip, .rc-slider-tooltip-inner, .rc-slider-tooltip-content, .rc-tooltip, .rc-tooltip-inner, .rc-tooltip-content, .tooltip, .tooltip-inner, [role="tooltip"], [class*="tooltip"], [class*="Tooltip"]'
      );

      tooltipCandidates.forEach((node) => {
        const rect = node.getBoundingClientRect();
        if (!rect || rect.width === 0 || rect.height === 0) return;

        const inBand = rect.bottom >= top && rect.top <= bottom;
        const tooltipSized = rect.width <= 260 && rect.height <= 140;
        if (!inBand || !tooltipSized) return;

        const text = (node.textContent || '').replace(/\u00a0/g, ' ').trim();
        const hasVisibleText = text.length > 0;
        const hasValueDescendant = !!node.querySelector(
          '.rc-slider-tooltip-inner, .rc-tooltip-inner, [class*="tooltip-inner"], [role="tooltip"]'
        );

        if (!hasVisibleText && !hasValueDescendant) {
          node.style.setProperty('display', 'none', 'important');
          node.style.setProperty('visibility', 'hidden', 'important');
          node.style.setProperty('opacity', '0', 'important');
          return;
        }

        node.style.setProperty('background-color', '#1e293b', 'important');
        node.style.setProperty('color', '#ffffff', 'important');
        node.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
        node.style.setProperty('border', '1px solid #64748b', 'important');
        node.style.setProperty('border-radius', '6px', 'important');
        node.style.setProperty('box-shadow', '0 2px 8px rgba(0,0,0,0.35)', 'important');
      });

      // Absolute fallback for iPad ghost rectangles: hide any small bright,
      // textless box in the projection slider band, regardless of class name.
      document.querySelectorAll('body *').forEach((node) => {
        const rect = node.getBoundingClientRect();
        if (!rect || rect.width < 24 || rect.height < 16 || rect.width > 280 || rect.height > 140) return;

        const inBand = rect.bottom >= top && rect.top <= bottom;
        if (!inBand) return;

        const text = (node.textContent || '').replace(/\u00a0/g, ' ').trim();
        if (text.length > 0) return;

        const cs = window.getComputedStyle(node);
        const bg = cs.backgroundColor || '';
        const border = cs.borderColor || '';
        const bgBright = /rgba?\((2[0-5][0-5]|1\d\d),\s*(2[0-5][0-5]|1\d\d),\s*(2[0-5][0-5]|1\d\d)(?:,\s*(0\.[3-9]|1(?:\.0)?))?\)/i.test(bg);
        const borderBright = /rgba?\((2[0-5][0-5]|1\d\d),\s*(2[0-5][0-5]|1\d\d),\s*(2[0-5][0-5]|1\d\d)(?:,\s*(0\.[3-9]|1(?:\.0)?))?\)/i.test(border);
        if (!bgBright && !borderBright) return;

        node.style.setProperty('display', 'none', 'important');
        node.style.setProperty('visibility', 'hidden', 'important');
        node.style.setProperty('opacity', '0', 'important');
      });
    }
  }

  function styleSimulatorSliders() {
    if (!shouldForce()) return;

    const simRoot = document.getElementById('simulator-sliders-container');
    if (!simRoot) return;

    simRoot.style.setProperty('color-scheme', 'dark', 'important');

    simRoot.querySelectorAll('.rc-slider-rail').forEach((node) => {
      node.style.setProperty('background-color', '#344055', 'important');
      node.style.setProperty('opacity', '1', 'important');
    });

    simRoot.querySelectorAll('.rc-slider-track').forEach((node) => {
      node.style.setProperty('background-color', '#8b5cf6', 'important');
      node.style.setProperty('opacity', '1', 'important');
    });

    simRoot.querySelectorAll('.rc-slider-handle').forEach((node) => {
      node.style.setProperty('background-color', '#8b5cf6', 'important');
      node.style.setProperty('border-color', '#d8b4fe', 'important');
      node.style.setProperty('border-width', '2px', 'important');
      node.style.setProperty('border-style', 'solid', 'important');
      node.style.setProperty('border-radius', '999px', 'important');
      node.style.setProperty('width', '16px', 'important');
      node.style.setProperty('height', '16px', 'important');
      node.style.setProperty('opacity', '1', 'important');
      node.style.setProperty('box-shadow', '0 0 0 2px rgba(139,92,246,0.35)', 'important');
      node.style.setProperty('-webkit-appearance', 'none', 'important');
      node.style.setProperty('appearance', 'none', 'important');
      node.style.setProperty('outline', 'none', 'important');
    });

    simRoot.querySelectorAll('.rc-slider-dot').forEach((node) => {
      node.style.setProperty('background-color', '#151c24', 'important');
      node.style.setProperty('border-color', '#64748b', 'important');
      node.style.setProperty('border-radius', '999px', 'important');
    });

    simRoot.querySelectorAll('.rc-slider-mark-text, label, span, div').forEach((node) => {
      node.style.setProperty('color', '#e2e8f0', 'important');
      node.style.setProperty('-webkit-text-fill-color', '#e2e8f0', 'important');
    });

    simRoot.querySelectorAll('input, textarea, [contenteditable="true"]').forEach((node) => {
      node.style.setProperty('background-color', 'transparent', 'important');
      node.style.setProperty('color', '#e2e8f0', 'important');
      node.style.setProperty('-webkit-text-fill-color', '#e2e8f0', 'important');
      node.style.setProperty('border-color', 'rgba(255,255,255,0.2)', 'important');
    });
  }

  function styleProjectionChart() {
    if (!shouldForce()) return;

    const chartRoot = document.getElementById('projections-chart');
    if (!chartRoot) return;

    chartRoot.style.setProperty('color-scheme', 'dark', 'important');
    chartRoot.style.setProperty('background-color', '#151c24', 'important');

    chartRoot.querySelectorAll('.js-plotly-plot, .plot-container, .svg-container').forEach((node) => {
      node.style.setProperty('background-color', '#151c24', 'important');
    });

    chartRoot.querySelectorAll('.xtick text, .ytick text, .gtitle, .xtitle, .ytitle, .legend text, .legendtext, .annotation-text').forEach((node) => {
      node.style.setProperty('fill', '#e2e8f0', 'important');
      node.style.setProperty('color', '#e2e8f0', 'important');
      node.style.setProperty('-webkit-text-fill-color', '#e2e8f0', 'important');
    });

    chartRoot.querySelectorAll('.hoverlayer .hovertext, .hoverlayer .name, .hoverlayer .nums').forEach((node) => {
      node.style.setProperty('fill', '#ffffff', 'important');
      node.style.setProperty('color', '#ffffff', 'important');
      node.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
    });

    chartRoot.querySelectorAll('.modebar, .modebar-container').forEach((node) => {
      node.style.setProperty('background-color', 'rgba(21,28,36,0.85)', 'important');
    });

    chartRoot.querySelectorAll('.modebar-btn').forEach((node) => {
      node.style.setProperty('color', '#ffffff', 'important');
      node.style.setProperty('fill', '#ffffff', 'important');
    });
  }

  function runForcePass() {
    // Always force overflow on the preset dropdown card (not gated by width)
    forcePresetOverflow();
    // Nuclear: dark-paint any open menu anywhere in the DOM (only needs dark theme, not width)
    nukeAllOpenMenus();
    // Width-gated passes for the rest
    styleBenchmarkRoot();
    styleLikelyOpenMenu();
    styleProjectionSliders();
    styleSimulatorSliders();
    styleProjectionChart();
    // If a menu is open, start rapid polling to keep it painted
    if (document.querySelector('.Select-menu-outer, .Select.is-open')) {
      startMenuPoll();
    }
  }

  const observer = new MutationObserver(() => {
    runForcePass();
  });

  function init() {
    runForcePass();
    observer.observe(document.documentElement, { childList: true, subtree: true, attributes: true });
    window.addEventListener('resize', runForcePass, { passive: true });
    window.addEventListener('orientationchange', runForcePass, { passive: true });
    document.addEventListener('click', function () {
      runForcePass();
      // Delayed pass to catch react-select async renders after click
      setTimeout(runForcePass, 50);
      setTimeout(runForcePass, 150);
      setTimeout(runForcePass, 300);
    }, true);
    document.addEventListener('focusin', function () {
      runForcePass();
      setTimeout(runForcePass, 50);
      setTimeout(runForcePass, 150);
    }, true);
    document.addEventListener('touchend', function () {
      runForcePass();
      setTimeout(runForcePass, 80);
      setTimeout(runForcePass, 200);
      setTimeout(runForcePass, 400);
    }, true);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
