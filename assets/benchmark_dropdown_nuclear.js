(function () {
  'use strict';

  const MAX_WIDTH = 1400;

  function shouldForce() {
    const root = document.querySelector('[data-theme="dark"]');
    return !!root && window.innerWidth <= MAX_WIDTH;
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

  function styleBenchmarkRoot() {
    if (!shouldForce()) return;

    const roots = [
      document.getElementById('benchmark-dropdown'),
      document.getElementById('strategy-preset-checklist')
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
        const bg = node.matches('[role="option"], .Select-option, [class*="option"], [class*="Option"]')
          ? '#151c24'
          : '#151c24';
        paint(node, bg);
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
    styleBenchmarkRoot();
    styleLikelyOpenMenu();
    styleProjectionSliders();
    styleSimulatorSliders();
    styleProjectionChart();
  }

  const observer = new MutationObserver(() => {
    runForcePass();
  });

  function init() {
    runForcePass();
    observer.observe(document.documentElement, { childList: true, subtree: true, attributes: true });
    window.addEventListener('resize', runForcePass, { passive: true });
    window.addEventListener('orientationchange', runForcePass, { passive: true });
    document.addEventListener('click', runForcePass, true);
    document.addEventListener('focusin', runForcePass, true);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
