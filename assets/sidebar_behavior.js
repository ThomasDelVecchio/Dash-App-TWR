/**
 * Sidebar Manual Toggle Controller
 * 
 * Features:
 * - Manual collapse/expand via icon button
 * - Close (hide) via X button
 * - Reopen via floating open button
 */

(function() {
    'use strict';
    
    let sidebarElement = null;
    let contentElement = null;
    let toggleButton = null;
    let collapseButton = null;
    let collapseIcon = null;
    let openButton = null;
    
    function init() {
        sidebarElement = document.getElementById('sidebar');
        contentElement = document.getElementById('page-content');
        toggleButton = document.getElementById('btn-sidebar-toggle');
        collapseButton = document.getElementById('btn-sidebar-collapse');
        collapseIcon = document.getElementById('sidebar-collapse-icon');
        openButton = document.getElementById('btn-sidebar-open');
        
        if (!sidebarElement) {
            // Retry after DOM loads
            setTimeout(init, 100);
            return;
        }
        
        setupManualToggle();
        syncCollapseIcon();
    }
    
    function updateOpenButtonVisibility() {
        if (!openButton) return;
        
        if (sidebarElement.classList.contains('hidden')) {
            // Show the floating open button
            openButton.style.display = 'flex';
        } else {
            // Hide the floating open button
            openButton.style.display = 'none';
        }
    }
    
    function hideSidebar() {
        if (!sidebarElement) return;
        
        // Add collapsed class for icon-only mode on desktop
        if (window.innerWidth >= 992) {
            sidebarElement.classList.add('collapsed');
            if (contentElement) contentElement.classList.add('sidebar-collapsed');
        } else {
            // Full hide on mobile/tablet
            sidebarElement.classList.add('hidden');
            if (contentElement) contentElement.classList.add('expanded');
        }
        
        updateOpenButtonVisibility();
        syncCollapseIcon();
    }
    
    function showSidebar() {
        if (!sidebarElement) return;
        
        sidebarElement.classList.remove('hidden', 'collapsed');
        if (contentElement) contentElement.classList.remove('expanded', 'sidebar-collapsed');
        
        updateOpenButtonVisibility();
        syncCollapseIcon();
    }
    
    function toggleCollapsed() {
        if (!sidebarElement) return;
        
        if (window.innerWidth >= 992) {
            // Desktop: icon-only collapse
            if (sidebarElement.classList.contains('collapsed')) {
                sidebarElement.classList.remove('collapsed');
                if (contentElement) contentElement.classList.remove('sidebar-collapsed');
            } else {
                sidebarElement.classList.add('collapsed');
                if (contentElement) contentElement.classList.add('sidebar-collapsed');
            }
        } else {
            // Mobile/tablet: full hide/show
            if (sidebarElement.classList.contains('hidden')) {
                showSidebar();
            } else {
                hideSidebar();
            }
        }
        
        updateOpenButtonVisibility();
        syncCollapseIcon();
    }

    function syncCollapseIcon() {
        if (!collapseIcon || !sidebarElement) return;

        const isCollapsed = sidebarElement.classList.contains('collapsed');
        if (collapseButton) {
            collapseButton.title = isCollapsed ? 'Expand sidebar' : 'Collapse sidebar';
        }
        if (isCollapsed) {
            collapseIcon.classList.remove('bi-layout-sidebar-inset');
            collapseIcon.classList.add('bi-layout-sidebar-inset-reverse');
        } else {
            collapseIcon.classList.remove('bi-layout-sidebar-inset-reverse');
            collapseIcon.classList.add('bi-layout-sidebar-inset');
        }
    }
    
    function setupManualToggle() {
        // Close button inside sidebar header - hide sidebar
        if (toggleButton) {
            toggleButton.addEventListener('click', function() {
                hideSidebar();
            });
        }
        
        // Collapse button inside sidebar header - toggle collapse
        if (collapseButton) {
            collapseButton.addEventListener('click', function() {
                toggleCollapsed();
            });
        }
        
        // Open button (floating) - show sidebar
        if (openButton) {
            openButton.addEventListener('click', function() {
                showSidebar();
            });
        }
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
    // Re-initialize on Dash page navigation (client-side routing)
    // MutationObserver watches for content changes
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList' && mutation.target.id === 'page-content') {
                // Page changed, ensure sidebar refs are current
                sidebarElement = document.getElementById('sidebar');
                contentElement = document.getElementById('page-content');
            }
        });
    });
    
    // Start observing once DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
        const pageContent = document.getElementById('page-content');
        if (pageContent) {
            observer.observe(pageContent, { childList: true });
        }
    });
    
})();

/**
 * Phase 3 — Storytelling Card Mobile Tap-to-Expand
 * Toggles .story-card--expanded on tap for mobile viewports.
 */
(function() {
    'use strict';
    document.addEventListener('click', function(e) {
        if (window.innerWidth > 992) return;
        var card = e.target.closest('.story-card');
        if (!card) return;
        card.classList.toggle('story-card--expanded');
    });
})();

/**
 * Phase 3 — Count-Up Animation for Hero Metrics
 * Animates .story-countup elements from 0 to data-final when they
 * scroll into view. Uses IntersectionObserver for efficiency.
 */
(function() {
    'use strict';

    function animateValue(el) {
        var raw = (el.getAttribute('data-final') || '').trim();
        if (!raw || el.dataset.animated === '1') return;
        el.dataset.animated = '1';

        // Decompose: detect prefix (e.g. "$", "-$"), suffix (e.g. "%"), commas
        var match = raw.match(/^([^0-9\-]*-?\$?\s*)?(-?[\d,]+\.?\d*)\s*(%?)(.*)$/);
        if (!match) { el.textContent = raw; return; }

        var prefix  = match[1] || '';
        var numStr  = match[2].replace(/,/g, '');
        var pct     = match[3] || '';
        var suffix  = match[4] || '';
        var target  = parseFloat(numStr);
        if (isNaN(target)) { el.textContent = raw; return; }

        var decimals = (numStr.split('.')[1] || '').length;
        var useCommas = match[2].indexOf(',') !== -1;
        var duration = 800; // ms
        var startTime = null;

        function formatNum(v) {
            var s = v.toFixed(decimals);
            if (useCommas) {
                var parts = s.split('.');
                parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');
                s = parts.join('.');
            }
            return s;
        }

        function step(ts) {
            if (!startTime) startTime = ts;
            var progress = Math.min((ts - startTime) / duration, 1);
            // ease-out-expo
            var ease = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
            var current = target * ease;
            el.textContent = prefix + formatNum(current) + pct + suffix;
            if (progress < 1) requestAnimationFrame(step);
        }

        el.textContent = prefix + formatNum(0) + pct + suffix;
        requestAnimationFrame(step);
    }

    function observe() {
        var els = document.querySelectorAll('.story-countup[data-final]');
        if (!els.length) return;

        if ('IntersectionObserver' in window) {
            var io = new IntersectionObserver(function(entries) {
                entries.forEach(function(entry) {
                    if (entry.isIntersecting) {
                        animateValue(entry.target);
                        io.unobserve(entry.target);
                    }
                });
            }, { threshold: 0.3 });
            els.forEach(function(el) { io.observe(el); });
        } else {
            els.forEach(animateValue);
        }
    }

    // Observe on load and after Dash re-renders
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', observe);
    } else {
        observe();
    }

    // Re-observe after Dash client-side navigation
    var bodyObserver = new MutationObserver(function() {
        setTimeout(observe, 100);
    });
    document.addEventListener('DOMContentLoaded', function() {
        var pc = document.getElementById('page-content');
        if (pc) bodyObserver.observe(pc, { childList: true, subtree: true });
    });
})();
