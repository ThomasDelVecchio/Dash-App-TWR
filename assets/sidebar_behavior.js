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
