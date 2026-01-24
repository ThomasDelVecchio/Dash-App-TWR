/**
 * Sidebar Auto-Hide & Touch Gesture Controller
 * 
 * Features:
 * - Auto-hide after 5 seconds of inactivity (desktop & mobile)
 * - Swipe-left to close on mobile
 * - Hover to expand on desktop (when in icon-only mode)
 * - Touch swipe-right from edge to open on mobile
 */

(function() {
    'use strict';
    
    let hideTimer = null;
    let touchStartX = 0;
    let touchStartY = 0;
    let sidebarElement = null;
    let contentElement = null;
    let toggleButton = null;
    
    const HIDE_DELAY = 5000; // 5 seconds
    const SWIPE_THRESHOLD = 80; // pixels
    const EDGE_ZONE = 30; // pixels from edge to trigger swipe-open
    
    function init() {
        sidebarElement = document.getElementById('sidebar');
        contentElement = document.getElementById('page-content');
        toggleButton = document.getElementById('btn-sidebar-toggle');
        
        if (!sidebarElement) {
            // Retry after DOM loads
            setTimeout(init, 100);
            return;
        }
        
        setupAutoHide();
        setupTouchGestures();
        setupHoverExpand();
        
        // Start the auto-hide timer on page load
        resetHideTimer();
    }
    
    function resetHideTimer() {
        clearTimeout(hideTimer);
        hideTimer = setTimeout(hideSidebar, HIDE_DELAY);
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
    }
    
    function showSidebar() {
        if (!sidebarElement) return;
        
        sidebarElement.classList.remove('hidden', 'collapsed');
        if (contentElement) contentElement.classList.remove('expanded', 'sidebar-collapsed');
        
        resetHideTimer();
    }
    
    function setupAutoHide() {
        // Reset timer on any interaction with sidebar
        const interactionEvents = ['mouseenter', 'mousemove', 'click', 'touchstart', 'scroll'];
        
        interactionEvents.forEach(event => {
            sidebarElement.addEventListener(event, function(e) {
                resetHideTimer();
            }, { passive: true });
        });
        
        // Also reset when toggle button is clicked (handled by Dash callback, but we track it)
        if (toggleButton) {
            toggleButton.addEventListener('click', function() {
                // If sidebar was hidden, show it and reset timer
                if (sidebarElement.classList.contains('hidden') || 
                    sidebarElement.classList.contains('collapsed')) {
                    showSidebar();
                }
                resetHideTimer();
            });
        }
    }
    
    function setupTouchGestures() {
        // Swipe on sidebar to close
        sidebarElement.addEventListener('touchstart', function(e) {
            touchStartX = e.touches[0].clientX;
            touchStartY = e.touches[0].clientY;
        }, { passive: true });
        
        sidebarElement.addEventListener('touchend', function(e) {
            const touchEndX = e.changedTouches[0].clientX;
            const touchEndY = e.changedTouches[0].clientY;
            
            const deltaX = touchStartX - touchEndX;
            const deltaY = Math.abs(touchStartY - touchEndY);
            
            // Swipe left to close (horizontal swipe, not vertical scroll)
            if (deltaX > SWIPE_THRESHOLD && deltaY < 50) {
                hideSidebar();
            }
        }, { passive: true });
        
        // Swipe from left edge to open (global listener)
        document.addEventListener('touchstart', function(e) {
            const startX = e.touches[0].clientX;
            
            // Only trigger if starting from edge zone
            if (startX <= EDGE_ZONE) {
                touchStartX = startX;
                touchStartY = e.touches[0].clientY;
            } else {
                touchStartX = -1; // Invalid, ignore
            }
        }, { passive: true });
        
        document.addEventListener('touchend', function(e) {
            if (touchStartX < 0) return; // Not an edge swipe
            
            const touchEndX = e.changedTouches[0].clientX;
            const touchEndY = e.changedTouches[0].clientY;
            
            const deltaX = touchEndX - touchStartX;
            const deltaY = Math.abs(touchStartY - touchEndY);
            
            // Swipe right from edge to open
            if (deltaX > SWIPE_THRESHOLD && deltaY < 50 && 
                (sidebarElement.classList.contains('hidden') || 
                 sidebarElement.classList.contains('collapsed'))) {
                showSidebar();
            }
            
            touchStartX = -1; // Reset
        }, { passive: true });
    }
    
    function setupHoverExpand() {
        // Desktop only: hover to expand from icon-only mode
        sidebarElement.addEventListener('mouseenter', function() {
            if (window.innerWidth >= 992 && sidebarElement.classList.contains('collapsed')) {
                sidebarElement.classList.remove('collapsed');
                if (contentElement) contentElement.classList.remove('sidebar-collapsed');
                resetHideTimer();
            }
        });
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
