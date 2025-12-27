document.addEventListener('DOMContentLoaded', function() {
    // Function to initialize draggable behavior
    function makeDraggable() {
        const btn = document.getElementById('btn-chatbot-toggle');
        if (!btn) return;
        
        // Avoid double initialization
        if (btn.dataset.draggableInitialized) return;
        btn.dataset.draggableInitialized = 'true';

        let isDragging = false;
        let startX, startY;
        let offsetX, offsetY;

        // --- MOUSE EVENTS ---
        
        btn.addEventListener('mousedown', function(e) {
            if (e.button !== 0) return;
            
            startX = e.clientX;
            startY = e.clientY;
            
            const rect = btn.getBoundingClientRect();
            offsetX = e.clientX - rect.left;
            offsetY = e.clientY - rect.top;
            
            isDragging = false;
            
            function onMouseMove(e) {
                const dx = e.clientX - startX;
                const dy = e.clientY - startY;
                
                if (!isDragging && Math.sqrt(dx*dx + dy*dy) > 5) {
                    isDragging = true;
                    document.body.style.userSelect = 'none';
                    btn.style.bottom = 'auto';
                    btn.style.right = 'auto';
                }
                
                if (isDragging) {
                    const newLeft = e.clientX - offsetX;
                    const newTop = e.clientY - offsetY;
                    
                    const maxLeft = window.innerWidth - btn.offsetWidth;
                    const maxTop = window.innerHeight - btn.offsetHeight;
                    
                    btn.style.left = Math.min(Math.max(0, newLeft), maxLeft) + 'px';
                    btn.style.top = Math.min(Math.max(0, newTop), maxTop) + 'px';
                    btn.style.position = 'fixed';
                }
            }
            
            function onMouseUp(e) {
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
                document.body.style.userSelect = '';
                
                if (isDragging) {
                    const captureClick = function(clickEvent) {
                        clickEvent.stopPropagation();
                        clickEvent.stopImmediatePropagation();
                        clickEvent.preventDefault();
                        btn.removeEventListener('click', captureClick, true);
                    };
                    btn.addEventListener('click', captureClick, true);
                    isDragging = false;
                }
            }
            
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });

        // --- TOUCH EVENTS ---

        btn.addEventListener('touchstart', function(e) {
            // Prevent default to stop scrolling while dragging
            // but we only prevent default if we determine it's a drag?
            // Actually, for a floating button, usually safe to prevent default on start to catch the drag.
            // But checking if we want to allow tap.
            
            if (e.touches.length > 1) return;
            
            const touch = e.touches[0];
            startX = touch.clientX;
            startY = touch.clientY;
            
            const rect = btn.getBoundingClientRect();
            offsetX = touch.clientX - rect.left;
            offsetY = touch.clientY - rect.top;
            
            isDragging = false;
            
            function onTouchMove(e) {
                const touch = e.touches[0];
                const dx = touch.clientX - startX;
                const dy = touch.clientY - startY;
                
                if (!isDragging && Math.sqrt(dx*dx + dy*dy) > 5) {
                    isDragging = true;
                    document.body.style.userSelect = 'none';
                    btn.style.bottom = 'auto';
                    btn.style.right = 'auto';
                }
                
                if (isDragging) {
                    // Prevent scrolling
                    e.preventDefault();
                    
                    const newLeft = touch.clientX - offsetX;
                    const newTop = touch.clientY - offsetY;
                    
                    const maxLeft = window.innerWidth - btn.offsetWidth;
                    const maxTop = window.innerHeight - btn.offsetHeight;
                    
                    btn.style.left = Math.min(Math.max(0, newLeft), maxLeft) + 'px';
                    btn.style.top = Math.min(Math.max(0, newTop), maxTop) + 'px';
                    btn.style.position = 'fixed';
                }
            }
            
            function onTouchEnd(e) {
                document.removeEventListener('touchmove', onTouchMove);
                document.removeEventListener('touchend', onTouchEnd);
                document.body.style.userSelect = '';
                
                if (isDragging) {
                    // Prevent click if it was a drag
                    e.preventDefault(); 
                    
                    const captureClick = function(clickEvent) {
                        clickEvent.stopPropagation();
                        clickEvent.stopImmediatePropagation();
                        clickEvent.preventDefault();
                        btn.removeEventListener('click', captureClick, true);
                    };
                    btn.addEventListener('click', captureClick, true);
                    
                    isDragging = false;
                }
            }
            
            // Add passive: false to allow preventDefault inside touchmove
            document.addEventListener('touchmove', onTouchMove, { passive: false });
            document.addEventListener('touchend', onTouchEnd);
        }, { passive: false });
    }

    // Since Dash renders components dynamically, we use a MutationObserver
    // to detect when the chatbot button is added to the DOM.
    const observer = new MutationObserver(function(mutations) {
        makeDraggable();
    });

    observer.observe(document.body, { childList: true, subtree: true });
    
    // Attempt initial setup
    makeDraggable();
});
