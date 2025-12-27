var dagfuncs = window.dashAgGridFunctions = window.dashAgGridFunctions || {};

// Helper for parsing currency/percent strings
function parseValue(v) {
    if (typeof v === 'number') return v;
    if (v === null || v === undefined) return -Infinity;
    
    var s = String(v).trim();
    if (s === "N/A" || s === "") return -Infinity;
    
    // Handle (Value) as negative
    var sign = 1;
    if (s.includes('(') && s.includes(')')) {
        sign = -1;
        s = s.replace(/[()]/g, '');
    }
    
    // Remove symbols: $ , %
    s = s.replace(/[$,%]/g, '');
    
    var f = parseFloat(s);
    if (isNaN(f)) return -Infinity;
    return f * sign;
}

// Comparator for Grouped Tables (Horizon Returns/PL)
dagfuncs.GroupedRowComparator = function (valueA, valueB, nodeA, nodeB, isDescending) {
    // 1. Sort by Asset Class Rank (Fixed Order)
    var rankA = nodeA.data._sort_rank;
    var rankB = nodeB.data._sort_rank;
    
    if (rankA === undefined || rankA === null) rankA = nodeA.data.asset_class || "";
    if (rankB === undefined || rankB === null) rankB = nodeB.data.asset_class || "";

    if (rankA !== rankB) {
        var res = 0;
        if (typeof rankA === 'string' && typeof rankB === 'string') {
            res = rankA.localeCompare(rankB);
        } else {
            res = (rankA < rankB) ? -1 : 1;
        }
        return isDescending ? -res : res; // Counteract flip
    }
    
    // 2. Header Top
    var headerA = nodeA.data._is_header || 0;
    var headerB = nodeB.data._is_header || 0;
    
    if (headerA !== headerB) {
        var res = headerB - headerA; 
        return isDescending ? -res : res; // Counteract flip
    }
    
    // 3. Value
    if (valueA === valueB) return 0;
    
    var numA = parseValue(valueA);
    var numB = parseValue(valueB);
    
    if (numA !== -Infinity || numB !== -Infinity) {
        if (numA === -Infinity) return isDescending ? -1 : 1; 
        if (numB === -Infinity) return isDescending ? 1 : -1;
        return numA - numB;
    }
    
    return String(valueA).localeCompare(String(valueB));
};

// Comparator for Standard Tables (Money/Percent)
dagfuncs.MoneyComparator = function (valueA, valueB, nodeA, nodeB, isDescending) {
    if (valueA === valueB) return 0;
    
    var numA = parseValue(valueA);
    var numB = parseValue(valueB);
    
    if (numA !== -Infinity || numB !== -Infinity) {
        // Handle N/A at bottom
        if (numA === -Infinity) return isDescending ? -1 : 1; 
        if (numB === -Infinity) return isDescending ? 1 : -1;
        return numA - numB;
    }
    
    return String(valueA).localeCompare(String(valueB));
};

// ============================================================
// AUDIT TRAIL EVENT LISTENERS
// ============================================================

document.addEventListener('DOMContentLoaded', function() {
    // Only attach once
    if (window.auditListenerAttached) return;
    window.auditListenerAttached = true;
    
    // Helper to trigger audit
    function triggerAudit(cell) {
        var gridDiv = cell.closest('.audit-target');
        if (!gridDiv) return;
        
        var gridId = gridDiv.id;
        var colId = cell.getAttribute('col-id');
        
        // Use Dash AG Grid API
        dash_ag_grid.getApiAsync(gridId).then((api) => {
            // Find Row Index
            var rowEl = cell.closest('.ag-row');
            if (!rowEl) return;
            
            var rowIndex = rowEl.getAttribute('row-index');
            var idx = parseInt(rowIndex);
            var rowNode = null;

            // Check if pinned
            if (rowEl.closest('.ag-floating-bottom-container')) {
                rowNode = api.getPinnedBottomRow(idx);
            } else if (rowEl.closest('.ag-floating-top-container')) {
                rowNode = api.getPinnedTopRow(idx);
            } else {
                rowNode = api.getDisplayedRowAtIndex(idx);
            }
            
            if (!rowNode) return;
            
            // FIX: Get Field Name from Column ID
            var column = api.getColumn(colId);
            var field = colId; // Fallback
            if (column && column.getColDef()) {
                field = column.getColDef().field;
            }
            
            var payload = {
                gridId: gridId,
                colId: field, // Send FIELD name to backend
                rowIndex: rowIndex,
                rowData: rowNode.data,
                value: rowNode.data[field]
            };
            
            // Send to Dash via Store
            if (window.dash_clientside && window.dash_clientside.set_props) {
                window.dash_clientside.set_props("audit-request-store", {data: payload});
            } else {
                console.warn("Audit Trail: dash_clientside.set_props not available.");
            }
        });
    }

    // 1. Right Click (Desktop)
    document.addEventListener('contextmenu', function(e) {
        var cell = e.target.closest('.audit-target .ag-cell');
        if (!cell) return;
        
        // Prevent default context menu
        e.preventDefault();
        triggerAudit(cell);
    });
    
    // 2. Long Press (Touch)
    var touchTimer = null;
    var touchDuration = 600; // ms
    
    document.addEventListener('touchstart', function(e) {
        var cell = e.target.closest('.audit-target .ag-cell');
        if (!cell) return;
        
        touchTimer = setTimeout(function() {
            triggerAudit(cell);
        }, touchDuration);
    }, {passive: true});
    
    document.addEventListener('touchend', function() {
        if (touchTimer) clearTimeout(touchTimer);
    });
    
    document.addEventListener('touchmove', function() {
        if (touchTimer) clearTimeout(touchTimer);
    });
});
