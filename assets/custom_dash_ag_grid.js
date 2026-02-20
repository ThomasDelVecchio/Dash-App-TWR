var dagfuncs = window.dashAgGridFunctions = window.dashAgGridFunctions || {};
var dagcomponentfuncs = window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {};

// ============================================================
// SPARKLINE CELL RENDERER (Phase 3 — In-Cell 30-Day Trends)
// ============================================================
dagcomponentfuncs.SparklineRenderer = function (props) {
    var h = window.React.createElement;

    // props.value is a JSON-encoded array of normalised [0..1] values
    var raw = props.value;
    var points;

    if (!raw || raw === '[]' || raw === '') {
        return h('div', {className: 'sparkline-nodata'}, '—');
    }

    try {
        points = typeof raw === 'string' ? JSON.parse(raw) : raw;
    } catch (e) {
        return h('div', {className: 'sparkline-nodata'}, '—');
    }

    if (!Array.isArray(points) || points.length < 2) {
        return h('div', {className: 'sparkline-nodata'}, '—');
    }

    // SVG dimensions
    var W = 100, H = 28, PAD = 1;
    var n = points.length;
    var xStep = (W - 2 * PAD) / (n - 1);

    // Build polyline string (Y is inverted: 0 = top)
    var polyPoints = points.map(function(v, i) {
        var x = PAD + i * xStep;
        var y = PAD + (1 - v) * (H - 2 * PAD);
        return x.toFixed(1) + ',' + y.toFixed(1);
    }).join(' ');

    // Determine colour based on trend direction
    var first = points[0], last = points[points.length - 1];
    var lineColor = last >= first ? '#22c55e' : '#ef4444';
    var fillColor = last >= first ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)';

    // Fill polygon: polyline + bottom-right + bottom-left
    var firstX = PAD, lastX = PAD + (n - 1) * xStep;
    var fillPoints = polyPoints + ' ' + lastX.toFixed(1) + ',' + H + ' ' + firstX.toFixed(1) + ',' + H;

    return h('div', {className: 'sparkline-cell'},
        h('svg', {
            width: W,
            height: H,
            viewBox: '0 0 ' + W + ' ' + H,
            style: {display: 'block'}
        },
            // Fill area
            h('polygon', {
                points: fillPoints,
                fill: fillColor,
                stroke: 'none'
            }),
            // Line
            h('polyline', {
                points: polyPoints,
                fill: 'none',
                stroke: lineColor,
                strokeWidth: '1.5',
                strokeLinecap: 'round',
                strokeLinejoin: 'round'
            }),
            // End dot
            h('circle', {
                cx: (PAD + (n - 1) * xStep).toFixed(1),
                cy: (PAD + (1 - last) * (H - 2 * PAD)).toFixed(1),
                r: '2',
                fill: lineColor
            })
        )
    );
};

// ============================================================
// DATA BAR CELL RENDERER (Phase 1 — In-Cell Bars)
// ============================================================
dagcomponentfuncs.DataBarRenderer = function (props) {
    // Expects: props.value = formatted string like "12.34%" or "-2.50%"
    //          props.colDef.cellRendererParams.maxVal (optional, default 100)
    //          props.colDef.cellRendererParams.field  (optional raw numeric field)
    var rawVal;
    var displayVal = props.value;

    // Try to get raw numeric from a meta field first (more accurate)
    var rendererParams = props.colDef.cellRendererParams || {};
    var metaField = rendererParams.field;
    if (metaField && props.data && props.data[metaField] !== undefined) {
        rawVal = parseFloat(props.data[metaField]);
    }
    if (isNaN(rawVal) || rawVal === undefined || rawVal === null) {
        rawVal = parseValue(displayVal);
    }
    if (rawVal === -Infinity || isNaN(rawVal)) rawVal = 0;

    var maxVal = rendererParams.maxVal || 100;
    var pct = Math.min(Math.abs(rawVal) / maxVal * 100, 100);

    // colorMode: "accent" = neutral blue bar (for weights), "semantic" = green/red (for deltas)
    var colorMode = rendererParams.colorMode || 'semantic';
    var fillClass, textClass;
    if (colorMode === 'accent') {
        fillClass = 'data-bar-fill--accent';
        textClass = 'data-bar-text--accent';
    } else {
        var isNeg = rawVal < -0.005;
        var isPos = rawVal > 0.005;
        fillClass = isNeg ? 'data-bar-fill--negative' : (isPos ? 'data-bar-fill--positive' : 'data-bar-fill--neutral');
        textClass = isNeg ? 'data-bar-text--negative' : (isPos ? 'data-bar-text--positive' : 'data-bar-text--neutral');
    }

    // Build React elements
    var h = window.React.createElement;
    return h('div', {className: 'data-bar-cell'}, 
        h('div', {className: 'data-bar-fill ' + fillClass, style: {width: pct + '%'}}),
        h('span', {className: 'data-bar-text ' + textClass}, displayVal || '0.00%')
    );
};
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
// StageOrderButton - Stages order to dcc.Store and navigates to Trade page
// ============================================================
dagfuncs.StageOrderButton = function (props) {
    // Show button for all actions - disabled for Hold, enabled for Buy/Sell
    var action = props.data.Action;
    
    // For Hold or missing actions, show a disabled button
    if (!action || action === "Hold" || action === "") {
        var holdBtn = document.createElement('button');
        holdBtn.className = 'btn btn-sm btn-outline-secondary';
        holdBtn.innerText = 'Hold';
        holdBtn.disabled = true;
        holdBtn.style.opacity = '0.5';
        holdBtn.style.cursor = 'not-allowed';
        return holdBtn;
    }
    
    // Extract data from row
    var ticker = props.data.Ticker || "";
    var sharesStr = props.data.Shares || "0";
    
    // Parse shares (remove formatting like commas)
    var shares = parseFloat(sharesStr.toString().replace(/[^0-9.-]/g, "")) || 0;
    
    // Get price from hidden meta column (meta_price)
    var price = props.data.meta_price || 0;
    
    // Build payload for staged-order-store
    var payload = {
        ticker: ticker,
        quantity: Math.round(shares),
        action: action,  // "Buy" or "Sell"
        price: price,
        timestamp: Date.now()
    };
    
    // Create button element
    var btn = document.createElement("button");
    btn.className = "btn btn-sm btn-outline-primary";
    btn.style.padding = "2px 8px";
    btn.style.fontSize = "0.75rem";
    btn.innerText = "Stage";
    
    btn.onclick = function(e) {
        e.stopPropagation();
        
        // Update the staged-order-store using Dash clientside API
        if (window.dash_clientside && window.dash_clientside.set_props) {
            window.dash_clientside.set_props("staged-order-store", {data: payload});
        }
        
        // Navigate to Trade Execution page
        setTimeout(function() {
            window.location.href = "/trade";
        }, 50);
    };
    
    return btn;
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
