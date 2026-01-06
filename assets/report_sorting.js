window.dash_clientside = Object.assign({}, window.dash_clientside, {
    report_sorting: {
        enable_sortable: function(dummy_trigger) {
            var el = document.getElementById("report-sections-container");
            if (el) {
                if (el.dataset.sortableInitialized) {
                    return window.dash_clientside.no_update;
                }
                
                new Sortable(el, {
                    animation: 150,
                    handle: ".drag-handle",
                    ghostClass: "sortable-ghost",
                    onEnd: function (evt) {
                        var newOrder = [];
                        var children = el.children;
                        for (var i = 0; i < children.length; i++) {
                            newOrder.push(children[i].getAttribute('data-value'));
                        }
                        // Update the persistent store
                        dash_clientside.set_props("report-order-store", {data: newOrder});
                    }
                });
                
                el.dataset.sortableInitialized = "true";
            }
            return window.dash_clientside.no_update;
        }
    }
});
