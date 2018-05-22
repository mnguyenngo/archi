let get_query = function() {
    let q = $("textarea#user-query").val()
    return q
};

let send_query_json = function(query) {
    $.ajax({
        url: '/predict',
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        data: JSON.stringify(query),
        success: function (data) {
            display_results(data);
        }
    });
};

let display_results = function(results) {
    let user_query = $("div#show-user-query")
    user_query.html(results.user_query)
    let comp_results = $("div#component-results")
    comp_results.html(results.components)
    let prov_results = $("div#provision-results")
    prov_results.html(results.provisions)
};

$(document).ready(function() {
    $("button#predict").click(function() {
        let query = get_query();
        send_query_json(query);
    })
})
