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
            console.log(data)
            display_results(data);
        }
    });
};

let display_results = function(results) {
    let uq_title = $("p#uq-title")
    uq_title.text("Your query")
    let user_query = $("div#show-user-query")
    user_query.html(results.user_query)
    let result_table = $("div#result-table")
    result_table.html(results.table)
};

$(document).ready(function() {
    $("button#predict").click(function() {
        let query = get_query();
        send_query_json(query);
    })
})
