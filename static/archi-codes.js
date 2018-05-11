let get_query = function() {
    let q = $("textarea#query").val()
    return q
};

let send_query_json = function(query) {
    $.ajax({
        url: '/solve',
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        success: function (data) {
            display_solutions(data);
        },
        data: JSON.stringify(query)
    });
};

let display_solutions = function(solutions) {
    $("span#solution").html(solution)
};

$(document).ready(function() {

    $("button#solve").click(function() {
        let query = get_query();
        send_query_json(query);
    })

})
