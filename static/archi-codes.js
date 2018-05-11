let get_query = function() {
    let q = $("textarea#query").val()
    // console.log(q)
    return q
};

let send_query_json = function(query) {
    $.ajax({
        url: '/solve',
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        success: function (data) {
            console.log(data)
            display_solutions(data);
        },
        data: JSON.stringify(query)
        // data: query
    });
    // console.log("send" + query)
};

let display_solutions = function(solutions) {
    let s = $("span#solution")
    console.log(s)
    s.text(solutions.user_query)
    // console.log("display works? ")
};

$(document).ready(function() {

    $("button#solve").click(function() {
        let query = get_query();
        send_query_json(query);
    })

})
