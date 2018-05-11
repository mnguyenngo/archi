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
        data: JSON.stringify(query),
        success: function (data) {
            console.log(data)
            display_solutions(data);
        }
    });
    // console.log("send" + query)
};

let display_solutions = function(solutions) {
    let s = $("p#solution")
    console.log(s)
    s.text(solutions.user_query)
    let result_table = $("div#result-table")
    result_table.html(solutions.table)
};

$(document).ready(function() {

    $("button#solve").click(function() {
        let query = get_query();
        send_query_json(query);
    })

    $("textarea#query").keypress(function (e) {
      if(e.which == 13 && !e.shiftKey) {
          $(this).closest("form").submit();
          e.preventDefault();
          return false;
      }
    })

})
