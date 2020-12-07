// hide elements by passing their ids as a string array
function hide_elements_by_id(ids){
    ids.forEach(id => {
        $("#"+ id).hide();
    });
}

// show elements by passing their ids as a string array
function show_elements_by_id(ids){
    ids.forEach(id => {
        $("#"+ id).show();
    });
}

function disable_input(ids){
    ids.forEach(id => {
        $("#"+ id).prop("disabled", true);
    });
}

function enable_input(ids){
    ids.forEach(id => {
        $("#"+ id).prop("disabled", false);
    });
}

// prints a message on a given html element according to an http response code
function print_server_response(code, element){
    switch(code){
        case 200:
            element.innerHtml = "OK";
            break;
        case 404:
            element.innerHtml = "Resource not found";
            break;
        case 500:
            element.innerHtml = "Internal server error";
            break;
    }
}