// load an image from a base 64 string
// second argument is optional: id of an img tag to show the image in
function image_select(image, img_tag_id = null){
    console.log("selected an imagenet image ...")
}

// save the uploaded image file
// second argument is optional: id of an img tag to show the image in
function image_upload(image, img_tag_id = null){

}

// save the selected network for the attack
function select_network(network){
    selected_network = network;
}

// predict the class of an image using a specific network
// input: file: input image, string: network name, callback: function (optional)
// output: string: name of the predicted class 
//         or int: server codes 200, 404, 500
function run_prediction(image, network, callback = null){
    // generate json data
    jsonData = {
        "network": network
    }
    var formData = new FormData();
    formData.append("image", image);
    formData.append("jsonData", JSON.stringify(jsonData));

    // ajax request, contentType and processData have to be false for file uploading to work
    $.ajax({
        method: "POST",
        url: $SCRIPT_ROOT + "/predict/",
        contentType: false,
        processData: false,
        data: formData,
        dataType: "json",
        xhr:function(){
            var xhr = new XMLHttpRequest();
            xhr.onreadystatechange = function(){
                if(xhr.readyState == XMLHttpRequest.DONE && xhr.status == 200){
                    // get json response data
                    response = xhr.responseText;
                    response = JSON.parse(response);
                    output = {
                        'class_name': response['class_name'],
                        'class_code': response['class_code']
                    }
                    if(callback){
                        return callback(output)
                    }
                    else return xhr.status
                }
                // error handling
                else if(xhr.readyState == XMLHttpRequest.DONE){
                    if (callback){
                        return callback(xhr.status)
                    }
                    else
                        return xhr.status
                }
            }
            return xhr;
        }
    });
}

// run an adversarial attack on the input image using one of the available models
// input: file: input image, string: attack name, string: network name, callback: function (optional) 
// output: string: modified image as base64 string, string: the predicted class on the image
//         or int: server codes 200, 404, 500
function run_attack(image, model, network,callback = null){
    // generate json data
    var jsonData = {
        'model': model,
        'network': network
    }

    // validate input
    if(!image){
        $("#s-msg").html("Please select an image to upload.");
        $("#s-msg").show()
        return
    }
    if(jsonData['model'] == ''){
        $("#s-msg").html("Please select a model for the attack.");
        $("#s-msg").show()
        return
    }
    if(jsonData['network'] == ''){
        $("#s-msg").html("Please select a network to attack.");
        $("#s-msg").show()
        return
    }

    // add data to FormData to send it in ajax
    var formData = new FormData();
    formData.append("image", image);
    formData.append("jsonData", JSON.stringify(jsonData));

    // ajax request
    $.ajax({
        method: "POST",
        url: $SCRIPT_ROOT + "/runattack/",
        contentType: false,
        processData: false,
        data: formData,
        dataType: "json",
        xhr:function(){
            var xhr = new XMLHttpRequest();
            xhr.onreadystatechange = function(){
                if(xhr.readyState == XMLHttpRequest.DONE && xhr.status == 200){
                    //get json response data
                    response = xhr.responseText;
                    response = JSON.parse(response);
                    output = {
                        'encoding': response['encoding'],
                        'base64' : response['img_base64'],
                        'class_name': response['mod_class_name'],
                        'class_code': response['mod_class_code']
                    };

                    if(callback){
                        return callback(output);
                    }
                    else
                        return 202;
                }
                // error handling
                else if(xhr.readyState == XMLHttpRequest.DONE){
                    if (callback){
                        return callback(xhr.status);
                    }
                    else
                        return xhr.status;
                }
            }
            return xhr;
        }
    });
}


