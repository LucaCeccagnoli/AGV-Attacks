// convert a base64 string to a blob of a given MIME type
// if no extension is passed, jpeg will be used
function base64_to_blob(b64data, MIMEtype = 'image/jpeg'){
    if(MIMEtype == 'image/jpg'){ MIMEtype == 'image/jpeg'}    // src attribute can only read jpeg

    // atob decodes a base64 string into a new string with a character for each byte.
    const byteCharacters = atob(b64data);
    const byteNumbers = new Array(byteCharacters.length);

    // each byte character has a byte value which can be obtain with charCodeAt()
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }

    // Uint8Array converts to a real type array
    const byteArray = new Uint8Array(byteNumbers);

    // the new array can ve converted into a blob
    const blob = new Blob([byteArray], {type: MIMEtype});
    return blob
}

// take an image and a tag to load it in, set show = true to show the image, false to hide it
function set_img_tag(img_tag_id, image, show = true){
    if(image){
        var imageUrl = window.URL || window.webkitURL;
        $('#' + img_tag_id).attr("src",imageUrl.createObjectURL(image));
    }
    if(show){
        $('#' + img_tag_id).show();
    }
    else{
        $('#' + img_tag_id).hide();
    }
}