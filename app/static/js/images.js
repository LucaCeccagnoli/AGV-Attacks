
// if an src is assigned, also handles UI operations
class CustomImage{
    // pass a tag id to save the associated element
    constructor(element_id = null){
        this.blob = null;
        this.base64 = null;
        this.class_code = null;         // last predicted class code
        this.class_name = null;         // last predicted class name
        this.element = null;            // associated image tag
        if(element_id){this.element = document.getElementById(element_id) }
    }

    // load ad image from a base64 string
    // converts the base64 string to a blob, saves both data and returns the blob
    // if the image has an associated tag, generate an URL and assign it to it
    fromBase64(b64data){
        //reset data
        this.reset();

        // convert base64 to blob
        const byteCharacters = atob(b64data);
        const byteNumbers = new Array(byteCharacters.length);    
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], {type: 'image.jpeg'});
        this.base64 = b64data
        this.blob = blob;
        return blob
    }

    // load ad image from a blob, save it and return it
    // TODO: base64 conversion?
    fromBlob(blob){
        //reset data
        this.reset();

        this.blob = blob;
        console.log(this.element);
        console.log(this.blob);
        return blob
    }

    // reset fields to null
    reset(){
        this.blob = null;
        this.base64 = null;
        this.class_code = null;
        this.class_prediction = null;
        this.class_name = null;         
    }

    // show the image in its currently assigned tag with a new url
    visible(flag){
        if(this.element){
            if(flag){
                const imageUrl = window.URL || window.webkitURL;
                this.element.src = imageUrl.createObjectURL(this.blob);
                this.element.style.display = "block";
            }
            else{
                this.element.src = "";
                this.element.style.display = "none";
            }
        }
        else{
            return null;
        }
    }
}


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