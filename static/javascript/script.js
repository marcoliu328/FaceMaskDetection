/*function getImagePath(){
    if (s == 1){
        return videoRoute;
    }
    else {
        return "static/images/placeholder.jpg";
    }
}

window.onload = function(){
    document.getElementById('img').setAttribute("src", getImagePath());
}*/

$(document).ready(function() {
    $('form').on('submit', function(event) {

        var btnClicked = event.originalEvent.submitter.id;
        newRequest = "";
        if (btnClicked == "startButton") {
            newRequest = "start";
        }
        else {
            newRequest = "stop";
        }
        $.ajax({
            data: {'request' : newRequest},
            type: 'POST',
            url: '/requests',
            dataType:'json'
        })
        .done(function(data) {
            console.log(data.switch);
            d = new Date();
            if (data.switch == 1) {
                $('img').attr("src", videoRoute+"?"+d.getTime());
                $('img').css({
                    'box-shadow' : '0px 4px 10px #000000'
                })
            }
            else {
                $('img').attr("src", "static/images/facemaskicon.png");
                $('img').css({
                    'box-shadow' : '0px 0px 0px #000000'
                })
            }
        });
        event.preventDefault();

    });
});