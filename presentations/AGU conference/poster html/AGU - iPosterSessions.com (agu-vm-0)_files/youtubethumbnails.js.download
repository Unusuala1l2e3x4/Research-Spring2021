$(document).ready(function () {
    var queries = {};
    $.each(document.location.search.substr(1).split('&'), function (c, q) {
        var i = q.split('=');
        queries[i[0].toString()] = i[1].toString();
    });
    if (queries.p === '1' && queries.thumbnail === 'true' || queries.createnails === 'true') {
        $('#countdown').remove();
        $('iframe').each(function (i, ele) {
            var iframe = $(ele);
            var parent = iframe.parent('div');
            var iframeSrc = iframe.attr('src');
            if (iframeSrc != undefined) {
                if (iframeSrc.match(/youtube\.com.*(\?v=|\/embed\/)(.{11})/) != null) {
                    var youtubeVideoId = iframeSrc.match(/youtube\.com.*(\?v=|\/embed\/)(.{11})/).pop();
                    if (youtubeVideoId.length == 11) {
                        var videoThumbnail = $('<img src = "https://img.youtube.com/vi/' + youtubeVideoId + '/0.jpg">');
                        $(this).hide();
                        parent.append(videoThumbnail);
                    }
                }
            } if (iframeSrc != undefined) {
                iframeSrc.match(/(http:|https:|)\/\/(player.|www.)?(vimeo\.com|youtu(be\.com|.be|be\.googleapis\.com))\/(video\/|embed\/|watch\?v=|v\/)?([A-Za-z0-9._ %-]*)(\&\S+)?/);
                if (RegExp.$3.indexOf('vimeo') > -1) {
                    var vimeoId = iframeSrc.split('/')[4];
                    $(this).hide();
                    $.ajax({
                        type: 'GET',
                        url: 'http://vimeo.com/api/v2/video/' + vimeoId + '.json',
                        jsonp: 'callback',
                        dataType: 'jsonp',
                        success: function (data) {
                            $('<a/>', '').append($('<img/>', { src: data[0].thumbnail_large })).appendTo(parent);
                        }
                    });
                }
            }
        });
    }
});