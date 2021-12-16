$(document).ready(function () {
    init();
});
function init() {
    if ($('#hdnfldUserId').val() != '' && $('#hdnfldSessionId').val() != '') {
        QuickDemo();
        $("#popup-chat").PopupWindow("close");
    } else {
        $('#popup-chat').hide();
    }
    if ($('#ctl00_LoginView_LoginStatus1').text() == "" || null || undefined) {
        $('#ctl00_popup_chat_Text').hide();
    } else {
        $('#ctl00_popup_chat_Text').show();
    }
    getChatTime();
    getSessionTime();
}

function getChatTime() {
    //if ($('#ctl00_hdnChatDateTime').val() != null && $('#ctl00_hdnChatDateTime').val() != ''
    //    && $('#ctl00_hdnChatDateTime').val() != undefined && $('#ctl00_hdnChatDateTime').val().indexOf('/') > -1) {
    //    var hdnChatDateTime = $('#ctl00_hdnChatDateTime').val();
    //    $('#txtChatFrom').val($('#ctl00_hdnChatDateTime').val().split(',')[2].split('-')[0].trim());
    //    $('#txtChatTo').val($('#ctl00_hdnChatDateTime').val().split(',')[2].split('-')[1].trim());
    //    var chatDateFormat = $('#ctl00_hdnChatDateTime').val().split(',')[1].split('/')[2].trim() + '-' + $('#ctl00_hdnChatDateTime').val().split(',')[1].split('/')[0].trim() + '-' + $('#ctl00_hdnChatDateTime').val().split(',')[1].split('/')[1].trim()
    //    $('#txtChatStartEndTime').val(chatDateFormat);
    //}
}

function getSessionTime() {
    //var hdnSessionDateTime = $('#hdnSessionChatDateTime').val();
    //if (hdnSessionDateTime != '') {
    //    $('#txtSessionFrom').val(hdnSessionDateTime.split(',')[2].split('-')[0].trim());
    //    $('#txtSessionTo').val(hdnSessionDateTime.split(',')[2].split('-')[1].trim());
    //    var sessionDateFormat = hdnSessionDateTime.split(',')[1].split('/')[2].trim() + '-' + hdnSessionDateTime.split(',')[1].split('/')[0].trim() + '-' + $('#hdnSessionChatDateTime').val().split(',')[1].split('/')[1].trim()
    //    $('#txtSessionStartEndTime').val(sessionDateFormat);
    //}
}

function QuickDemo() {
    $("#popup-chat").PopupWindow({
        title: "Chat Window",
        modal: true,
        autoOpen: false,
        animationTime: 300,
        customClass: "chat-box-z-index",
        buttons: {
            close: true,
            maximize: false,
            collapse: true,
            minimize: true
        },
        buttonsPosition: "right",
        buttonsTexts: {
            close: "Close",
            maximize: "Maximize",
            unmaximize: "Restore",
            minimize: "Minimize",
            unminimize: "Show",
            collapse: "Collapse",
            uncollapse: "Expand"
        },
        draggable: true,
        dragOpacity: 0.6,
        resizable: true,
        resizeOpacity: 0.6,
        statusBar: true,
        top: "auto",
        left: "auto",
        bottom: 300,
        height: 630,
        width: 700,
        maxHeight: 800,
        maxWidth: 900,
        minHeight: 700,
        minWidth: 700,
        collapsedWidth: 800,
        keepInViewport: true,
        mouseMoveEvents: true
    });
}
$(function () { if (typeof loadbalance_debug_mode != 'undefined' && loadbalance_debug_mode) $("body").append('<div style="position:fixed;bottom:0;left;0;background:white;color:black;font-size:10px;z-index:1000000;height:12px;">' + loadbalance_name + '</div>') });