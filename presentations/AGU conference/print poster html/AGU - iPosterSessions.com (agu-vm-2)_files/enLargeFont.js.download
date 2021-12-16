var preferredHeight = 768;
var fontsize = 26;

$(window).load(function () {
    enLargeFont();
});
function enLargeFont() {
    if ($('[id$=LoginStatus1]').text() == "" || null || undefined) {
        $('[id^=iTextField]').find('span').each(function (index) {
            if ($(this).css('font-size').replace('px', '') < 5.5) {
                $(this).css('font-size', '0.857em');
            }
        });
    }
}