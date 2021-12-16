var contenttitle;
var content;
var cf;
var showingColumns = false;
var nbrOfColumns = 4;
var colpadding = 30;
var nbrOfColsToDisplay;
var colwidth;
var screenWidth;
var pageAnimUnlocked = true;
var currentPageCount = 1;
var numberOfPages = 1;
var numberOfUsedColumns = 0;
var mouseOverIframe = false;

colwidth = ((1920 - (nbrOfColumns * colpadding * 2)) / nbrOfColumns);

function reflowContent() {
    showingColumns = true;

    //$("#wrapper").hide();
    $(".columncontainer").show();
    $(".columncontainer").css("visibility", "visible");
    $("#article").html("");
    $("#article").html(content);
    $(".texttitle").html("");
    $(".floatcontent").html("");
    $(".texttitle").html(contenttitle);
    $(".floatcontent").html(content);

    screenWidth = $(".columncontainer").width();
    colwidth = ((1920 - (nbrOfColumns * colpadding * 2)) / nbrOfColumns);
    if (screenWidth >= 2 * colwidth) {
        $("#viewport").css("left", "0px");
        $(".innercontainer").show();
        $(".innercontainer").css("visibility", "visible");
        $(".floatcontainer").hide();
        $(".floatcontainer").css("visibility", "hidden");
        nbrOfColsToDisplay = Math.floor(screenWidth / colwidth);
        $(".innercontainer").css("width", (colwidth * nbrOfColsToDisplay) + "px");
        cf = new FTColumnflow('myarticle', 'viewport', {
            columnCount: nbrOfColsToDisplay,
            standardiseLineHeight: true,
            pagePadding: colpadding,
            columnGap: 40,
            allowReflow: true,
            noWrapOnTags: ['img', 'h1', 'h2', 'h3', 'h4'],
        });

        cf.flow(content, "");

        $(".cf-column iframe").hover(function () {
            mouseOverIframe = true
        }, function () {
            mouseOverIframe = false;
        });

        $(".nextpage").show();

        numberOfPages = cf.pageCount;
        numberOfUsedColumns = $(".cf-render-area .cf-column").length;

        if (numberOfPages == 1) { //hide arrows and change size of less columns filled than available
            $(".prevpage").hide();
            $(".nextpage").hide();

            if (numberOfUsedColumns < cf.layoutDimensions.columnCount) {
                //lets do some magic
                $(".innercontainer").css("width", numberOfUsedColumns * colwidth + colpadding);
            }
        } else {
            $(".prevpage").hide();
        }
    } else {
        $(".floatcontainer").show();
        $(".innercontainer").hide();
        $(".floatcontainer").css("visibility", "visible");
        $(".innercontainer").css("visibility", "hidden");
    }
}
function unlockAnim() {
    pageAnimUnlocked = true;
}

$(function () {
    $(document).on("mouseup touchend", ".nextpage", function () {
        var pageWidth = $(".innercontainer").width();
        var pagePos = parseInt($("#viewport").css("left"));
        var newpos = pagePos - pageWidth;
        var textblockWidth = $("#myarticle").width();

        if (pageAnimUnlocked && newpos > (0 - textblockWidth)) {
            pageAnimUnlocked = false;
            $("#viewport").animate({ "left": "-=" + pageWidth }, 300, unlockAnim);
        }

        currentPageCount++;

        if (currentPageCount == numberOfPages) $(this).hide();
        if (currentPageCount > 1) $(".prevpage").show();
    });
    $(document).on("mouseup touchend", ".prevpage", function () {
        var pageWidth = $(".innercontainer").width();
        var pagePos = parseInt($("#viewport").css("left"));
        var newpos = pagePos + pageWidth;

        if (pageAnimUnlocked && newpos <= 0) {
            pageAnimUnlocked = false;
            $("#viewport").animate({ "left": "+=" + pageWidth }, 300, unlockAnim);
        }

        currentPageCount--;

        if (currentPageCount == 1) $(this).hide();
        if (currentPageCount < numberOfPages) $(".nextpage").show();
    });
    $(document).on("mouseup touchend", ".close", function (ev) {
        showingColumns = false;
        $(".floatcontainer").hide();
        $(".innercontainer").hide();
        $(".columncontainer").hide();
        $(".floatcontainer").css("visibility", "hidden");
        $(".innercontainer").css("visibility", "hidden");
        $(".columncontainer").css("visibility", "hidden");

        $(".innercontainer, .floatcontent").css("background", "");
    });

    $("html").on("click", ".columncontainer", function (ev) {
        
        if ($(ev.target).parents(".cf-column").length == 0 && !$(ev.target).parent().is('.nextpage,.prevpage')) {
            showingColumns = false;

            $(".floatcontainer").hide();
            $(".innercontainer").hide();
            $(".columncontainer").hide();
            $(".floatcontainer").css("visibility", "hidden");
            $(".innercontainer").css("visibility", "hidden");
            $(".columncontainer").css("visibility", "hidden");

            $(".innercontainer, .floatcontent").css("background", "");
        }
    });

    $(window).on("resize", function (ev) {
        if (showingColumns && !mouseOverIframe) {
            reflowContent();
        }
    });
});
