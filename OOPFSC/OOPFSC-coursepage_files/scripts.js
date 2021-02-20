function initializeJS() {

    //tool tips
    jQuery('.tooltips').tooltip();

    //popovers
    jQuery('.popovers').popover();

    //sidebar dropdown menu
    jQuery('#sidebar .sub-menu > a').click(function () {
        // Close previous open submenu
        var last = jQuery('.sub.open', jQuery('#sidebar'));
        jQuery(last).slideUp(200);
        jQuery(last).removeClass("open");
        jQuery('.menu-arrow', jQuery(last).parent()).addClass('fa-angle-right');
        jQuery('.menu-arrow', jQuery(last).parent()).removeClass('fa-angle-down');

        // Toggle current submenu
        var sub = jQuery(this).next();
        if (sub.is(":visible")) {
            jQuery('.menu-arrow', this).addClass('fa-angle-right');
            jQuery('.menu-arrow', this).removeClass('fa-angle-down');
            sub.slideUp(200);
            jQuery(sub).removeClass("open")
        } else {
            jQuery('.menu-arrow', this).addClass('fa-angle-down');
			jQuery('.menu-arrow', this).removeClass('fa-angle-right');
            sub.slideDown(200);
            jQuery(sub).addClass("open")
        }

        // Center menu on screen
        var o = (jQuery(this).offset());
        diff = 200 - o.top;
        if(diff>0)
            jQuery("#sidebar").scrollTo("-="+Math.abs(diff),500);
        else
            jQuery("#sidebar").scrollTo("+="+Math.abs(diff),500);
    });


    // sidebar menu toggle
    jQuery(function() {
        function responsiveView() {
            var wSize = jQuery(window).width();
            if (wSize <= 768 || $( "#hidesidebar" ).length) {
                jQuery('#sidebar > ul').hide();
            }
			else {
                jQuery('#sidebar > ul').show();
			}

            if (jQuery('#sidebar > ul').is(":visible") === true) {
                jQuery('#main-content').removeClass('sidebar-closed');
			}
			else {
                jQuery('#main-content').addClass('sidebar-closed');
            }
        }
        jQuery(window).on('load', responsiveView);
        jQuery(window).on('resize', responsiveView);
    });

    jQuery('.toggle-nav').click(function () {
        if (jQuery('#sidebar > ul').is(":visible") === true) {
            jQuery('#sidebar > ul').hide();
            jQuery("#main-content").addClass("sidebar-closed");
        } else {
            jQuery('#sidebar > ul').show();
            jQuery("#main-content").removeClass("sidebar-closed");
        }
    });

    //bar chart
    if (jQuery(".custom-custom-bar-chart")) {
        jQuery(".bar").each(function () {
            var i = jQuery(this).find(".value").html();
            jQuery(this).find(".value").html("");
            jQuery(this).find(".value").animate({
                height: i
            }, 2000)
        })
    }

}

jQuery(document).ready(function(){
    initializeJS();
});
