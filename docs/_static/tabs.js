$(document).ready(function () {
	var selected

	function label(lbl) {
        return $.trim(lbl.replace(/[ .]/g, '-').replace('+-', '').toLowerCase());
    }

	function showTOC(framework) {
        if (framework == 'mxnet') {
            $('.localtoc ul').children().eq(1).show()
            $('.localtoc ul').children().eq(2).hide()
        } 
        else if (framework == 'pytorch') {
            $('.localtoc ul').children().eq(1).hide()
            $('.localtoc ul').children().eq(2).show()
        }
	}

	function setSelected() {
		selected = localStorage.getItem("selectedFramework")
		if (selected === null) {
			selected = 'mxnet';
			localStorage.setItem('selectedFramework', selected)
		}
	    $('.framework-group .framework').each(function(){
	        if (label($(this).text()).indexOf(selected) != -1) {
	            $(this).addClass('selected');
	        }
	    });
	}

	setSelected()

    // find the user os, and set the according option to active
    function setSelectedButton() {
        $('.selected').addClass('selected');
    }

	setSelectedButton();

    // apply theme
    function setStyle() {
        $('.framework').each(function(){
            $(this).addClass('mdl-button mdl-js-button mdl-js-ripple-effect mdl-button--raised');
        });
        $('.selected').each(function(){
            $(this).addClass('mdl-button--colored');
        });
    }
    setStyle();

    function showContent() {
        $('.framework-group .framework').each(function(){
            $('.'+label($(this).text())).hide();
        });
        $('.framework-group .selected').each(function(){
            $('.'+label($(this).text())).show();
            if ($(this).text() == 'MXNet') {
            	showTOC('mxnet');
            } else {
            	showTOC('pytorch');
            }
        });
    }

    showContent();

    function setOptions() {
        var el = $(this);
        el.siblings().removeClass('selected');
        el.siblings().removeClass('mdl-button--colored');
        el.addClass('selected');
        el.addClass('mdl-button--colored');
        selected = label($(this).text());
        localStorage.setItem('selectedFramework', selected);
        showContent();
    }

    $('.framework-group').on('click', '.framework', setOptions);
});
