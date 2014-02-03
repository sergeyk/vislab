var ava_explorer = function() {
    var get_full_query = function(query) {
        var full_query = {
            'page': page
        };
        $('.query_arg').each(function(i, x) {
            full_query[$(x).attr('id')] = $(x).val();
        });
        $.extend(full_query, query);
        console.log(full_query);
        return full_query;
    };

    // When selected things change, this function redirects to new url.
    var update_url_with_query = function(query) {
        full_query = get_full_query(query);
        search_url = './' + page_mode + '?' + $.param(full_query);
        window.location = search_url;
    };

    var update_results = function() {
        full_query = get_full_query({});

        // Generate the plot
        var plot_vars = null;
        if (page_mode == 'data' && dataset_name == 'ava') {
            // Calling ava_scatterplot loads ava_df csv data into all_data global var.
            plot_vars = ava_scatterplot(
                600, 300, 12, update_url_with_query,
                '#data-scatterplot', full_query);
            // TODO: pass in the range stuff here to set the brush
        }
        if (page_mode == 'results') {
            ava_results_barplot(
                1360, 500, update_url_with_query,
                '#results-barplot', './results_table',
                {'setting': full_query['setting'], 'task': full_query['task']});
        }

        $('#results').html("...fetching results...");
        var results_url = '';
        if (page_mode == 'results') {
            results_url += './results_images_json';
        } else if (page_mode == 'data') {
            results_url += './images_json';
        } else {
            console.log("ERROR");
        }

        $.getJSON(results_url,
            data=full_query,
            success=display_results).error(function() {
                $('#num-results').text('0');
                $('#results').html('no results for given query!');
                $('.results-nav').show();
            });
        return false;
    };

    var display_results = function(json_data) {
        var with_ratings_hist = false;

        var items = $.map(json_data['results'], function(val, i) {
            var result = $('<div/>')
                .addClass('result');

            image_page_url = './image?image_id=' + val['image_id'];
            var link = $('<a/>')
                .attr('href', image_page_url);

            var img = $('<img/>')
                .attr('src', val['image_url'])
                .attr('width', '160px');

            var caption = $('<div/>');
            if (page_mode == 'data' && dataset_name == 'ava') {
                caption.append(sprintf('%.2f/%.2f',
                    val['rating_mean'], val['rating_std'])
                );

                with_ratings_hist = true;
                var sparkline = sprintf(' | <span class="inlinebar">%s</span>',
                    val['ratings']);
                caption.append(sparkline);
            }

            if (page_mode == 'results') {
                caption.append(sprintf(' | %d/%.2f %s',
                    val['label'], val['selected_pred'], val['split'])
                );
                // result.addClass(
                //     (val['label'] == val['selected_pred_binarized']) ? 'green' : 'red'
                // );
            }

            result.append(link.append(img)).append(caption);
            return result;
        });
        $('#num-results').text(json_data['num_results']);
        $('#page').text(page);
        $('#results').html(items);

        $('.results-nav').show();

        if (with_ratings_hist) {
            $('.inlinebar').sparkline(
                'html', {
                    type: 'bar', barColor: 'gray', chartRangeMin: 0}
                );
            $.sparkline_display_visible();
        }
    };

    $('#prev-page').click(function() {
        page = Math.max(1, page - 1);
        update_url_with_query();
    });

    $('#next-page').click(function() {
        page += 1;
        update_url_with_query();
    });

    $('select').change(function() {
        page = 1;
        update_url_with_query();
    });

    // Load images with current settings.
    update_results();
};
