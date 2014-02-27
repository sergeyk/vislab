function ava_results_barplot(w, h, callback, selector, csv_url, selected) {

  var margin = {top: 20, right: 20, bottom: 120, left: 40},
      width = w - margin.left - margin.right,
      height = h - margin.top - margin.bottom;

  var x0 = d3.scale.ordinal()
      .rangeRoundBands([0, width], 0.1);

  var x1 = d3.scale.ordinal();

  var y = d3.scale.linear()
      .range([height, 0]);

  var color = d3.scale.category20();

  var xAxis = d3.svg.axis()
      .scale(x0)
      .orient("bottom");

  var yAxis = d3.svg.axis()
      .scale(y)
      .orient("left")
      .tickFormat(d3.format(".2s"));

  var svg = d3.select(selector).append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  d3.csv(csv_url, function(error, data) {
    var settings = d3.keys(data[0]).filter(function(key) { return key !== "full_task"; });
    var tasks = data.map(function(d) { return d['full_task']; });

    data.forEach(function(d) {
      d.scores = settings.map(function(setting) {
        return {setting: setting, value: +d[setting], task: d['full_task']};
      });
    });

    x0.domain(tasks);
    x1.domain(settings).rangeRoundBands([0, x0.rangeBand()]);
    y.domain([0.4, 1]);

    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis)
        .selectAll("text")
          .style("text-anchor", "end")
          .attr("transform", function(d) {
            return "rotate(-45)";
          });

    svg.append("g")
        .attr("class", "y axis")
        .call(yAxis)
      .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dy", ".71em")
        .style("text-anchor", "end")
        .text("Score (accuracy or R^2)");

    var tooltip = d3.select("body").append("div")
      .attr("class", "tooltip")
      .style("opacity", 0);

    var task = svg.selectAll(".task")
        .data(data)
      .enter().append("g")
        .attr("class", "g")
        .attr("transform", function(d) { return "translate(" + x0(d.full_task) + ",0)"; });

    task.selectAll("rect")
        .data(function(d) { return d.scores; })
      .enter().append("rect")
        .attr("width", x1.rangeBand())
        .attr("x", function(d) { return x1(d.setting); })
        .attr("y", function(d) { return y(d.value); })
        .attr("height", function(d) { return height - y(d.value); })
        .style("fill", function(d) { return color(d.setting); })
        .classed('bar', true)
        .classed('selected', function(d) {
            return (d.setting == selected['setting']) && (d.task == selected['task']);
          })
        .on("mouseover", function(d) {
          d3.select(d3.event.target).classed("highlight", true);
          tooltip.style("opacity", 0.75);
          tooltip.html(d['task'] + '<br />' + d['setting'] + '<br />' + d3.format(".3f")(d['value']))
            .style("left", (d3.event.pageX + 12) + "px")
            .style("top", d3.event.pageY + "px");
        })
        .on("mouseout", function() {
          tooltip.style("opacity", 0);
          d3.select(d3.event.target).classed("highlight", false);
        })
        .on("click", function(d) {
          d3.select(d3.event.target).classed("highlight", false);
          callback({'task': d['task'], 'setting': d['setting']});
        });

    var legend = svg.selectAll(".legend")
        .data(settings.slice())
      .enter().append("g")
        .attr("class", "legend")
        .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

    legend.append("rect")
        .attr("x", width)
        .attr("width", 18)
        .attr("height", 18)
        .style("fill", color);

    legend.append("text")
        .attr("x", width - 8)
        .attr("y", 9)
        .attr("dy", ".35em")
        .style("text-anchor", "end")
        .text(function(d) { return d; });

  });

}
