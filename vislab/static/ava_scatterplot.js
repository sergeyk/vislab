function ava_scatterplot(w, h, hex_radius, callback, id, query) {
  /*
  Make the ava plot and return the brush, for reading its extent.
  */
  var margin = {top: 20, right: 20, bottom: 30, left: 40};
  var width = w - margin.left - margin.right;
  var height = h - margin.top - margin.bottom;

  var hexbin = d3.hexbin()
      .size([width, height])
      .radius(hex_radius);

  x = d3.scale.linear()
      .range([0, width]);

  y = d3.scale.linear()
      .range([height, 0]);

  var xAxis = d3.svg.axis()
      .scale(x)
      .orient("bottom");

  var yAxis = d3.svg.axis()
      .scale(y)
      .orient("left");

  var brush = d3.svg.brush()
    .x(x)
    .y(y)
    .on("brushend", brushend);

  // If the brush is empty, select all circles.
  // If brush is active, hide circles that are not in the extent.
  function brushend() {
    if (brush.empty()) {
      svg.selectAll(".hidden").classed("hidden", false);
      callback({
        'rating_mean_min': null, 'rating_mean_max': null,
        'rating_std_min': null, 'rating_std_max': null});
    } else {
      var e = brush.extent();
      svg.selectAll("circle").classed("hidden", function(d) {
        return e[0][0] > d.rating_mean || d.rating_mean > e[1][0] ||
               e[0][1] > d.rating_std  || d.rating_std > e[1][1];
      });
      callback({
        'rating_mean_min': e[0][0], 'rating_mean_max': e[1][0],
        'rating_std_min': e[0][1], 'rating_std_max': e[1][1]
      });
    }
  }

  var svg = d3.select(id).append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
      .attr("id", "plot-canvas");

  x.domain([1, 9]).nice();
  y.domain([0.5, 3]).nice();

  // For hexbin
  svg.append("clipPath")
      .attr("id", "clip")
    .append("rect")
      .attr("class", "mesh")
      .attr("width", width)
      .attr("height", height);

  svg.append("g")
    .attr("id", "clip-holder")
    .attr("clip-path", "url(#clip)");

  svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)
    .append("text")
      .attr("class", "label")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("Rating Standard Deviation");

  svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
    .append("text")
      .attr("class", "label")
      .attr("x", width)
      .attr("y", -6)
      .style("text-anchor", "end")
      .text("Rating Mean");

  // NOTE: all_data is global!
  all_data = [];
  d3.csv("/static/ava_df.csv", function(error, data) {
    // Clean up data
    data.forEach(function(d) {
      d.rating_mean = +d.rating_mean;
      d.rating_std = +d.rating_std;
    });

    // Filter data if query requires it.
    filtered_data = data;
    if (query['style'] != 'all') {
        filtered_data = filtered_data.filter(function(d) {
            return d[query['style']] == 'True';
        });
    }
    if (query['tag'] != 'all') {
        filtered_data = filtered_data.filter(function(d) {
            return ((d['semantic_tag_1_name'] == query['tag']) ||
                    (d['semantic_tag_2_name'] == query['tag']));
        });
    }

    draw_points(filtered_data, hexbin, x, y);

    svg.append("g")
    .attr("class", "brush")
    .call(brush);
  });

  if (!!(query['rating_mean_min'])) {
    brush.extent([
      [parseFloat(query['rating_mean_min']), parseFloat(query['rating_std_min'])],
      [parseFloat(query['rating_mean_max']), parseFloat(query['rating_std_max'])]
    ]);
  }

  return {
    'hexbin': hexbin,
    'x': x,
    'y': y,
    'brush': brush
  };
}

function draw_points(data, hexbin, x, y) {
  // Have to remove all points, because I'm only adding a sample of points.
  d3.select('#plot-canvas').selectAll("circle").remove();

  d3.select('#plot-canvas').selectAll("circle")
    .data(data.slice(0, 2000))
    .enter().append("circle")
      .attr("r", 3.5)
      .attr("cx", function(d) { return x(d.rating_mean); })
      .attr("cy", function(d) { return y(d.rating_std); });

  var hexbin_color = d3.scale.linear()
      .domain([1, data.length / 8])
      .range(["white", "gold"])
      .interpolate(d3.interpolateLab);

  d3.select('#clip-holder')
      .selectAll(".hexagon").remove();

  d3.select('#clip-holder')
      .selectAll(".hexagon")
      .data(hexbin(data.map(function(p) {
          return [x(p.rating_mean), y(p.rating_std)];
        })))
      .enter().append("path")
        .attr("class", "hexagon")
        .attr("d", hexbin.hexagon())
        .attr("transform", function(d) { return "translate(" + (d.x) + "," + (d.y) + ")"; })
        .style("fill", function(d) { return hexbin_color(d.length); });
}
