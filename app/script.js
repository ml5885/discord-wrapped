// Dimensions
const width = window.innerWidth;
const height = 800;

// SVG setup
const svg = d3.select("#chart").attr("width", width).attr("height", height);

// Add a group for zooming
const g = svg.append("g");

// Color scale with a fallback for noise points
const color = d3.scaleOrdinal(d3.schemeCategory10);

function getColor(cluster) {
	return cluster === -1 ? "#cccccc" : color(cluster);
}

// Tooltip
const tooltip = d3.select("#tooltip");

// Load data
d3.json("vis_data.json").then((data) => {
	// Scale
	const xExtent = d3.extent(data, (d) => d.x);
	const yExtent = d3.extent(data, (d) => d.y);

	const xScale = d3.scaleLinear().domain(xExtent).nice().range([0, width]);

	const yScale = d3.scaleLinear().domain(yExtent).nice().range([0, height]);

	// Draw points
	const points = g
		.selectAll("circle")
		.data(data)
		.enter()
		.append("circle")
		.attr("cx", (d) => xScale(d.x))
		.attr("cy", (d) => yScale(d.y))
		.attr("r", 5)
		.attr("fill", (d) => getColor(d.cluster))
		.on("mouseover", (event, d) => {
			tooltip
				.classed("hidden", false)
				.style("left", event.pageX + 15 + "px")
				.style("top", event.pageY - 28 + "px");
			tooltip.select("#content").text(d.content);
			tooltip.select("#author").text(d.author);
			tooltip.select("#timestamp").text(d.timestamp);
			tooltip.select("#cluster").text(d.cluster_label);
		})
		.on("mouseout", () => {
			tooltip.classed("hidden", true);
		});

	// Add cluster labels
	const clusters = d3.group(data, (d) => d.cluster_label);
	clusters.forEach((values, key) => {
		if (key === "No Label") return;
		// Calculate centroid of the cluster
		const xMean = d3.mean(values, (d) => xScale(d.x));
		const yMean = d3.mean(values, (d) => yScale(d.y));
		g.append("text")
			.attr("class", "cluster-label")
			.attr("x", xMean)
			.attr("y", yMean)
			.attr("text-anchor", "middle")
			.attr("font-size", "14px")
			.attr("font-weight", "bold")
			.text(key);
	});

	// Add zoom behavior
	const zoom = d3
		.zoom()
		.scaleExtent([0.5, 20]) // Adjust as needed
		.on("zoom", zoomed);

	svg.call(zoom);

	function zoomed(event) {
		g.attr("transform", event.transform);
		g.selectAll("circle").attr("r", 5 / event.transform.k); // Adjust point size
		g.selectAll(".cluster-label").attr(
			"font-size",
			14 / event.transform.k + "px"
		);
	}
});
