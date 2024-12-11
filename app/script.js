const width = window.innerWidth;
const height = 800;
const pixelRatio = window.devicePixelRatio || 1;

const canvas = document.getElementById("chart");
canvas.width = width * pixelRatio;
canvas.height = height * pixelRatio;
canvas.style.width = width + "px";
canvas.style.height = height + "px";
const context = canvas.getContext("2d");
context.scale(pixelRatio, pixelRatio);

const tooltip = d3.select("#tooltip");

function seededRandom(seed) {
	let value = seed;
	return function () {
		value = (value * 9301 + 49297) % 233280;
		return value / 233280;
	};
}

function shuffleArray(array, seed) {
	const random = seededRandom(seed);
	for (let i = array.length - 1; i > 0; i--) {
		const j = Math.floor(random() * (i + 1));
		[array[i], array[j]] = [array[j], array[i]];
	}
	return array;
}

function interpolateCategory10(numColors) {
	const baseColors = [
		"#F1BB7B",
		"#FD6467",
		"#5B1A18",
		"#D67236",
		"#F3DF6C",
		"#CEAB07",
		"#D5D5D3",
		"#24281A",
		"#899DA4",
		"#C93312",
		"#FAEFD1",
		"#DC863B",
		"#798E87",
		"#C27D38",
		"#CCC591",
		"#29211F",
		"#D8B70A",
		"#02401B",
		"#A2A475",
		"#81A88D",
		"#972D15",
		"#9A8822",
		"#F5CDB4",
		"#F8AFA8",
		"#FDDDA0",
		"#74A089",
		"#E6A0C4",
		"#C6CDF7",
		"#D8A499",
		"#7294D4",
		"#85D4E3",
		"#F4B5BD",
		"#9C964A",
		"#CDC08C",
		"#FAD77B",
		"#446455",
		"#FDD262",
		"#D3DDDC",
		"#C7B19C",
		"#3B9AB2",
		"#78B7C5",
		"#EBCC2A",
		"#E1AF00",
		"#F21A00",
		"#DD8D29",
		"#E2D200",
		"#46ACC8",
		"#E58601",
		"#B40F20",
		"#FF0000",
		"#00A08A",
		"#F2AD00",
		"#F98400",
		"#5BBCD6",
		"#E1BD6D",
		"#EABE94",
		"#0B775E",
		"#35274A",
		"#F2300F",
		"#A42820",
		"#5F5647",
		"#9B110E",
		"#3F5151",
		"#4E2A1E",
		"#550307",
		"#0C1707",
		"#ECCBAE",
		"#046C9A",
		"#D69C4E",
		"#ABDDDE",
		"#000000",
	];
	const interpolatedColors = [];
	const step = (baseColors.length - 1) / (numColors - 1);

	for (let i = 0; i < numColors; i++) {
		const index = i * step;
		const lowerIndex = Math.floor(index);
		const upperIndex = Math.ceil(index);
		const t = index - lowerIndex;
		const color = d3.interpolateRgb(
			baseColors[lowerIndex],
			baseColors[upperIndex]
		)(t);
		interpolatedColors.push(color);
	}
	return interpolatedColors;
}

d3.json("vis_data.json").then((data) => {
	const clusterLabels = Array.from(new Set(data.map((d) => d.cluster_label)));

	const interpolatedColors = interpolateCategory10(clusterLabels.length);

	const seed = 123;
	const shuffledColors = shuffleArray(interpolatedColors, seed);

	const color = d3.scaleOrdinal().domain(clusterLabels).range(shuffledColors);

	function getColor(d, selectedCluster) {
		let baseColor = color(d.cluster_label);

		if (selectedCluster && d.cluster_label !== selectedCluster) {
			// Use a faded grey color
			return d3.color("lightgrey").copy({ opacity: 0.3 });
		}

		return baseColor;
	}

	const xExtent = d3.extent(data, (d) => d.x);
	const yExtent = d3.extent(data, (d) => d.y);

	const xScale = d3.scaleLinear().domain(xExtent).nice().range([0, width]);
	const yScale = d3.scaleLinear().domain(yExtent).nice().range([0, height]);

	function adjustLabelPositions(labels) {
		const simulation = d3
			.forceSimulation(labels)
			.force("x", d3.forceX((d) => d.x).strength(1))
			.force("y", d3.forceY((d) => d.y).strength(1))
			.force("collide", d3.forceCollide(20))
			.stop();

		for (let i = 0; i < 100; ++i) simulation.tick();

		return labels;
	}

	let selectedCluster = null;

	function drawPoints(transform) {
		context.save();
		context.clearRect(0, 0, width, height);
		context.translate(transform.x, transform.y);
		context.scale(transform.k, transform.k);

		data.forEach((d) => {
			context.beginPath();
			context.arc(xScale(d.x), yScale(d.y), 3 / transform.k, 0, 2 * Math.PI);

			context.fillStyle = getColor(d, selectedCluster);
			context.fill();
		});

		let labelsData = [];
		const clusters = d3.group(data, (d) => d.cluster_label);
		clusters.forEach((values, key) => {
			if (
				key === "Miscellaneous" ||
				key === "Other" ||
				(selectedCluster && key !== selectedCluster)
			)
				return;
			const xMean = d3.mean(values, (d) => xScale(d.x));
			const yMean = d3.mean(values, (d) => yScale(d.y));
			labelsData.push({ label: key, x: xMean, y: yMean });
		});

		labelsData = adjustLabelPositions(labelsData);

		labelsData.forEach((d) => {
			const fontSize = 14 / transform.k;
			context.font = `bold ${fontSize}px Arial`;
			context.textAlign = "center";

			const textWidth = context.measureText(d.label).width;
			const padding = 4 / transform.k;
			const rectX = d.x - textWidth / 2 - padding;
			const rectY = d.y - fontSize;
			const rectWidth = textWidth + 2 * padding;
			const rectHeight = fontSize + padding;
			const radius = 5 / transform.k;

			context.fillStyle = "rgba(255, 255, 255, 0.8)";
			context.beginPath();
			context.moveTo(rectX + radius, rectY);
			context.lineTo(rectX + rectWidth - radius, rectY);
			context.quadraticCurveTo(
				rectX + rectWidth,
				rectY,
				rectX + rectWidth,
				rectY + radius
			);
			context.lineTo(rectX + rectWidth, rectY + rectHeight - radius);
			context.quadraticCurveTo(
				rectX + rectWidth,
				rectY + rectHeight,
				rectX + rectWidth - radius,
				rectY + rectHeight
			);
			context.lineTo(rectX + radius, rectY + rectHeight);
			context.quadraticCurveTo(
				rectX,
				rectY + rectHeight,
				rectX,
				rectY + rectHeight - radius
			);
			context.lineTo(rectX, rectY + radius);
			context.quadraticCurveTo(rectX, rectY, rectX + radius, rectY);
			context.closePath();
			context.fill();

			context.fillStyle = "black";
			context.fillText(d.label, d.x, d.y);
		});

		context.restore();
	}

	drawPoints({ x: 0, y: 0, k: 1 });

	const zoom = d3
		.zoom()
		.scaleExtent([0.5, 20])
		.filter((event) => event.type !== "dblclick") // Prevent double-click zooming
		.on("zoom", zoomed);

	d3.select(canvas).call(zoom);

	function zoomed(event) {
		drawPoints(event.transform);
	}

	d3.select(canvas)
		.on("click", function (event) {
			const [mx, my] = d3.pointer(event);
			const transform = d3.zoomTransform(canvas);

			const x0 = xScale.invert((mx - transform.x) / transform.k);
			const y0 = yScale.invert((my - transform.y) / transform.k);

			let radius = 5 / transform.k;
			let found = false;

			for (let i = data.length - 1; i >= 0; i--) {
				let d = data[i];
				let dx = xScale(d.x) - xScale(x0);
				let dy = yScale(d.y) - yScale(y0);
				if (dx * dx + dy * dy < radius * radius) {
					selectedCluster =
						selectedCluster === d.cluster_label ? null : d.cluster_label;
					found = true;
					break;
				}
			}

			if (!found) {
				selectedCluster = null;
			}

			drawPoints(d3.zoomTransform(canvas));
		})
		.on("mousemove", function (event) {
			const [mx, my] = d3.pointer(event);
			const transform = d3.zoomTransform(canvas);

			const x0 = xScale.invert((mx - transform.x) / transform.k);
			const y0 = yScale.invert((my - transform.y) / transform.k);

			let radius = 5 / transform.k;
			let found = false;

			for (let i = data.length - 1; i >= 0; i--) {
				let d = data[i];
				let dx = xScale(d.x) - xScale(x0);
				let dy = yScale(d.y) - yScale(y0);
				if (dx * dx + dy * dy < radius * radius) {
					tooltip
						.classed("hidden", false)
						.style("left", event.pageX + 15 + "px")
						.style("top", event.pageY - 28 + "px");
					tooltip.select("#content").text(d.content);
					tooltip.select("#author").text(d.author);
					tooltip.select("#timestamp").text(d.timestamp);
					tooltip.select("#cluster").text(d.cluster_label);
					found = true;
					break;
				}
			}
			if (!found) {
				tooltip.classed("hidden", true);
			}
		})
		.on("mouseout", function () {
			tooltip.classed("hidden", true);
		});
});
