import "./styles.css";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

const renderBarChart = () => {
  const data = [
    { index: "Jill", value: 10 },
    { index: "Jane", value: 20 },
    { index: "Ivan", value: 30 }
  ];
  const container = document.getElementById("barchart-cont");
  tfvis.render.barchart(container, data, {
    xLabel: "Customer",
    yLabel: "Payment",
    height: 450,
    fontSize: 16
  });
};

const renderHistogram = () => {
  const data = Array(20)
    .fill(0)
    .map(x => Math.random() * 50);

  const container = document.getElementById("histogram-cont");
  tfvis.render.histogram(container, data, {
    maxBins: 5,
    height: 450,
    fontSize: 16
  });
};

const renderScatter = () => {
  const apples = Array(14)
    .fill(0)
    .map(y => Math.random() * 100 + Math.random() * 50)
    .map((y, x) => ({ x: x, y: y }));

  const oranges = Array(14)
    .fill(0)
    .map(y => Math.random() * 100 + Math.random() * 150)
    .map((y, x) => ({ x: x, y: y }));

  const series = ["Apples", "Oranges"];

  const data = { values: [apples, oranges], series };

  const container = document.getElementById("scatter-cont");
  tfvis.render.scatterplot(container, data, {
    xLabel: "day",
    yLabel: "sales",
    height: 450,
    zoomToFit: true,
    fontSize: 16
  });
};

const kgToLbs = kg => kg * 2.2;

const trainModel = async () => {
  console.log("Training...");

  const xs = tf.tensor(Array.from({ length: 2000 }, (x, i) => i));
  const ys = tf.tensor(Array.from({ length: 2000 }, (x, i) => kgToLbs(i)));

  const model = tf.sequential();

  model.add(tf.layers.dense({ units: 1, inputShape: 1 }));

  model.compile({ loss: "meanSquaredError", optimizer: "adam" });

  await model.fit(xs, ys, {
    epochs: 100,
    shuffle: true
  });

  const lbs = model
    .predict(tf.tensor([10]))
    .asScalar()
    .dataSync();

  console.log("10 kg to lbs: " + lbs);
};

async function run() {
  // Vector
  const t = tf.tensor([1, 2, 3]);

  console.log(t.rank);

  console.log(t.shape);

  console.log(t);

  t.print();

  // String vector
  const st = tf.tensor(["hello", "world"]);
  console.log(st.shape);
  st.print();

  // Matrix

  const t2d = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
  console.log(t2d.shape);

  t2d.print();

  // Utilities

  tf.ones([3, 3]).print();

  tf.truncatedNormal([3, 3]).print();

  tf.tensor([1, 2, 3, 4, 5, 6])
    .reshape([2, 3])
    .print();

  // Math

  const a = tf.tensor([1, 2, 3]);
  const b = tf.tensor([4, 5, 6]);
  a.add(b).print();

  const d1 = tf.tensor([[1, 2], [1, 2]]);
  const d2 = tf.tensor([[3, 4], [3, 4]]);
  d1.dot(d2).print();

  tf.tensor([[1, 2], [3, 4]])
    .transpose()
    .print();

  // Visualization

  // Model training
  await trainModel();
}

if (document.readyState !== "loading") {
  run();
} else {
  document.addEventListener("DOMContentLoaded", run);
}
