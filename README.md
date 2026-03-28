<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Perceptron L/T Classifier</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, 40px);
      grid-template-rows: repeat(4, 40px);
      gap: 4px;
      margin-bottom: 10px;
    }
    .cell {
      width: 40px;
      height: 40px;
      border: 1px solid #333;
      background-color: white;
      cursor: pointer;
    }
    .cell.on {
      background-color: black;
    }
    .controls { margin-top: 10px; }
    button { margin-right: 8px; }
    #samplesList {
      margin-top: 10px;
      max-height: 200px;
      overflow-y: auto;
      font-size: 0.9em;
      border: 1px solid #ccc;
      padding: 5px;
    }
  </style>
</head>
<body>
  <h1>Perceptron L/T Classifier (4×4)</h1>

  <p>Click cells to toggle pixels (black = 1, white = 0). Choose label, add to dataset, then train.</p>

  <!-- 4×4 grid -->
  <div id="grid" class="grid"></div>

  <!-- Label selection -->
  <div>
    <label><input type="radio" name="label" value="1" checked> L</label>
    <label><input type="radio" name="label" value="-1"> T</label>
  </div>

  <div class="controls">
    <button id="clearGrid">Clear grid</button>
    <button id="addSample">Add sample</button>
    <button id="train">Train perceptron</button>
    <button id="predict">Predict current drawing</button>
  </div>

  <p><strong>Samples:</strong> <span id="sampleCount">0</span></p>
  <p><strong>Training status:</strong> <span id="status">Not trained yet.</span></p>
  <p><strong>Prediction:</strong> <span id="prediction">N/A</span></p>

  <div id="samplesList"></div>

  <script>
    // ----- Perceptron class (JS version of your Python code) -----
    class Perceptron {
      constructor(size) {
        this.weights = new Array(size).fill(0);
        this.bias = 0;
      }

      predictRaw(inputs) {
        let s = this.bias;
        for (let i = 0; i < this.weights.length; i++) {
          s += this.weights[i] * inputs[i];
        }
        return s;
      }

      predict(inputs) {
        return this.predictRaw(inputs) >= 0 ? 1 : -1;
      }

      train(samples, maxEpochs = 1000) {
        let epoch = 0;
        while (epoch < maxEpochs) {
          let errors = 0;
          // shuffle samples
          for (let i = samples.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [samples[i], samples[j]] = [samples[j], samples[i]];
          }
          for (const sample of samples) {
            const activation = this.predictRaw(sample.pixels);
            if (sample.label * activation <= 0) {
              errors += 1;
              for (let i = 0; i < this.weights.length; i++) {
                this.weights[i] += sample.label * sample.pixels[i];
              }
              this.bias += sample.label;
            }
          }
          if (errors === 0) break;
          epoch += 1;
        }
        return epoch + 1;
      }
    }

    // ----- UI + data handling -----
    const GRID_SIZE = 4;
    const gridEl = document.getElementById('grid');
    const sampleCountEl = document.getElementById('sampleCount');
    const statusEl = document.getElementById('status');
    const predictionEl = document.getElementById('prediction');
    const samplesListEl = document.getElementById('samplesList');

    const cells = [];
    const samples = [];
    const perceptron = new Perceptron(GRID_SIZE * GRID_SIZE);

    // Build 4×4 grid
    for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.index = i;
      cell.addEventListener('click', () => {
        cell.classList.toggle('on');
      });
      gridEl.appendChild(cell);
      cells.push(cell);
    }

    function getCurrentPixels() {
      return cells.map(c => c.classList.contains('on') ? 1 : 0);
    }

    function clearGrid() {
      cells.forEach(c => c.classList.remove('on'));
    }

    function getSelectedLabel() {
      const radios = document.querySelectorAll('input[name="label"]');
      for (const r of radios) {
        if (r.checked) return parseInt(r.value, 10);
      }
      return 1;
    }

    function labelText(label) {
      return label === 1 ? 'L' : 'T';
    }

    function renderSamplesList() {
      samplesListEl.innerHTML = '';
      samples.forEach((s, idx) => {
        const div = document.createElement('div');
        div.textContent = `#${idx + 1}: label=${labelText(s.label)}, pixels=${JSON.stringify(s.pixels)}`;
        samplesListEl.appendChild(div);
      });
    }

    // Buttons
    document.getElementById('clearGrid').addEventListener('click', () => {
      clearGrid();
    });

    document.getElementById('addSample').addEventListener('click', () => {
      const pixels = getCurrentPixels();
      const label = getSelectedLabel();
      samples.push({ pixels, label });
      sampleCountEl.textContent = samples.length;
      renderSamplesList();
      statusEl.textContent = `Sample added. Total: ${samples.length}.`;
    });

    document.getElementById('train').addEventListener('click', () => {
      if (samples.length < 100) {
        statusEl.textContent = `Need at least 100 samples. Currently: ${samples.length}.`;
        return;
      }
      const epochs = perceptron.train(samples, 2000);

      // Evaluate accuracy on training set
      let correct = 0;
      for (const s of samples) {
        if (perceptron.predict(s.pixels) === s.label) correct++;
      }
      const acc = Math.round((correct / samples.length) * 100);
      statusEl.textContent = `Trained on ${samples.length} samples. Accuracy: ${acc}% after ${epochs} epochs.`;
    });

    document.getElementById('predict').addEventListener('click', () => {
      const pixels = getCurrentPixels();
      const yPred = perceptron.predict(pixels);
      predictionEl.textContent = labelText(yPred);
    });
  </script>
</body>
</html>
