<html>
<head></head>
<body>
  <button id="predict">Predict</button>
  <p id="output"></p>

  <script>
    const weights = [
      0, -1, -1, -1,
      1, -1,  0,  1,
      1, -1,  0,  0,
      1,  0,  1,  0
    ];
    const bias = 0;

    function activation_function(z) {
      return z >= 0 ? 1 : -1;
    }

    function predict(inputs) {
      let linear_product = 0;
      for (let i = 0; i < inputs.length; i += 1) {
        linear_product += inputs[i] * weights[i];
      }
      linear_product += bias;
      return activation_function(linear_product);
    }

    document.getElementById('predict').addEventListener('click', () => {
      const inputs = [
        /* fill this with 16 values from your 4×4 input, each 0 or 1 */
      ];
      const y_pred = predict(inputs);
      document.getElementById('output').textContent = y_pred === 1 ? 'L' : 'T';
    });
  </script>
</body>
</html>
