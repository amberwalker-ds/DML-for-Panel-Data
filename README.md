<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Causal Impact of Peace Agreements on Conflict Intensity</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f5f5f5; color: #333; }
    h1, h2, h3 { color: #3b5998; }
    .container { width: 80%; margin: 0 auto; padding: 20px; background-color: white; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .highlight { color: #d9534f; font-weight: bold; }
    .tech-icons img { width: 40px; height: 40px; margin-right: 10px; }
    .code-snippet { background-color: #f0f0f0; padding: 10px; border-left: 4px solid #3b5998; font-family: monospace; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Causal Impact of Peace Agreements on Conflict Intensity</h1>
    <p>
      This project leverages <span class="highlight">Double Machine Learning (DML)</span> to estimate the causal effects of peace agreements on violence reduction, specifically analyzing the intensity of conflicts before and after peace agreements are implemented.
    </p>
    
    <h2>📈 Purpose & Features</h2>
    <ul>
      <li><strong>Assess the Causal Impact:</strong> Using DML to better isolate the true effect of peace agreements from other influencing factors.</li>
      <li><strong>Panel Data Analysis:</strong> Integrates lagged data and fixed effects for more accurate, time-sensitive insights.</li>
      <li><strong>Custom Model Design:</strong> Combines Random Forest models with cross-fitting and machine learning for enhanced causal inference.</li>
    </ul>

    <h2>⚙️ Technology Stack</h2>
    <p>We chose these technologies to handle complex data structures, ensure high model accuracy, and provide interpretable results:</p>
    <div class="tech-icons">
      <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/GLM.png" alt="General Linear Model">
      <img src="https://upload.wikimedia.org/wikipedia/commons/1/1b/Scikit_learn_logo_small.svg" alt="Scikit-learn">
      <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/NumPy_logo_2020.svg" alt="NumPy">
      <img src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg" alt="Pandas">
    </div>

    <h2>🛠️ Challenges & Future Plans</h2>
    <ul>
      <li><strong>Challenge:</strong> Ensuring accurate causal estimates while handling high-dimensional, lagged data.</li>
      <li><strong>Future Plans:</strong> Integrate interactive visualizations, explore additional ML models for robustness, and optimize for faster processing times.</li>
    </ul>

    <h2>🚀 Quick Start</h2>
    <div class="code-snippet">
      <p># Clone this repository</p>
      <p>git clone https://github.com/yourusername/peace-agreement-causal-impact.git</p>

      <p># Install dependencies</p>
      <p>pip install -r requirements.txt</p>

      <p># Run the main analysis</p>
      <p>python main.py</p>
    </div>

    <h2>📜 License</h2>
    <p>This project is licensed under the MIT License.</p>

    <h2>🤝 Contributing</h2>
    <p>Contributions are welcome! Feel free to fork the repository and submit pull requests.</p>
  </div>
</body>
</html>

