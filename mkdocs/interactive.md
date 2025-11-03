# Interactive SCSA Demo

<div id="controls" style="margin: 20px 0;">
    <div style="margin: 10px 0;">
        <label>Signal Type:</label>
        <select id="signal-type" style="width: 100%; padding: 5px;">
            <option value="gaussian">Gaussian</option>
            <option value="sech">Sech</option>
            <option value="double_well">Double Well</option>
            <option value="chirp">Chirp</option>
        </select>
    </div>
    <div style="margin: 10px 0;">
        <label>Method:</label>
        <select id="method" style="width: 100%; padding: 5px;">
            <option value="reconstruct">Reconstruct (manual h)</option>
            <option value="filter_optimal">Filter with Optimal h</option>
            <option value="denoise">Denoise</option>
        </select>
    </div>
    <div style="margin: 10px 0;">
        <label>Gamma (γ): <span id="gamma-value">0.5</span></label>
        <input type="range" id="gamma-slider" min="0.1" max="2.0" step="0.1" value="0.5" style="width: 100%;">
    </div>
    <div style="margin: 10px 0;">
        <label>h parameter: <span id="h-value">1.0</span></label>
        <input type="range" id="h-slider" min="0.001" max="10.0" step="0.005" value="1.0" style="width: 100%;">
    </div>
    <div style="margin: 10px 0;">
        <label>Noise: <span id="noise-value">0.10</span></label>
        <input type="range" id="noise-slider" min="0.05" max="1" step="0.05" value="0.10" style="width: 100%;">
    </div>
    <span id="loading" style="display:none;">⏳ Processing...</span>
</div>

<div id="plot" style="width: 100%; height: 500px;"></div>
<div id="metrics"></div>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
const API_URL = 'https://pyscsa.onrender.com/api/scsa';

async function runSCSA() {
    const gamma = parseFloat(document.getElementById('gamma-slider').value);
    const h = parseFloat(document.getElementById('h-slider').value);
    const noise = parseFloat(document.getElementById('noise-slider').value);
    const signal_type = document.getElementById('signal-type').value;
    const method = document.getElementById('method').value;
    
    document.getElementById('loading').style.display = 'inline';
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ gamma, h, noise, signal_type, method })
        });
        const data = await response.json();

        const traces = [
            { x: data.x, y: data.signal, name: 'Original', line: { color: 'black', width: 2 } },
            { x: data.x, y: data.noisy, name: 'Noisy', line: { color: 'gray', width: 1 }, opacity: 0.5 },
            { x: data.x, y: data.reconstructed, name: 'SCSA', line: { color: 'red', width: 2, dash: 'dash' } }
        ];

        const layout = {
            title: 'SCSA Signal Reconstruction',
            xaxis: { title: 'x' },
            yaxis: { title: 'Signal' },
            height: 500,
            autosize: false
        };

        Plotly.react('plot', traces, layout); 

        document.getElementById('metrics').innerHTML = `
            <div style="padding: 15px; background: #e0f2f1; border-radius: 5px;">
                <strong>Results:</strong><br>
                MSE: ${data.metrics.mse.toFixed(6)}<br>
                PSNR: ${data.metrics.psnr.toFixed(2)} dB<br>
                Eigenvalues: ${data.num_eigenvalues}
            </div>`;
    } catch (error) {
        document.getElementById('metrics').innerHTML = `
            <div style="padding: 15px; background: #ffebee; border-radius: 5px;">
                Error: ${error.message}
            </div>`;
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

document.getElementById('gamma-slider').oninput = (e) => {
    document.getElementById('gamma-value').textContent = parseFloat(e.target.value).toFixed(2);
    runSCSA();
};
document.getElementById('h-slider').oninput = (e) => {
    document.getElementById('h-value').textContent = parseFloat(e.target.value).toFixed(2);
    runSCSA();
};
document.getElementById('noise-slider').oninput = (e) => {
    document.getElementById('noise-value').textContent = parseFloat(e.target.value).toFixed(2);
    runSCSA();
};
document.getElementById('signal-type').onchange = runSCSA;
document.getElementById('method').onchange = runSCSA;

runSCSA();  // Initial load
</script>
