// Chart instances
let lossChart, accSparsityChart, layerSparsityChart;
let globalMetrics = null;
let currentLambda = null;
let autoRefresh = true;
let refreshInterval = null;

// Chart.js global defaults
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 23, 42, 0.9)';
Chart.defaults.plugins.tooltip.titleColor = '#f8fafc';
Chart.defaults.plugins.tooltip.bodyColor = '#f8fafc';
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 8;
Chart.defaults.plugins.tooltip.displayColors = true;

const colors = {
    primary: '#3b82f6',
    secondary: '#8b5cf6',
    tertiary: '#10b981',
    quaternary: '#f59e0b',
    grid: 'rgba(255, 255, 255, 0.05)'
};

function fetchMetrics() {
    const script = document.createElement('script');
    script.src = `../results/metrics.js?t=${new Date().getTime()}`;
    script.onload = () => {
        if (window.globalMetricsData) {
            globalMetrics = window.globalMetricsData;
            updateLambdaDropdown();
            if (currentLambda && globalMetrics[currentLambda]) {
                updateCharts();
            }
        }
    };
    script.onerror = () => {
        console.log("Waiting for metrics.js to be generated...");
    };
    document.head.appendChild(script);
}

function updateLambdaDropdown() {
    const select = document.getElementById('lambdaSelect');
    const lambdas = Object.keys(globalMetrics);
    
    if (select.options.length === 1 && select.options[0].disabled) {
        select.innerHTML = '';
        lambdas.forEach(lam => {
            const option = document.createElement('option');
            option.value = lam;
            option.textContent = lam;
            select.appendChild(option);
        });
        currentLambda = lambdas[0];
        select.value = currentLambda;
    } else {
        // If new lambdas appeared during training, add them
        const existing = Array.from(select.options).map(o => o.value);
        lambdas.forEach(lam => {
            if (!existing.includes(lam)) {
                const option = document.createElement('option');
                option.value = lam;
                option.textContent = lam;
                select.appendChild(option);
            }
        });
    }
}

document.getElementById('lambdaSelect').addEventListener('change', (e) => {
    currentLambda = e.target.value;
    updateCharts();
});

document.getElementById('refreshBtn').addEventListener('click', (e) => {
    autoRefresh = !autoRefresh;
    e.target.textContent = `Auto-Refresh: ${autoRefresh ? 'ON' : 'OFF'}`;
    e.target.style.opacity = autoRefresh ? '1' : '0.7';
});

function initCharts() {
    // Loss Chart
    const ctxLoss = document.getElementById('lossChart').getContext('2d');
    lossChart = new Chart(ctxLoss, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: colors.grid }, title: { display: true, text: 'Epoch' } },
                y: { grid: { color: colors.grid }, title: { display: true, text: 'Loss' } }
            },
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { position: 'top' } }
        }
    });

    // Accuracy & Sparsity Chart
    const ctxAcc = document.getElementById('accSparsityChart').getContext('2d');
    accSparsityChart = new Chart(ctxAcc, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: colors.grid }, title: { display: true, text: 'Epoch' } },
                y: { 
                    grid: { color: colors.grid }, 
                    title: { display: true, text: 'Percentage (%)' },
                    min: 0, max: 100
                }
            },
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { position: 'top' } }
        }
    });

    // Layer-wise Sparsity Chart
    const ctxLayer = document.getElementById('layerSparsityChart').getContext('2d');
    layerSparsityChart = new Chart(ctxLayer, {
        type: 'bar',
        data: { labels: ['fc1', 'fc2', 'fc3', 'fc4'], datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { display: false } },
                y: { 
                    grid: { color: colors.grid }, 
                    title: { display: true, text: 'Sparsity (%)' },
                    min: 0, max: 100
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function updateCharts() {
    if (!globalMetrics || !currentLambda) return;
    
    const data = globalMetrics[currentLambda];
    const epochs = Array.from({length: data.train_loss.length}, (_, i) => i + 1);

    // Update Loss Chart
    lossChart.data.labels = epochs;
    lossChart.data.datasets = [
        {
            label: 'Total Train Loss',
            data: data.train_loss,
            borderColor: colors.primary,
            backgroundColor: colors.primary + '20',
            borderWidth: 2,
            tension: 0.3,
            fill: true
        },
        {
            label: 'Val Loss',
            data: data.val_loss,
            borderColor: colors.quaternary,
            borderWidth: 2,
            borderDash: [5, 5],
            tension: 0.3
        }
    ];
    lossChart.update();

    // Update Acc & Sparsity Chart
    accSparsityChart.data.labels = epochs;
    accSparsityChart.data.datasets = [
        {
            label: 'Val Accuracy (%)',
            data: data.val_acc,
            borderColor: colors.tertiary,
            borderWidth: 2,
            tension: 0.3
        },
        {
            label: 'Network Sparsity (%)',
            data: data.val_sparsity,
            borderColor: colors.secondary,
            backgroundColor: colors.secondary + '20',
            borderWidth: 2,
            tension: 0.3,
            fill: true
        }
    ];
    accSparsityChart.update();

    // Update Layer-wise Sparsity (Use the latest epoch's data)
    if (data.layer_sparsity && data.layer_sparsity.length > 0) {
        const latestLayerData = data.layer_sparsity[data.layer_sparsity.length - 1];
        layerSparsityChart.data.datasets = [{
            label: 'Layer Sparsity (%)',
            data: latestLayerData,
            backgroundColor: [
                colors.primary + '80',
                colors.secondary + '80',
                colors.tertiary + '80',
                colors.quaternary + '80'
            ],
            borderColor: [
                colors.primary,
                colors.secondary,
                colors.tertiary,
                colors.quaternary
            ],
            borderWidth: 1,
            borderRadius: 4
        }];
        layerSparsityChart.update();
    }
    
    // Refresh the image to bypass cache
    const img = document.getElementById('gateDistImg');
    img.src = `../results/gate_distribution.png?t=${new Date().getTime()}`;
}

// Initialize
initCharts();
fetchMetrics();

// Auto-refresh loop
setInterval(() => {
    if (autoRefresh) {
        fetchMetrics();
    }
}, 2000);
