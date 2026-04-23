'use strict';

const C = {
  bg:      '#0d0f14',
  surface: '#13161f',
  s2:      '#1a1e2b',
  border:  'rgba(255,255,255,0.07)',
  text:    '#e2e8f0',
  muted:   '#7a8399',
  accent:  '#6366f1',
  accent2: '#22d3ee',
  green:   '#10b981',
  red:     '#f43f5e',
  yellow:  '#f59e0b',
};

Chart.defaults.color = C.muted;
Chart.defaults.borderColor = C.border;
Chart.defaults.font.family = "'Inter','Segoe UI',system-ui,sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = C.surface;
Chart.defaults.plugins.tooltip.titleColor = C.text;
Chart.defaults.plugins.tooltip.bodyColor = C.muted;
Chart.defaults.plugins.tooltip.borderColor = C.border;
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 10;

function sparkline(id, data, color) {
  const ctx = document.getElementById(id).getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.map((_, i) => i),
      datasets: [{ data, borderColor: color, borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true,
        backgroundColor: (ctx) => {
          const g = ctx.chart.ctx.createLinearGradient(0,0,0,36);
          g.addColorStop(0, color + '40');
          g.addColorStop(1, color + '00');
          return g;
        }
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: { x: { display: false }, y: { display: false } },
      animation: false,
    }
  });
}

sparkline('spark1', [128,130,132,131,135,138,140,139,142], C.accent);
sparkline('spark2', [2.1,2.2,2.3,2.4,2.5,2.6,2.65,2.72,2.84], C.green);
sparkline('spark3', [82,80,81,79,80,78,79,78,78.3], C.yellow);
sparkline('spark4', [−1.5,−1.48,−1.46,−1.44,−1.43,−1.43,−1.42,−1.42,−1.42], C.accent2);
sparkline('spark5', [1.2,1.8,2.5,2.9,3.4,3.8,4.1,4.5,4.7], C.green);

/* Demand vs Price */
(function() {
  const labels = [];
  const demand = [], optimal = [], current = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date(); d.setDate(d.getDate() - i);
    labels.push(d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
    demand.push(+(800 - i*15 + Math.random()*40).toFixed(0));
    current.push(138);
    optimal.push(142.5);
  }

  const ctx = document.getElementById('demandChart').getContext('2d');
  const g1 = ctx.createLinearGradient(0, 0, 0, 260);
  g1.addColorStop(0, C.accent + '50');
  g1.addColorStop(1, C.accent + '00');

  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Demand Units', data: demand, borderColor: C.accent, backgroundColor: g1,
          fill: true, tension: 0.45, borderWidth: 2.5, pointRadius: 4, pointBackgroundColor: C.accent,
          yAxisID: 'y' },
        { label: 'Current Price ($)', data: current, borderColor: C.yellow, borderDash: [6,3],
          borderWidth: 2, pointRadius: 0, tension: 0, yAxisID: 'y2' },
        { label: 'Optimal Price ($)', data: optimal, borderColor: C.green, borderDash: [0],
          borderWidth: 2, pointRadius: 0, tension: 0, yAxisID: 'y2' },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true,
          labels: { boxWidth: 12, font: { size: 11 }, color: C.muted }
        }
      },
      scales: {
        x: { grid: { color: C.border }, ticks: { font: { size: 11 } } },
        y: {
          type: 'linear', position: 'left',
          grid: { color: C.border },
          ticks: { font: { size: 11 } },
          title: { display: true, text: 'Units', font: { size: 11 }, color: C.muted }
        },
        y2: {
          type: 'linear', position: 'right',
          grid: { drawOnChartArea: false },
          ticks: { font: { size: 11 }, callback: v => '$' + v },
          title: { display: true, text: 'Price ($)', font: { size: 11 }, color: C.muted }
        }
      }
    }
  });
})();

/* Revenue Optimization */
(function() {
  const months = ['Oct','Nov','Dec','Jan','Feb','Mar','Apr'];
  const actual  = [1.8, 1.95, 2.3, 2.1, 2.4, 2.55, 2.84];
  const forecast = [null, null, null, null, null, null, 2.84, 3.1, 3.35, 3.5];
  const fcastLabels = [...months, 'May', 'Jun', 'Jul'];

  const ctx = document.getElementById('revenueChart').getContext('2d');
  const g = ctx.createLinearGradient(0, 0, 0, 220);
  g.addColorStop(0, C.green + '55');
  g.addColorStop(1, C.green + '00');

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: fcastLabels,
      datasets: [
        { label: 'Actual Revenue ($M)', data: [...actual, null, null, null], backgroundColor: C.accent + 'cc',
          borderRadius: 6, barPercentage: 0.6 },
        { label: 'Forecast ($M)', data: forecast, backgroundColor: C.green + '55',
          borderColor: C.green, borderWidth: 2, borderRadius: 6, barPercentage: 0.6 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { boxWidth: 12, font: { size: 11 }, color: C.muted } }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 11 } } },
        y: { grid: { color: C.border }, ticks: { font: { size: 11 }, callback: v => '$' + v + 'M' } }
      }
    }
  });
})();

/* Price Elasticity by Segment */
(function() {
  const ctx = document.getElementById('elasticityChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Enterprise', 'SMB', 'Consumer', 'Government', 'Education'],
      datasets: [{
        label: 'Price Elasticity',
        data: [-0.62, -1.18, -1.85, -0.45, -2.1],
        backgroundColor: [C.yellow, C.accent, C.red, C.green, C.red],
        borderRadius: 6, barPercentage: 0.55,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: C.border }, ticks: { font: { size: 11 } },
          title: { display: true, text: 'Elasticity coefficient', font: { size: 11 }, color: C.muted } },
        y: { grid: { display: false }, ticks: { font: { size: 12 } } }
      }
    }
  });
})();

/* Competitor Benchmark */
(function() {
  const ctx = document.getElementById('competitorChart').getContext('2d');
  const labels = [];
  for (let i = 13; i >= 0; i--) {
    const d = new Date(); d.setDate(d.getDate() - i);
    labels.push(d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
  }
  const us = Array.from({length:14}, (_,i) => 136 + i * .35 + (Math.random() - .5) * 2);
  const comp1 = Array.from({length:14}, (_,i) => 145 + (Math.random() - .5) * 4);
  const comp2 = Array.from({length:14}, (_,i) => 132 + i * .2 + (Math.random() - .5) * 3);

  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Our Price', data: us, borderColor: C.accent, borderWidth: 2.5,
          tension: 0.4, pointRadius: 0 },
        { label: 'Competitor A', data: comp1, borderColor: C.yellow, borderWidth: 1.5,
          tension: 0.4, pointRadius: 0, borderDash: [5,3] },
        { label: 'Competitor B', data: comp2, borderColor: C.red, borderWidth: 1.5,
          tension: 0.4, pointRadius: 0, borderDash: [5,3] },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { boxWidth: 12, font: { size: 11 }, color: C.muted } } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 10 }, maxTicksLimit: 7 } },
        y: { grid: { color: C.border }, ticks: { font: { size: 11 }, callback: v => '$' + v.toFixed(0) } }
      }
    }
  });
})();

/* Scenario Analysis */
(function() {
  const ctx = document.getElementById('scenarioChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['-10%', '-5%', '0%', '+5%', '+10%', '+15%'],
      datasets: [{
        label: 'Revenue Impact ($K)',
        data: [-182, -88, 0, 95, 178, 248],
        backgroundColor: (ctx) => ctx.raw >= 0 ? C.green + 'bb' : C.red + 'bb',
        borderRadius: 5, barPercentage: 0.65,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 11 } } },
        y: { grid: { color: C.border }, ticks: { font: { size: 11 }, callback: v => '$' + v + 'K' } }
      }
    }
  });
})();

/* Recommendations Table */
const recs = [
  { product: 'Enterprise Suite Pro', category: 'Software', current: 138.00, optimal: 142.50, revImpact: '+$48K', conf: 91 },
  { product: 'Analytics Add-on', category: 'Software', current: 49.00, optimal: 44.99, revImpact: '+$12K', conf: 78 },
  { product: 'API Access (10K calls)', category: 'Usage', current: 29.00, optimal: 32.00, revImpact: '+$9K', conf: 85 },
  { product: 'Support Tier Gold', category: 'Service', current: 199.00, optimal: 199.00, revImpact: '—', conf: 60 },
  { product: 'Data Export Bundle', category: 'Feature', current: 15.00, optimal: 12.50, revImpact: '+$3K', conf: 72 },
  { product: 'Custom Integrations', category: 'Service', current: 500.00, optimal: 549.00, revImpact: '+$22K', conf: 88 },
];

const tbody = document.getElementById('recoTableBody');
recs.forEach(r => {
  const delta = r.optimal - r.current;
  const pct = (delta / r.current * 100).toFixed(1);
  const dir = delta > 0 ? 'up' : delta < 0 ? 'down' : 'hold';
  const label = dir === 'up' ? `+$${delta.toFixed(2)} (+${pct}%)` : dir === 'down' ? `$${delta.toFixed(2)} (${pct}%)` : '—';
  tbody.innerHTML += `<tr>
    <td><strong>${r.product}</strong></td>
    <td style="color:var(--muted)">${r.category}</td>
    <td>$${r.current.toFixed(2)}</td>
    <td><strong>$${r.optimal.toFixed(2)}</strong></td>
    <td><span class="badge badge-${dir}">${label}</span></td>
    <td style="color:${dir==='up'||r.revImpact==='—'&&dir==='hold'?'var(--green)':'var(--green)'}"><strong>${r.revImpact}</strong></td>
    <td>
      <div class="conf-bar">
        <div class="conf-track"><div class="conf-fill" style="width:${r.conf}%"></div></div>
        <span>${r.conf}%</span>
      </div>
    </td>
    <td><button class="action-btn">Apply</button></td>
  </tr>`;
});

/* Tab switcher */
document.querySelectorAll('.chart-controls .tab').forEach(btn => {
  btn.addEventListener('click', () => {
    btn.closest('.chart-controls').querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
});
