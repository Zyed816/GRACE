const STATUS_LABELS = {
  pending: "等待中",
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

function translateStatus(status) {
  return STATUS_LABELS[status] || status || "-";
}

function updateStatusPill(statusEl, status) {
  if (!statusEl) return;
  statusEl.textContent = translateStatus(status);
  statusEl.className = "status-pill";
  if (status) {
    statusEl.classList.add(`status-${status}`);
  }
}

function updateMetricCards(payload) {
  const metrics = payload.metrics || {};
  const metricNodes = document.querySelectorAll("[data-metric]");
  metricNodes.forEach((node) => {
    const key = node.getAttribute("data-metric");
    if (key && Object.prototype.hasOwnProperty.call(metrics, key)) {
      node.textContent = metrics[key] == null ? "-" : metrics[key];
    }
  });
}

function updateTerminalPanel(payload) {
  const terminalEl = document.getElementById("terminal-output");
  const metaEl = document.getElementById("terminal-meta");
  const artifactEl = document.getElementById("artifact-json");
  const cancelFlag = document.getElementById("cancel-requested-flag");
  if (!terminalEl) return;

  terminalEl.textContent = payload.terminal_tail || "";
  terminalEl.scrollTop = terminalEl.scrollHeight;

  if (metaEl) {
    const total = payload.terminal_total_lines ?? 0;
    const updated = payload.terminal_updated_at || "-";
    metaEl.textContent = `行数：${total} | 更新时间：${updated}`;
  }

  if (artifactEl && payload.artifacts) {
    artifactEl.textContent = JSON.stringify(payload.artifacts, null, 2);
  }

  if (cancelFlag) {
    cancelFlag.textContent = payload.cancel_requested ? "已发送停止请求，等待 Worker 结束任务..." : "";
  }
}

function renderTrainCharts(payload) {
  const monitorEl = document.getElementById("monitor-chart");
  if (!monitorEl || !window.echarts) return;

  const semanticEl = document.getElementById("semantic-chart");
  const lossChart = echarts.getInstanceByDom(monitorEl) || echarts.init(monitorEl);
  const semanticChart = semanticEl ? (echarts.getInstanceByDom(semanticEl) || echarts.init(semanticEl)) : null;

  const epochs = payload.logs.map((item) => item.epoch);
  const loss = payload.logs.map((item) => item.loss);
  const violationRate = payload.logs.map((item) => {
    const p = item.payload || {};
    const v = Number(p.violation_rate);
    return Number.isFinite(v) ? v : null;
  });
  const meanMargin = payload.logs.map((item) => {
    const p = item.payload || {};
    const v = Number(p.mean_margin);
    return Number.isFinite(v) ? v : null;
  });
  const meanPosSim = payload.logs.map((item) => {
    const p = item.payload || {};
    const v = Number(p.mean_pos_sim);
    return Number.isFinite(v) ? v : null;
  });
  const meanMaxNegSim = payload.logs.map((item) => {
    const p = item.payload || {};
    const v = Number(p.mean_max_neg_sim);
    return Number.isFinite(v) ? v : null;
  });

  lossChart.setOption({
    tooltip: { trigger: "axis" },
    legend: { data: ["训练损失", "违反率", "平均间隔"] },
    xAxis: { type: "category", data: epochs },
    yAxis: [{ type: "value" }, { type: "value" }],
    series: [
      { name: "训练损失", type: "line", data: loss, smooth: true, yAxisIndex: 0 },
      { name: "违反率", type: "line", data: violationRate, smooth: true, yAxisIndex: 1 },
      { name: "平均间隔", type: "line", data: meanMargin, smooth: true, yAxisIndex: 1 },
    ],
  });

  if (semanticChart) {
    semanticChart.setOption({
      tooltip: { trigger: "axis" },
      legend: { data: ["正样本相似度", "最大负样本相似度"] },
      xAxis: { type: "category", data: epochs },
      yAxis: [{ type: "value", name: "相似度" }],
      series: [
        { name: "正样本相似度", type: "line", data: meanPosSim, smooth: true },
        { name: "最大负样本相似度", type: "line", data: meanMaxNegSim, smooth: true },
      ],
    });
  }
}

function setupStopButton() {
  const stopBtn = document.getElementById("stop-task-btn");
  if (!stopBtn || !window.__STOP_ENDPOINT__) return;
  stopBtn.addEventListener("click", (event) => {
    event.preventDefault();
    stopBtn.disabled = true;
    fetch(window.__STOP_ENDPOINT__, {
      method: "POST",
      headers: {
        "X-Requested-With": "XMLHttpRequest",
        "X-CSRFToken": getCsrfToken(),
      },
    })
      .then(() => {
        stopBtn.textContent = "已发送停止请求";
      })
      .catch(() => {
        stopBtn.disabled = false;
      });
  });
}

function getCsrfToken() {
  const el = document.querySelector("input[name=csrfmiddlewaretoken]");
  return el ? el.value : "";
}

document.addEventListener("DOMContentLoaded", function () {
  const statusEl = document.getElementById("experiment-status");

  if (window.__MONITOR_ENDPOINT__) {
    setupStopButton();

    const doneStates = ["succeeded", "failed", "cancelled"];
    const tick = () => {
      fetch(window.__MONITOR_ENDPOINT__)
        .then((res) => res.json())
        .then((payload) => {
          updateStatusPill(statusEl, payload.status);
          updateMetricCards(payload);
          updateTerminalPanel(payload);
          if (payload.task_type === "train") {
            renderTrainCharts(payload);
          }
          if (!doneStates.includes(payload.status)) {
            window.setTimeout(tick, 1000);
          }
        })
        .catch(() => {
          window.setTimeout(tick, 2000);
        });
    };

    tick();
  }

  const resultEl = document.getElementById("result-chart");
  if (resultEl && window.__RESULT_PAYLOAD__) {
    const chart = echarts.init(resultEl);
    chart.setOption(window.__RESULT_PAYLOAD__);
  }
});
