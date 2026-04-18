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

  const lossChart = echarts.getInstanceByDom(monitorEl) || echarts.init(monitorEl);

  const epochs = payload.logs.map((item) => item.epoch);
  const loss = payload.logs.map((item) => item.loss);

  lossChart.setOption({
    tooltip: { trigger: "axis" },
    legend: { data: ["训练损失"] },
    xAxis: { type: "category", data: epochs },
    yAxis: [{ type: "value", name: "Loss" }],
    series: [
      { name: "训练损失", type: "line", data: loss, smooth: true },
    ],
  });
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

function setupDeleteExperimentForms() {
  const forms = document.querySelectorAll(".js-delete-experiment-form");
  forms.forEach((form) => {
    form.addEventListener("submit", (event) => {
      const message = form.getAttribute("data-confirm") || "确认删除该任务吗？";
      if (!window.confirm(message)) {
        event.preventDefault();
        return;
      }
      const btn = form.querySelector("button[type=submit]");
      if (btn) {
        btn.disabled = true;
        btn.textContent = "删除中...";
      }
    });
  });
}

document.addEventListener("DOMContentLoaded", function () {
  const statusEl = document.getElementById("experiment-status");
  setupDeleteExperimentForms();

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
