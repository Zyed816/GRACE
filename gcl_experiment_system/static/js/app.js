document.addEventListener("DOMContentLoaded", function () {
  const monitorEl = document.getElementById("monitor-chart");
  if (monitorEl && window.__MONITOR_ENDPOINT__) {
    const chart = echarts.init(monitorEl);
    const statusEl = document.getElementById("experiment-status");

    const renderPayload = (payload) => {
      const epochs = payload.logs.map((item) => item.epoch);
      const loss = payload.logs.map((item) => item.loss);
      const accuracy = payload.logs.map((item) => item.accuracy);
      if (statusEl) {
        statusEl.textContent = payload.status || "-";
      }
      chart.setOption({
        tooltip: { trigger: "axis" },
        legend: { data: ["Loss", "Accuracy"] },
        xAxis: { type: "category", data: epochs },
        yAxis: [{ type: "value" }],
        series: [
          { name: "Loss", type: "line", data: loss, smooth: true },
          { name: "Accuracy", type: "line", data: accuracy, smooth: true },
        ],
      });
      const doneStates = ["succeeded", "failed", "cancelled"];
      return doneStates.includes(payload.status);
    };

    const tick = () => {
      fetch(window.__MONITOR_ENDPOINT__)
        .then((res) => res.json())
        .then((payload) => {
          const done = renderPayload(payload);
          if (!done) {
            window.setTimeout(tick, 5000);
          }
        })
        .catch(() => {
          window.setTimeout(tick, 8000);
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
