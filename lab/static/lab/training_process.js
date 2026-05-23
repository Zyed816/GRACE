(function () {
  const root = document.querySelector(".process-workbench");
  if (!root) {
    return;
  }

  const lang = root.dataset.lang === "en" ? "en" : "zh";
  const isZh = lang === "zh";

  const labels = {
    play: isZh ? "播放" : "Play",
    pause: isZh ? "暂停" : "Pause",
    stagePrefix: isZh ? "阶段" : "Stage",
    className: isZh ? "类别" : "Class",
    degree: isZh ? "度" : "Degree",
    viewDrop: isZh ? "drop" : "drop",
    candidate: isZh ? "候选负样本" : "candidate negative",
    mined: isZh ? "隐藏正样本" : "hidden positive",
    score: isZh ? "相似度" : "score",
  };

  const stages = [
    {
      progress: 0,
      title: isZh ? "原始图" : "Original Graph",
      copy: isZh
        ? "初始表示还比较混杂，不同类别的节点在空间中仍然交织在一起。"
        : "The initial representation is still mixed, and many nodes from different classes stay close in space.",
    },
    {
      progress: 18,
      title: isZh ? "数据增强" : "Graph Augmentation",
      copy: isZh
        ? "同一张图被扰动成两个增强视图，边和节点特征会被轻微改变。"
        : "The graph is perturbed into two augmented views with slightly different edges and node features.",
    },
    {
      progress: 42,
      title: isZh ? "预热训练" : "Warmup Training",
      copy: isZh
        ? "模型先用常规对比学习稳定表示，同类节点开始靠近，异类节点逐渐分开。"
        : "The model first stabilizes representations with standard contrastive learning; similar nodes begin to move together.",
    },
    {
      progress: 62,
      title: isZh ? "隐藏正样本挖掘" : "Hidden Positive Mining",
      copy: isZh
        ? "根据当前表示相似度筛选可信样本对，部分原本按负样本处理的节点对被改判为隐藏正样本。"
        : "Reliable pairs are selected from the current embedding space; some pairs previously treated as negatives become hidden positives.",
    },
    {
      progress: 84,
      title: isZh ? "加权修正训练" : "Weighted Correction Training",
      copy: isZh
        ? "挖掘出的隐藏正样本带着语义权重进入损失函数，可信度更高的样本对产生更强的拉近作用。"
        : "Mined hidden positives enter the loss with semantic weights, so more reliable pairs pull together more strongly.",
    },
    {
      progress: 100,
      title: isZh ? "收敛表示" : "Final Representation",
      copy: isZh
        ? "训练后表示空间更加清晰，同类节点形成更紧凑的簇，不同类别之间的间隔更明显。"
        : "The final space is clearer: nodes from the same class are compact, and different classes are better separated.",
    },
  ];

  const state = {
    progress: 0,
    target: 0,
    playing: false,
    method: "sg-gr",
    hoverNode: null,
    lastPairKey: "",
  };

  const mainCanvas = document.getElementById("sgProcessCanvas");
  const viewOne = document.getElementById("viewCanvasOne");
  const viewTwo = document.getElementById("viewCanvasTwo");
  const tooltip = document.getElementById("processTooltip");
  const timeline = document.getElementById("processTimeline");
  const playButton = document.getElementById("processPlay");
  const resetButton = document.getElementById("processReset");
  const stageTitle = document.getElementById("processStageTitle");
  const stageCopy = document.getElementById("processStageCopy");
  const stageKicker = document.getElementById("processStageKicker");
  const metricEpoch = document.getElementById("metricEpoch");
  const metricMinedPairs = document.getElementById("metricMinedPairs");
  const metricWeight = document.getElementById("metricWeight");
  const metricCompactness = document.getElementById("metricCompactness");
  const viewOneStat = document.getElementById("viewOneStat");
  const viewTwoStat = document.getElementById("viewTwoStat");
  const pairStat = document.getElementById("pairStat");
  const pairList = document.getElementById("pairList");
  const methodButtons = Array.from(document.querySelectorAll("[data-method]"));
  const stageButtons = Array.from(document.querySelectorAll("[data-progress]"));

  const classColors = ["#60a5fa", "#f59e0b", "#34d399", "#e879f9"];
  const centers = [
    { x: 0.26, y: 0.34 },
    { x: 0.72, y: 0.32 },
    { x: 0.48, y: 0.72 },
  ];

  let screenNodes = [];
  let mouse = null;
  let lastTime = performance.now();

  function rand(seed) {
    const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
    return x - Math.floor(x);
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function smooth(edge0, edge1, value) {
    const t = clamp((value - edge0) / (edge1 - edge0), 0, 1);
    return t * t * (3 - 2 * t);
  }

  function resizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(320, Math.round(rect.width * ratio));
    const height = Math.max(190, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    return { width, height, ratio };
  }

  function createNodes() {
    const nodes = [];
    for (let cls = 0; cls < 3; cls += 1) {
      for (let i = 0; i < 18; i += 1) {
        const id = cls * 18 + i;
        const angle = rand(id + 3) * Math.PI * 2;
        const spread = 0.08 + rand(id + 17) * 0.22;
        const center = centers[cls];
        const mixedCenter = centers[(cls + (i % 2 === 0 ? 1 : 2)) % centers.length];
        const mix = 0.28 + rand(id + 9) * 0.28;
        const initialX = lerp(center.x, mixedCenter.x, mix) + Math.cos(angle) * spread * 0.7;
        const initialY = lerp(center.y, mixedCenter.y, mix) + Math.sin(angle) * spread * 0.7;
        const warmX = center.x + (rand(id + 31) - 0.5) * 0.2;
        const warmY = center.y + (rand(id + 41) - 0.5) * 0.2;
        const finalX = center.x + (rand(id + 51) - 0.5) * 0.105;
        const finalY = center.y + (rand(id + 61) - 0.5) * 0.105;
        nodes.push({
          id,
          cls,
          degree: 0,
          initial: { x: clamp(initialX, 0.08, 0.92), y: clamp(initialY, 0.1, 0.9) },
          warm: { x: clamp(warmX, 0.08, 0.92), y: clamp(warmY, 0.1, 0.9) },
          final: { x: clamp(finalX, 0.08, 0.92), y: clamp(finalY, 0.1, 0.9) },
        });
      }
    }
    return nodes;
  }

  const nodes = createNodes();

  function createEdges() {
    const edges = [];
    for (let cls = 0; cls < 3; cls += 1) {
      const base = cls * 18;
      for (let i = 0; i < 18; i += 1) {
        edges.push(makeEdge(base + i, base + ((i + 1) % 18), true, i));
        if (i % 3 === 0) {
          edges.push(makeEdge(base + i, base + ((i + 5) % 18), true, i + 100));
        }
      }
    }

    for (let i = 0; i < 16; i += 1) {
      const a = Math.floor(rand(i + 70) * nodes.length);
      const otherClass = (nodes[a].cls + 1 + (i % 2)) % 3;
      const b = otherClass * 18 + Math.floor(rand(i + 91) * 18);
      edges.push(makeEdge(a, b, false, i + 200));
    }

    edges.forEach((edge) => {
      nodes[edge.a].degree += 1;
      nodes[edge.b].degree += 1;
    });
    return edges;
  }

  function makeEdge(a, b, sameClass, seed) {
    return {
      a,
      b,
      sameClass,
      keepOne: rand(seed + 13) > 0.18,
      keepTwo: rand(seed + 23) > 0.25,
    };
  }

  const edges = createEdges();

  function createCandidates() {
    const pairs = [];
    for (let cls = 0; cls < 3; cls += 1) {
      const base = cls * 18;
      for (let i = 0; i < 5; i += 1) {
        const a = base + ((i * 3 + 1) % 18);
        const b = base + ((i * 3 + 8 + cls) % 18);
        pairs.push({
          a,
          b,
          score: 0.68 + rand(a * 17 + b) * 0.24,
          threshold: 0.12 + pairs.length * 0.075,
        });
      }
    }
    return pairs.sort((left, right) => right.score - left.score).slice(0, 13);
  }

  const candidates = createCandidates();

  function getNodePosition(node, progressPercent, viewShift) {
    const p = progressPercent / 100;
    const warmT = smooth(0.26, 0.52, p);
    const correctionT = smooth(0.72, 1, p);
    const methodBoost = state.method === "sg-gc" ? 1.06 : 1;
    const jitter = viewShift || { x: 0, y: 0 };
    const x1 = lerp(node.initial.x, node.warm.x, warmT);
    const y1 = lerp(node.initial.y, node.warm.y, warmT);
    const x2 = lerp(x1, node.final.x, clamp(correctionT * methodBoost, 0, 1));
    const y2 = lerp(y1, node.final.y, clamp(correctionT * methodBoost, 0, 1));
    return {
      x: clamp(x2 + jitter.x, 0.04, 0.96),
      y: clamp(y2 + jitter.y, 0.06, 0.94),
    };
  }

  function project(position, width, height) {
    const padX = width * 0.07;
    const padY = height * 0.08;
    return {
      x: padX + position.x * (width - padX * 2),
      y: padY + position.y * (height - padY * 2),
    };
  }

  function drawGrid(ctx, width, height) {
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "rgba(148, 163, 184, 0.09)";
    ctx.lineWidth = 1;
    const step = Math.max(32, Math.round(width / 18));
    for (let x = 0; x <= width; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y <= height; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  }

  function drawLine(ctx, a, b, color, width, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.restore();
  }

  function candidateMiningRatio(progressPercent) {
    return smooth(0.53, 0.72, progressPercent / 100);
  }

  function getMinedCandidates(progressPercent) {
    const ratio = candidateMiningRatio(progressPercent);
    return candidates.filter((pair) => ratio >= pair.threshold);
  }

  function isNodeMined(nodeId, progressPercent) {
    return getMinedCandidates(progressPercent).some((pair) => pair.a === nodeId || pair.b === nodeId);
  }

  function drawMain() {
    const info = resizeCanvas(mainCanvas);
    const ctx = mainCanvas.getContext("2d");
    const width = info.width;
    const height = info.height;
    const p = state.progress;
    const p01 = p / 100;
    const augT = smooth(0.1, 0.24, p01);
    const miningT = candidateMiningRatio(p);
    const correctionT = smooth(0.72, 1, p01);

    drawGrid(ctx, width, height);

    const positions = nodes.map((node) => project(getNodePosition(node, p), width, height));
    screenNodes = positions.map((point, index) => ({ ...point, node: nodes[index] }));

    edges.forEach((edge) => {
      const alpha = edge.sameClass ? 0.2 : 0.07;
      drawLine(ctx, positions[edge.a], positions[edge.b], edge.sameClass ? "#2dd4bf" : "#fb7185", edge.sameClass ? 1.2 : 0.8, alpha);
    });

    if (augT > 0.02 && p01 < 0.5) {
      const oneShift = { x: -0.018 * augT, y: 0.012 * augT };
      const twoShift = { x: 0.018 * augT, y: -0.012 * augT };
      const onePositions = nodes.map((node) => project(getNodePosition(node, p, oneShift), width, height));
      const twoPositions = nodes.map((node) => project(getNodePosition(node, p, twoShift), width, height));
      edges.forEach((edge) => {
        if (edge.keepOne) {
          drawLine(ctx, onePositions[edge.a], onePositions[edge.b], "#38bdf8", 1, 0.18 * augT);
        }
        if (edge.keepTwo) {
          drawLine(ctx, twoPositions[edge.a], twoPositions[edge.b], "#f97316", 1, 0.16 * augT);
        }
      });
    }

    if (p01 > 0.48) {
      candidates.forEach((pair) => {
        const mined = miningT >= pair.threshold;
        const color = mined ? "#facc15" : "#fb7185";
        const lineWidth = mined ? 3.2 : 1.5;
        const alpha = mined ? 0.28 + correctionT * 0.34 : 0.16;
        drawLine(ctx, positions[pair.a], positions[pair.b], color, lineWidth, alpha);
      });
    }

    nodes.forEach((node, index) => {
      const point = positions[index];
      const mined = isNodeMined(node.id, p);
      const r = mined ? 7.6 : 6.2;
      if (mined) {
        ctx.save();
        ctx.globalAlpha = 0.22 + correctionT * 0.18;
        ctx.fillStyle = "#facc15";
        ctx.beginPath();
        ctx.arc(point.x, point.y, r + 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      ctx.save();
      ctx.shadowColor = classColors[node.cls];
      ctx.shadowBlur = state.hoverNode === node.id ? 18 : 8;
      ctx.fillStyle = classColors[node.cls];
      ctx.strokeStyle = mined ? "#facc15" : "rgba(255, 255, 255, 0.82)";
      ctx.lineWidth = mined ? 2.8 : 1.4;
      ctx.beginPath();
      ctx.arc(point.x, point.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    });

    if (state.hoverNode !== null) {
      const item = screenNodes.find((entry) => entry.node.id === state.hoverNode);
      if (item) {
        ctx.save();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.88)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(item.x, item.y, 13, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
    }
  }

  function drawMini(canvas, view) {
    const info = resizeCanvas(canvas);
    const ctx = canvas.getContext("2d");
    const width = info.width;
    const height = info.height;
    const p = state.progress;
    const active = smooth(0.12, 0.24, p / 100);
    const shift = view === 1 ? { x: -0.024 * active, y: 0.018 * active } : { x: 0.024 * active, y: -0.018 * active };
    const positions = nodes.map((node) => project(getNodePosition(node, p, shift), width, height));

    ctx.fillStyle = "#111827";
    ctx.fillRect(0, 0, width, height);

    edges.forEach((edge) => {
      const keep = view === 1 ? edge.keepOne : edge.keepTwo;
      const color = view === 1 ? "#38bdf8" : "#f97316";
      drawLine(ctx, positions[edge.a], positions[edge.b], color, 1, keep ? 0.26 + active * 0.28 : 0.035);
    });

    nodes.forEach((node, index) => {
      const point = positions[index];
      const masked = rand(node.id + view * 101) < (view === 1 ? 0.12 : 0.16);
      ctx.save();
      ctx.fillStyle = masked && active > 0.2 ? "rgba(148, 163, 184, 0.72)" : classColors[node.cls];
      ctx.strokeStyle = "rgba(255, 255, 255, 0.75)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    });
  }

  function updateTextAndMetrics() {
    const stage = stages.reduce((active, item) => (state.progress >= item.progress ? item : active), stages[0]);
    const stageIndex = stages.indexOf(stage) + 1;
    const mined = getMinedCandidates(state.progress);
    const epoch = Math.round((state.progress / 100) * 400);
    const avgWeight = mined.length
      ? mined.reduce((sum, pair) => sum + Math.exp(0.7 * pair.score), 0) / mined.length
      : 0;
    const compactness = Math.round(18 + smooth(0.28, 1, state.progress / 100) * (state.method === "sg-gc" ? 76 : 72));

    stageTitle.textContent = stage.title;
    stageCopy.textContent = stage.copy;
    stageKicker.textContent = `${labels.stagePrefix} ${String(stageIndex).padStart(2, "0")}`;
    metricEpoch.textContent = `${epoch}`;
    metricMinedPairs.textContent = `${mined.length}`;
    metricWeight.textContent = avgWeight.toFixed(2);
    metricCompactness.textContent = `${compactness}%`;

    const oneDrop = Math.round((1 - edges.filter((edge) => edge.keepOne).length / edges.length) * 100);
    const twoDrop = Math.round((1 - edges.filter((edge) => edge.keepTwo).length / edges.length) * 100);
    viewOneStat.textContent = `${labels.viewDrop} ${state.progress >= 12 ? oneDrop : 0}%`;
    viewTwoStat.textContent = `${labels.viewDrop} ${state.progress >= 12 ? twoDrop : 0}%`;
    pairStat.textContent = `${mined.length} / ${candidates.length}`;

    stageButtons.forEach((button) => {
      const isActive = Number(button.dataset.progress) === stage.progress;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-pressed", isActive ? "true" : "false");
    });

    methodButtons.forEach((button) => {
      const isActive = button.dataset.method === state.method;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-selected", isActive ? "true" : "false");
    });

    const pairKey = `${mined.length}-${Math.round(state.progress)}-${lang}`;
    if (pairKey !== state.lastPairKey) {
      state.lastPairKey = pairKey;
      const miningRatio = candidateMiningRatio(state.progress);
      pairList.innerHTML = candidates
        .map((pair) => {
          const isMined = miningRatio >= pair.threshold;
          const label = isMined ? labels.mined : labels.candidate;
          return [
            `<div class="pair-item${isMined ? " is-mined" : ""}">`,
            `<span>N${String(pair.a).padStart(2, "0")} - N${String(pair.b).padStart(2, "0")}<br><small>${label}</small></span>`,
            `<small>${labels.score} ${pair.score.toFixed(2)}</small>`,
            "</div>",
          ].join("");
        })
        .join("");
    }
  }

  function updateTooltip() {
    if (!mouse || state.hoverNode === null) {
      tooltip.hidden = true;
      return;
    }
    const item = screenNodes.find((entry) => entry.node.id === state.hoverNode);
    if (!item) {
      tooltip.hidden = true;
      return;
    }
    tooltip.innerHTML = [
      `<strong>N${String(item.node.id).padStart(2, "0")}</strong>`,
      `<br>${labels.className}: C${item.node.cls + 1}`,
      `<br>${labels.degree}: ${item.node.degree}`,
    ].join("");
    tooltip.style.left = `${Math.min(mouse.x + 16, mainCanvas.clientWidth - 150)}px`;
    tooltip.style.top = `${Math.max(12, mouse.y - 28)}px`;
    tooltip.hidden = false;
  }

  function draw() {
    drawMain();
    drawMini(viewOne, 1);
    drawMini(viewTwo, 2);
    updateTextAndMetrics();
    updateTooltip();
  }

  function tick(now) {
    const dt = Math.min(0.06, (now - lastTime) / 1000);
    lastTime = now;

    if (state.playing) {
      state.progress = Math.min(100, state.progress + dt * 12);
      state.target = state.progress;
      if (state.progress >= 100) {
        state.playing = false;
      }
    } else {
      const delta = state.target - state.progress;
      state.progress += delta * Math.min(1, dt * 9);
      if (Math.abs(delta) < 0.08) {
        state.progress = state.target;
      }
    }

    timeline.value = String(Math.round(state.progress));
    playButton.textContent = state.playing ? labels.pause : labels.play;
    draw();
    requestAnimationFrame(tick);
  }

  function setProgress(value) {
    const next = clamp(Number(value) || 0, 0, 100);
    state.progress = next;
    state.target = next;
    state.playing = false;
    draw();
  }

  timeline.addEventListener("input", (event) => {
    setProgress(event.target.value);
  });

  playButton.addEventListener("click", () => {
    state.playing = !state.playing;
    if (state.progress >= 100) {
      state.progress = 0;
      state.target = 0;
    }
  });

  resetButton.addEventListener("click", () => {
    state.playing = false;
    state.progress = 0;
    state.target = 0;
    draw();
  });

  methodButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.method = button.dataset.method;
      draw();
    });
  });

  stageButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.playing = false;
      state.target = Number(button.dataset.progress);
    });
  });

  mainCanvas.addEventListener("mousemove", (event) => {
    const rect = mainCanvas.getBoundingClientRect();
    mouse = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    const ratioX = mainCanvas.width / rect.width;
    const ratioY = mainCanvas.height / rect.height;
    const canvasMouse = { x: mouse.x * ratioX, y: mouse.y * ratioY };
    let nearest = null;
    let nearestDistance = 16 * (window.devicePixelRatio || 1);
    screenNodes.forEach((entry) => {
      const distance = Math.hypot(entry.x - canvasMouse.x, entry.y - canvasMouse.y);
      if (distance < nearestDistance) {
        nearest = entry.node.id;
        nearestDistance = distance;
      }
    });
    state.hoverNode = nearest;
  });

  mainCanvas.addEventListener("mouseleave", () => {
    mouse = null;
    state.hoverNode = null;
    tooltip.hidden = true;
  });

  window.addEventListener("resize", draw);

  draw();
  requestAnimationFrame(tick);
})();
