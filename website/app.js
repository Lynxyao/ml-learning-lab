const views = document.querySelectorAll(".view");
const navButtons = document.querySelectorAll("[data-target]");
const repoUrl = (window.SITE_REPO_URL || "").replace(/\/$/, "");

document.querySelectorAll(".repo-link").forEach((link) => {
  const path = link.dataset.repoPath || "";
  if (!repoUrl || repoUrl.includes("YOUR_USERNAME")) {
    link.setAttribute("aria-disabled", "true");
    link.title = "Set SITE_REPO_URL in deploy-config.js after creating the GitHub repository.";
    return;
  }

  const isFile = /\.[A-Za-z0-9]+$/.test(path);
  const pathKind = isFile ? "blob" : "tree";
  link.href = path ? `${repoUrl}/${pathKind}/main/${path}` : repoUrl;
  link.target = "_blank";
  link.rel = "noreferrer";
});

function showView(targetId) {
  views.forEach((view) => {
    view.classList.toggle("active", view.id === targetId);
  });

  document.querySelectorAll(".nav-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.target === targetId);
  });

  window.scrollTo({ top: 0, behavior: "smooth" });
}

navButtons.forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.target));
});

const labState = {
  wfm: null,
  ecg: null,
  motion: null,
  resistance: null,
};

let backendAvailable = false;

fetch("/api/health")
  .then((response) => {
    backendAvailable = response.ok;
    updateBackendStatus();
  })
  .catch(() => {
    backendAvailable = false;
    updateBackendStatus();
  });

function updateBackendStatus() {
  const statusText = backendAvailable
    ? "Local backend detected. Realtime training is available on this computer."
    : "Local backend not detected. Online students should use saved experiments.";

  document.querySelector("#wfm-backend-status").textContent = statusText;
  document.querySelector("#ecg-backend-status").textContent = statusText;

  document.querySelectorAll(".advanced-local").forEach((panel) => {
    panel.classList.toggle("offline", !backendAvailable);
  });
}

document.querySelectorAll(".lab-tab").forEach((button) => {
  button.addEventListener("click", () => {
    const moduleName = button.dataset.module;
    const step = button.dataset.step;

    document.querySelectorAll(`.lab-tab[data-module="${moduleName}"]`).forEach((tab) => {
      tab.classList.toggle("active", tab === button);
    });

    document.querySelectorAll(`[data-panel^="${moduleName}-"]`).forEach((panel) => {
      panel.classList.toggle("active", panel.dataset.panel === `${moduleName}-${step}`);
    });
  });
});

function normalizeQuizText(value) {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function evaluateChoiceQuestion(card) {
  const checked = card.querySelector('input[type="radio"]:checked');
  if (!checked) {
    return {
      correct: false,
      message: "Choose one answer first. Then compare your reasoning with the feedback.",
    };
  }

  const correct = checked.value === card.dataset.correct;
  const feedback = correct
    ? card.querySelector(".quiz-feedback").dataset.correctFeedback
    : card.querySelector(".quiz-feedback").dataset.incorrectFeedback;

  return { correct, message: feedback };
}

function evaluateTextQuestion(card) {
  const answer = normalizeQuizText(card.querySelector(".quiz-answer").value);
  if (answer.length < 12) {
    return {
      correct: false,
      message: "Add a little more detail. A useful answer should name the evidence, metric, or validation issue.",
    };
  }

  const keywords = (card.dataset.keywords || "")
    .split("|")
    .map((keyword) => keyword.trim().toLowerCase())
    .filter(Boolean);
  const minMatches = Number(card.dataset.minMatches || 1);
  const matches = keywords.filter((keyword) => answer.includes(keyword)).length;
  const correct = matches >= minMatches;
  const feedback = correct
    ? card.querySelector(".quiz-feedback").dataset.correctFeedback
    : card.querySelector(".quiz-feedback").dataset.incorrectFeedback;

  return { correct, message: feedback };
}

function checkQuiz(quiz) {
  const cards = Array.from(quiz.querySelectorAll(".quiz-card"));
  let correctCount = 0;

  cards.forEach((card) => {
    const feedback = card.querySelector(".quiz-feedback");
    const result = card.dataset.questionType === "text"
      ? evaluateTextQuestion(card)
      : evaluateChoiceQuestion(card);

    card.classList.toggle("correct", result.correct);
    card.classList.toggle("incorrect", !result.correct);
    feedback.textContent = result.message;
    feedback.classList.add("visible");

    if (result.correct) correctCount += 1;
  });

  const summary = quiz.querySelector(".quiz-summary");
  const total = cards.length;
  if (correctCount === total) {
    summary.textContent = `Score: ${correctCount}/${total}. Good. You are ready to move to the mini-project.`;
  } else {
    summary.textContent = `Score: ${correctCount}/${total}. Review the feedback above, revise your reasoning, and check again.`;
  }
}

function resetQuiz(quiz) {
  quiz.querySelectorAll('input[type="radio"]').forEach((input) => {
    input.checked = false;
  });
  quiz.querySelectorAll(".quiz-answer").forEach((answer) => {
    answer.value = "";
  });
  quiz.querySelectorAll(".quiz-card").forEach((card) => {
    card.classList.remove("correct", "incorrect");
  });
  quiz.querySelectorAll(".quiz-feedback").forEach((feedback) => {
    feedback.textContent = "";
    feedback.classList.remove("visible");
  });
  quiz.querySelector(".quiz-summary").textContent = "";
}

document.querySelectorAll("[data-check-quiz]").forEach((button) => {
  button.addEventListener("click", () => {
    const quiz = document.querySelector(`[data-quiz="${button.dataset.checkQuiz}"]`);
    if (quiz) checkQuiz(quiz);
  });
});

document.querySelectorAll("[data-reset-quiz]").forEach((button) => {
  button.addEventListener("click", () => {
    const quiz = document.querySelector(`[data-quiz="${button.dataset.resetQuiz}"]`);
    if (quiz) resetQuiz(quiz);
  });
});

const roleCopy = {
  generator: "The generator learns to produce a target-like image from the input microscopy image.",
  discriminator: "The discriminator learns to judge whether an input-output pair looks real or generated.",
  loss: "GAN loss rewards realism, while L1 loss rewards similarity to the paired target image.",
};

document.querySelectorAll("[data-role]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelector("#wfm-role-copy").textContent = roleCopy[button.dataset.role];
    document.querySelectorAll("[data-role]").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
  });
});

const wfmImage = document.querySelector("#wfm-image");
const wfmCaption = document.querySelector("#wfm-caption");
const wfmModes = {
  paired: {
    src: "assets/wfm-paired-sample.jpg",
    alt: "WFM paired microscopy sample",
    caption: "A paired training image stores input and target domains in one file before preprocessing.",
  },
  result: {
    src: "assets/wfm-result-100.png",
    alt: "WFM generated result comparison",
    caption: "A saved result image lets you compare input, generated output, and reference target.",
  },
};

document.querySelectorAll("[data-wfm-mode]").forEach((button) => {
  button.addEventListener("click", () => {
    const mode = wfmModes[button.dataset.wfmMode];
    wfmImage.src = mode.src;
    wfmImage.alt = mode.alt;
    wfmCaption.textContent = mode.caption;

    document.querySelectorAll("[data-wfm-mode]").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
  });
});

const wfmEpochs = document.querySelector("#wfm-epochs");
wfmEpochs.addEventListener("input", () => {
  document.querySelector("#wfm-epoch-label").textContent = wfmEpochs.value;
});

document.querySelector("#load-wfm-saved").addEventListener("click", () => {
  const epochs = Number(wfmEpochs.value);
  const useGan = document.querySelector("#wfm-use-gan").checked;
  const useL1 = document.querySelector("#wfm-use-l1").checked;
  const noCheckpoints = document.querySelector("#wfm-no-checkpoints").checked;
  runGuidedWfmTraining({ epochs, useGan, useL1, noCheckpoints });
});

document.querySelector("#run-wfm-training").addEventListener("click", () => {
  const epochs = Number(wfmEpochs.value);
  const useGan = document.querySelector("#wfm-use-gan").checked;
  const useL1 = document.querySelector("#wfm-use-l1").checked;
  const noCheckpoints = document.querySelector("#wfm-no-checkpoints").checked;

  if (!backendAvailable) return;
  runRealtimeWfmTraining({ epochs, useGan, useL1, noCheckpoints });
});

function runGuidedWfmTraining({ epochs, useGan, useL1, noCheckpoints }) {
  const meter = document.querySelector("#wfm-meter");
  const copy = document.querySelector("#wfm-training-copy");
  const liveLog = document.querySelector("#wfm-live-log");
  liveLog.classList.remove("active");
  liveLog.textContent = "";
  meter.style.width = "0";
  copy.textContent = "Loading the closest saved WFM checkpoint for comparison...";

  window.setTimeout(() => {
    meter.style.width = "45%";
    copy.textContent = "Saved checkpoints let students compare generated image quality without waiting for GAN training.";
  }, 450);

  window.setTimeout(() => {
    meter.style.width = "78%";
    copy.textContent = "The key question is whether the output preserves structure or only looks visually plausible.";
  }, 1000);

  window.setTimeout(() => {
    meter.style.width = "100%";
    copy.textContent = `Saved checkpoint loaded for epoch ${epochs}. Open the Test step and compare generated output with the reference target.`;
    labState.wfm = { epochs, useGan, useL1, noCheckpoints };
    updateWfmTest();
  }, 1550);
}

function runRealtimeWfmTraining({ epochs, useGan, useL1, noCheckpoints }) {
  const realtimeEpochs = Math.min(epochs, 20);
  const meter = document.querySelector("#wfm-meter");
  const copy = document.querySelector("#wfm-training-copy");
  const liveLog = document.querySelector("#wfm-live-log");
  const l1Lambda = useL1 ? 100 : 0;
  const params = new URLSearchParams({
    epochs: String(realtimeEpochs),
    l1_lambda: String(l1Lambda),
    test_count: "10",
  });

  meter.style.width = "4%";
  liveLog.classList.add("active");
  liveLog.textContent = "";
  copy.textContent =
    epochs > realtimeEpochs
      ? `Starting ${realtimeEpochs}-epoch local WFM quick preview. Longer saved checkpoints are precomputed.`
      : "Starting real WFM GAN training. Even a quick preview can take about a minute.";

  const source = new EventSource(`/api/wfm/train?${params.toString()}`);

  source.addEventListener("status", (event) => {
    const data = JSON.parse(event.data);
    appendLog(data.message);
    copy.textContent = data.message;
  });

  source.addEventListener("log", (event) => {
    const data = JSON.parse(event.data);
    appendLog(data.line);
    const epochMatch = data.line.match(/==> Epoch (\\d+) finished.*D_loss: ([0-9.]+), G_loss: ([0-9.]+)/);
    if (epochMatch) {
      const current = Number(epochMatch[1]);
      meter.style.width = `${Math.round((current / realtimeEpochs) * 90)}%`;
      copy.textContent = `Epoch ${current}/${realtimeEpochs}: D loss ${epochMatch[2]}, G loss ${epochMatch[3]}.`;
    }
  });

  source.addEventListener("complete", (event) => {
    const data = JSON.parse(event.data);
    meter.style.width = "100%";
    source.close();
    appendLog(data.message);
    copy.textContent = "Real WFM training and testing complete. Open the Test step to inspect your generated image.";
    labState.wfm = {
      epochs,
      realtimeEpochs,
      useGan,
      useL1,
      noCheckpoints,
      realtime: true,
      runId: data.run_id,
      outputRoot: data.output_root,
      fixedImageUrl: data.fixed_image_url,
      testImageUrl: data.test_image_url,
    };
    updateWfmTest();
  });

  source.addEventListener("error", (event) => {
    source.close();
    meter.style.width = "0";
    const message = event.data ? JSON.parse(event.data).message : "Realtime connection failed.";
    appendLog(`ERROR: ${message}`);
    copy.textContent = "Realtime WFM training failed. Try fewer epochs, then check the log.";
  });

  function appendLog(line) {
    liveLog.textContent += `${line}\n`;
    liveLog.scrollTop = liveLog.scrollHeight;
  }
}

function nearestWfmCheckpoint(epochs) {
  const checkpoints = [20, 100, 200];
  return checkpoints.reduce((closest, current) =>
    Math.abs(current - epochs) < Math.abs(closest - epochs) ? current : closest
  );
}

function updateWfmTest() {
  if (!labState.wfm) return;

  const checkpoint = nearestWfmCheckpoint(labState.wfm.epochs);
  const pill = document.querySelector("#wfm-test-run-pill");
  const caption = document.querySelector("#wfm-test-caption");
  const summary = document.querySelector("#wfm-run-summary");
  const image = document.querySelector("#wfm-test-image");

  pill.textContent = `${labState.wfm.epochs} epoch ${labState.wfm.realtime ? "local run" : "saved checkpoint"}`;
  pill.classList.add("ready");
  if (labState.wfm.realtime && labState.wfm.testImageUrl) {
    image.src = labState.wfm.testImageUrl;
    image.alt = `Realtime WFM held-out test result from ${labState.wfm.runId}`;
    caption.textContent = `This is a held-out test image generated by your realtime run ${labState.wfm.runId}.`;
  } else {
    image.src = `assets/wfm-result-${checkpoint}.png`;
    image.alt = `WFM generated result comparison near epoch ${checkpoint}`;
    caption.textContent =
      checkpoint === labState.wfm.epochs
        ? `This is the saved visual checkpoint for epoch ${checkpoint}.`
        : `Your run used ${labState.wfm.epochs} epochs. The closest available saved visual checkpoint is epoch ${checkpoint}.`;
  }
  summary.innerHTML = `
    <strong>Your ${labState.wfm.realtime ? "local realtime preview" : "saved checkpoint exploration"}</strong>
    <p>Epochs: ${labState.wfm.epochs}. GAN loss: ${labState.wfm.useGan ? "on" : "off"}. L1 loss: ${
      labState.wfm.useL1 ? "on" : "off"
    }.${labState.wfm.realtimeEpochs ? ` Local training epochs: ${labState.wfm.realtimeEpochs}.` : ""} Fixed visual checkpoints: ${labState.wfm.noCheckpoints ? "off" : "on"}.${
      labState.wfm.outputRoot ? ` Results folder: ${labState.wfm.outputRoot}.` : ""
    }</p>
  `;
}

const waveformImages = {
  normal: {
    src: "assets/ecg-normal.png",
    alt: "Normal ECG waveform sample",
  },
  sveb: {
    src: "assets/ecg-sveb.png",
    alt: "SVEB ECG waveform sample",
  },
  veb: {
    src: "assets/ecg-veb.png",
    alt: "VEB ECG waveform sample",
  },
};

document.querySelector("#waveform-select").addEventListener("change", (event) => {
  const waveform = waveformImages[event.target.value];
  const image = document.querySelector("#waveform-image");
  image.src = waveform.src;
  image.alt = waveform.alt;
});

document.querySelectorAll('input[name="ecg-split"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    const split = document.querySelector('input[name="ecg-split"]:checked').value;
    document.querySelector("#split-explanation").textContent =
      split === "record"
        ? "Record-wise split is harder because test records are unseen during training, but it is a better generalization test."
        : "Random beat-level split may produce higher scores because similar beats from the same record can appear in both training and testing.";
  });
});

const ecgEpochs = document.querySelector("#ecg-epochs");
ecgEpochs.addEventListener("input", () => {
  document.querySelector("#ecg-epoch-label").textContent = ecgEpochs.value;
});

document.querySelector("#load-ecg-saved").addEventListener("click", () => {
  const model = document.querySelector('input[name="ecg-model"]:checked').value;
  const epochs = Number(ecgEpochs.value);
  const weighting = document.querySelector("#class-weighting").checked;
  runGuidedEcgTraining({ model, epochs, weighting });
});

document.querySelector("#run-ecg-training").addEventListener("click", () => {
  const model = document.querySelector('input[name="ecg-model"]:checked').value;
  const epochs = Number(ecgEpochs.value);
  const weighting = document.querySelector("#class-weighting").checked;
  const splitMode = document.querySelector('input[name="ecg-split"]:checked')?.value || "record";

  if (!backendAvailable) return;
  runRealtimeEcgTraining({ model, epochs, weighting, splitMode });
});

function runGuidedEcgTraining({ model, epochs, weighting }) {
  const meter = document.querySelector("#ecg-meter");
  const copy = document.querySelector("#ecg-training-copy");
  const liveLog = document.querySelector("#ecg-live-log");

  liveLog.classList.remove("active");
  liveLog.textContent = "";
  meter.style.width = "0";
  copy.textContent = `Loading saved ${model.toUpperCase()} experiment summary...`;

  window.setTimeout(() => {
    meter.style.width = "35%";
    copy.textContent = "Saved training curves help students focus on patterns instead of waiting for compute.";
  }, 450);

  window.setTimeout(() => {
    meter.style.width = "72%";
    copy.textContent = weighting
      ? "This saved run uses class weighting, so rare classes influence the objective more."
      : "This saved run represents the risk of favoring the dominant normal class.";
  }, 1000);

  window.setTimeout(() => {
    meter.style.width = "100%";
    copy.textContent = `${epochs}-epoch saved experiment loaded. Go to Test and compare accuracy with macro-F1 before judging the model.`;
    labState.ecg = { model, epochs, weighting };
    updateEcgTest();
  }, 1550);
}

function runRealtimeEcgTraining({ model, epochs, weighting, splitMode }) {
  const meter = document.querySelector("#ecg-meter");
  const copy = document.querySelector("#ecg-training-copy");
  const liveLog = document.querySelector("#ecg-live-log");
  const apiModel = model === "cnnlstm" ? "cnn_lstm" : model;
  const classWeighting = weighting ? "inverse" : "none";
  const params = new URLSearchParams({
    model: apiModel,
    epochs: String(epochs),
    split_mode: splitMode,
    class_weighting: classWeighting,
  });

  meter.style.width = "4%";
  liveLog.classList.add("active");
  liveLog.textContent = "";
  copy.textContent = "Starting real PyTorch training. Keep this page open while logs stream in.";

  const source = new EventSource(`/api/ecg/train?${params.toString()}`);

  source.addEventListener("status", (event) => {
    const data = JSON.parse(event.data);
    appendLog(data.message);
    copy.textContent = data.message;
  });

  source.addEventListener("log", (event) => {
    const data = JSON.parse(event.data);
    appendLog(data.line);
    const match = data.line.match(/\\[Epoch (\\d+)\\/(\\d+)\\].*train_acc=([0-9.]+).*val_acc=([0-9.]+)/);
    if (match) {
      const current = Number(match[1]);
      const total = Number(match[2]);
      meter.style.width = `${Math.round((current / total) * 85)}%`;
      copy.textContent = `Epoch ${current}/${total}: train acc ${match[3]}, validation acc ${match[4]}.`;
    }
    if (data.line.startsWith("Test accuracy:")) {
      copy.textContent = data.line;
    }
  });

  source.addEventListener("complete", (event) => {
    const data = JSON.parse(event.data);
    meter.style.width = "100%";
    source.close();
    appendLog(data.message);
    copy.textContent = "Real ECG training and testing complete. Open the Test step to inspect your metrics.";
    labState.ecg = {
      model,
      epochs,
      weighting,
      realtime: true,
      runId: data.run_id,
      saveRoot: data.save_root,
      metrics: data.metrics,
    };
    updateEcgTest();
  });

  source.addEventListener("error", (event) => {
    source.close();
    meter.style.width = "0";
    const message = event.data ? JSON.parse(event.data).message : "Realtime connection failed.";
    appendLog(`ERROR: ${message}`);
    copy.textContent = "Realtime training failed. Check the log, then try fewer epochs or the guided fallback.";
  });

  function appendLog(line) {
    liveLog.textContent += `${line}\n`;
    liveLog.scrollTop = liveLog.scrollHeight;
  }
}

function updateEcgTest() {
  if (!labState.ecg) return;

  const pill = document.querySelector("#ecg-test-run-pill");
  const summary = document.querySelector("#ecg-run-summary");
  const metrics = labState.ecg.metrics;
  pill.textContent = `${labState.ecg.model.toUpperCase()} · ${labState.ecg.epochs} epochs`;
  pill.classList.add("ready");
  if (metrics) {
    metricContent.accuracy.value = metrics.accuracy.toFixed(3);
    metricContent.accuracy.copy = `This is the real test accuracy from run ${labState.ecg.runId}.`;
    metricContent.macro.value = metrics.macro_f1.toFixed(3);
    metricContent.macro.copy = `This is the real macro-F1 from run ${labState.ecg.runId}. Compare it with accuracy before judging the model.`;
    document.querySelector("#metric-value").textContent = metricContent.accuracy.value;
    document.querySelector("#metric-copy").textContent = metricContent.accuracy.copy;
    document.querySelectorAll("[data-metric]").forEach((item) => {
      item.classList.toggle("active", item.dataset.metric === "accuracy");
    });
  }
  summary.innerHTML = labState.ecg.realtime
    ? `
      <strong>Your realtime run</strong>
      <p>Model: ${labState.ecg.model.toUpperCase()}. Epochs: ${labState.ecg.epochs}. Class weighting: ${
        labState.ecg.weighting ? "on" : "off"
      }. Results folder: ${labState.ecg.saveRoot}.</p>
    `
    : `
      <strong>Your saved experiment</strong>
      <p>Model: ${labState.ecg.model.toUpperCase()}. Epochs: ${labState.ecg.epochs}. Class weighting: ${
        labState.ecg.weighting ? "on" : "off"
      }. The displayed metrics use the prepared CNN baseline for online student exploration.</p>
    `;
}

const metricContent = {
  accuracy: {
    value: "0.780",
    copy: "Overall correctness looks reasonable because normal beats dominate the test set.",
  },
  macro: {
    value: "0.333",
    copy: "Macro-F1 is much lower because minority classes remain difficult, especially F and Q.",
  },
};

document.querySelectorAll("[data-metric]").forEach((button) => {
  button.addEventListener("click", () => {
    const metric = metricContent[button.dataset.metric];
    document.querySelector("#metric-value").textContent = metric.value;
    document.querySelector("#metric-copy").textContent = metric.copy;

    document.querySelectorAll("[data-metric]").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
  });
});

const matrixCopy = document.querySelector("#matrix-copy");

document.querySelectorAll("[data-cell]").forEach((button) => {
  button.addEventListener("click", () => {
    const [truth, prediction] = button.dataset.cell.split("->");
    const count = button.textContent.trim();
    const isCorrect = truth === prediction;
    matrixCopy.textContent = isCorrect
      ? `${count} test beats from class ${truth} were correctly predicted as ${prediction}.`
      : `${count} test beats from true class ${truth} were predicted as ${prediction}. This is an error pattern students should explain.`;

    document.querySelectorAll("[data-cell]").forEach((cell) => {
      cell.classList.toggle("active", cell === button);
    });
  });
});

const postureSamples = {
  neutral: {
    label: "Neutral posture",
    risk: "Low risk",
    points: [
      ["Head", 0.02, 0.92],
      ["Neck", 0.01, 0.78],
      ["First thoracic vertebra", 0.0, 0.62],
      ["First lumbar vertebra", 0.0, 0.42],
      ["Sacral vertebrae", 0.01, 0.22],
    ],
  },
  forward: {
    label: "Forward head posture",
    risk: "Moderate risk",
    points: [
      ["Head", 0.19, 0.91],
      ["Neck", 0.1, 0.77],
      ["First thoracic vertebra", 0.03, 0.62],
      ["First lumbar vertebra", 0.0, 0.42],
      ["Sacral vertebrae", 0.0, 0.22],
    ],
  },
  tilt: {
    label: "Left spinal tilt",
    risk: "Review needed",
    points: [
      ["Head", -0.1, 0.91],
      ["Neck", -0.07, 0.77],
      ["First thoracic vertebra", -0.03, 0.62],
      ["First lumbar vertebra", 0.04, 0.42],
      ["Sacral vertebrae", 0.09, 0.22],
    ],
  },
};

function renderPostureSample(sampleKey) {
  const sample = postureSamples[sampleKey];
  const svg = document.querySelector("#posture-svg");
  const table = document.querySelector("#landmark-table");
  if (!svg || !table) return;

  const toX = (value) => 160 + value * 430;
  const toY = (value) => 330 - value * 300;
  const points = sample.points.map(([name, x, y]) => ({ name, x, y, px: toX(x), py: toY(y) }));
  const path = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.px.toFixed(1)} ${point.py.toFixed(1)}`).join(" ");
  const guide = points.map((point) => `<line x1="160" y1="${point.py}" x2="${point.px}" y2="${point.py}" class="skeleton-guide" />`).join("");
  const nodes = points
    .map(
      (point, index) => `
        <g>
          <circle cx="${point.px}" cy="${point.py}" r="8" class="skeleton-node" />
          <text x="${point.px + 12}" y="${point.py + 4}">${index + 1}</text>
        </g>
      `
    )
    .join("");

  svg.innerHTML = `
    <title id="posture-svg-title">${sample.label}</title>
    <rect x="0" y="0" width="320" height="360" rx="8" class="skeleton-bg" />
    <line x1="160" y1="28" x2="160" y2="332" class="skeleton-axis" />
    ${guide}
    <path d="${path}" class="skeleton-line" />
    ${nodes}
    <text x="18" y="32" class="skeleton-label">${sample.label}</text>
    <text x="18" y="54" class="skeleton-risk">${sample.risk}</text>
  `;

  table.innerHTML = sample.points
    .map(
      ([name, x, y]) => `
        <div>
          <strong>${name}</strong>
          <span>x ${x.toFixed(2)} · y ${y.toFixed(2)}</span>
        </div>
      `
    )
    .join("");
}

const postureSelect = document.querySelector("#posture-select");
if (postureSelect) {
  postureSelect.addEventListener("change", (event) => renderPostureSample(event.target.value));
  renderPostureSample(postureSelect.value);
}

const motionEpochs = document.querySelector("#motion-epochs");
if (motionEpochs) {
  motionEpochs.addEventListener("input", () => {
    document.querySelector("#motion-epoch-label").textContent = motionEpochs.value;
  });
}

const motionCurveSets = {
  posture: [4, 5, 6, 8, 9, 10, 9, 8, 7, 6, 5, 4],
  rom: [8, 16, 28, 42, 58, 71, 82, 88, 84, 73, 55, 34],
  gait: [22, 31, 43, 36, 24, 18, 27, 41, 45, 33, 21, 17],
};

const holomotionFeatureGroups = {
  timing: {
    title: "Step Timing and Support",
    copy:
      "Timing features describe how the subject distributes time across walking phases. They can reveal cautious or unstable gait even when posture looks normal.",
    fields: ["ZV_Step_Time", "ZV_Cadence", "ZV_Single_Support", "ZV_Double_Support", "ZV_Walking_Speed"],
    question: "If double-support time increases, what might that suggest about balance confidence?",
  },
  rom: {
    title: "Joint Range of Motion",
    copy:
      "ROM features summarize how far a joint moves during a gait cycle. Limited knee or foot motion may indicate compensation, stiffness, or measurement issues.",
    fields: ["ZV_Knee_FleExt_List", "ZV_Knee_FleExt_Max", "ZV_Foot_DorsiPlant_List", "ZV_Foot_Dorsi_Max", "ZV_Foot_Plant_Max"],
    question: "Why is a full angle curve more informative than only one maximum value?",
  },
  stability: {
    title: "Trunk and Pelvis Stability",
    copy:
      "Pelvis and trunk features connect lower-limb motion to whole-body control. They help students reason beyond simple fall/no-fall labels.",
    fields: ["ZV_Pelvic_Tilt_List", "ZV_Pelvic_Rot_List", "ZV_Trunk_Tilt_List", "ZV_Trunk_AntPos_List"],
    question: "What validation would be needed before calling trunk tilt a clinical risk marker?",
  },
  asymmetry: {
    title: "Left-Right Asymmetry",
    copy:
      "Asymmetry features compare left and right gait cycles. They can show uneven movement patterns that average walking speed may hide.",
    fields: ["Step length difference", "Stride length difference", "Step time difference", "Knee ROM difference", "Foot ROM difference"],
    question: "Why can two people with the same walking speed have different mobility-risk profiles?",
  },
};

function renderHolomotionFeatureGroup(groupKey) {
  const card = document.querySelector("#feature-card");
  const group = holomotionFeatureGroups[groupKey];
  if (!card || !group) return;

  card.innerHTML = `
    <strong>${group.title}</strong>
    <p>${group.copy}</p>
    <ul>
      ${group.fields.map((field) => `<li>${field}</li>`).join("")}
    </ul>
    <div class="feedback-box">
      <strong>Student question</strong>
      <p>${group.question}</p>
    </div>
  `;
}

document.querySelectorAll("[data-feature-group]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-feature-group]").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    renderHolomotionFeatureGroup(button.dataset.featureGroup);
  });
});

renderHolomotionFeatureGroup("timing");

const motionCaseFeedback = {
  a: "Not the best choice. Case A has slow walking speed, but the other features are mostly stable. A single speed number is not enough to localize risk.",
  b: "Good choice. Case B has similar walking speed but stronger risk evidence: asymmetry, reduced knee ROM, and trunk sway. This is why Holomotion-style features add depth beyond a single time or speed score.",
  same: "Not quite. Similar walking speed does not mean similar risk. Side-to-side timing, ROM, and trunk/pelvis control can create different risk profiles.",
};

document.querySelectorAll("[data-case-answer]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-case-answer]").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    const feedback = document.querySelector("#motion-case-feedback");
    if (feedback) {
      feedback.textContent = motionCaseFeedback[button.dataset.caseAnswer];
      feedback.classList.add(button.dataset.caseAnswer === "b" ? "correct" : "incorrect");
      feedback.classList.toggle("correct", button.dataset.caseAnswer === "b");
      feedback.classList.toggle("incorrect", button.dataset.caseAnswer !== "b");
    }
  });
});

const riskBuilderButton = document.querySelector("#build-motion-risk");
if (riskBuilderButton) {
  riskBuilderButton.addEventListener("click", () => {
    const selected = Array.from(document.querySelectorAll("[data-risk-feature]:checked")).map((input) => input.value);
    const output = document.querySelector("#motion-risk-output");
    if (!output) return;
    if (!selected.length) {
      output.textContent = "Select at least one Holomotion finding before generating an explanation.";
      output.classList.add("warning");
      return;
    }
    const validationFlag = selected.some((feature) => feature.includes("missing"));
    const evidence = selected.filter((feature) => !feature.includes("missing"));
    const evidenceText = evidence.length ? evidence.join(", ") : "the available gait features";
    output.classList.remove("warning");
    output.innerHTML = `
      <strong>Draft interpretation</strong>
      <p>This subject may need closer mobility review because the feature profile shows ${evidenceText}. These findings should be treated as screening evidence, not a diagnosis.</p>
      <p>${validationFlag ? "Because repeated trials or clinical labels are missing, the result should be marked uncertain and routed to validation." : "Next, compare this interpretation with repeated trials, clinician assessment, or follow-up outcomes before making a stronger claim."}</p>
    `;
  });
}

const validationChoice = document.querySelector("#validation-choice");
const validationFeedback = {
  repeat: "This checks measurement stability: do the same features appear when the same person repeats the test?",
  clinical: "This checks clinical meaning: do Holomotion features agree with a human assessment or established mobility score?",
  followup: "This checks predictive validity: do today's gait features relate to future fall events or mobility decline?",
  domain: "This checks domain gap: does a model trained on public data behave sensibly on real-device feature outputs?",
};
if (validationChoice) {
  validationChoice.addEventListener("change", () => {
    const feedback = document.querySelector("#validation-feedback");
    if (feedback) feedback.textContent = validationFeedback[validationChoice.value];
  });
}

const motionProjectPlanButton = document.querySelector("#build-motion-project-plan");
if (motionProjectPlanButton) {
  motionProjectPlanButton.addEventListener("click", () => {
    const target = document.querySelector("#motion-design-target")?.value || "fall-risk screening";
    const input = document.querySelector("#motion-design-input")?.value || "public gait features";
    const model = document.querySelector("#motion-design-model")?.value || "a baseline classifier";
    const validation = document.querySelector("#motion-design-validation")?.value || "repeated trials";
    const output = document.querySelector("#motion-project-plan-output");
    if (!output) return;

    output.classList.remove("warning");
    output.innerHTML = `
      <strong>Draft mini-project plan</strong>
      <p><b>Question:</b> Can ${input} support ${target}, and where does the evidence become uncertain?</p>
      <p><b>Experiment:</b> Train ${model}, report one test metric table, then inspect at least two individual cases instead of only reporting average accuracy.</p>
      <p><b>Holomotion audit:</b> Compare the model output with device-style features such as speed, step timing, joint ROM, trunk/pelvis stability, and asymmetry. Mark any disagreement as a domain-gap or validation question.</p>
      <p><b>Validation:</b> Use ${validation} as the next evidence source before claiming clinical usefulness.</p>
    `;
  });
}

const resistanceProjectPlanButton = document.querySelector("#build-resistance-project-plan");
if (resistanceProjectPlanButton) {
  resistanceProjectPlanButton.addEventListener("click", () => {
    const arraySize =
      document.querySelector("#resistance-design-size")?.value || "3x3 baseline";
    const formulation =
      document.querySelector("#resistance-design-formulation")?.value || "physics-informed conductance regression";
    const failure =
      document.querySelector("#resistance-design-failure")?.value || "underestimated high-resistance cell";
    const improvement =
      document.querySelector("#resistance-design-improvement")?.value || "adding a forward-model consistency loss";
    const validation =
      document.querySelector("#resistance-design-validation")?.value || "a physical resistor-array test";
    const output = document.querySelector("#resistance-project-plan-output");
    if (!output) return;

    output.classList.remove("warning");
    output.innerHTML = `
      <strong>Draft inverse-sensor project plan</strong>
      <p><b>Question:</b> Can ${formulation} reconstruct a hidden resistance map for the ${arraySize} from limited current measurements?</p>
      <p><b>Audit target:</b> Focus on ${failure}. Include one true map, predicted map, and absolute-error map instead of reporting only average accuracy.</p>
      <p><b>Scaling task:</b> If moving beyond 3x3, update the output dimension to n*n cells, regenerate or load matching current data, and check whether extra row, column, diagonal, or path measurements are needed.</p>
      <p><b>Improvement:</b> Test ${improvement}, then compare MAE, high-resistance recall, and exact-map accuracy against the baseline.</p>
      <p><b>Validation:</b> Use ${validation} before claiming the predicted resistance map reflects real cell growth, local coverage, or device behavior.</p>
    `;
  });
}

function drawMotionCurve(task) {
  const svg = document.querySelector("#motion-curve");
  if (!svg) return;
  const values = motionCurveSets[task] || motionCurveSets.posture;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = Math.max(max - min, 1);
  const coords = values.map((value, index) => {
    const x = 34 + (index / (values.length - 1)) * 352;
    const y = 178 - ((value - min) / range) * 128;
    return { x, y, value };
  });
  const path = coords.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" ");
  const nodes = coords
    .map((point) => `<circle cx="${point.x}" cy="${point.y}" r="4.5" class="curve-node"><title>${point.value} degrees</title></circle>`)
    .join("");

  svg.innerHTML = `
    <title id="motion-curve-title">Simulated ${task} joint angle sequence</title>
    <rect x="0" y="0" width="420" height="220" rx="8" class="curve-bg" />
    <line x1="34" y1="178" x2="386" y2="178" class="curve-axis" />
    <line x1="34" y1="38" x2="34" y2="178" class="curve-axis" />
    <path d="${path}" class="curve-line" />
    ${nodes}
    <text x="34" y="28" class="curve-label">Joint angle over time</text>
    <text x="310" y="202" class="curve-label">frames</text>
  `;
}

const motionButton = document.querySelector("#run-motion-sim");
if (motionButton) {
  motionButton.addEventListener("click", () => {
    const task = document.querySelector('input[name="motion-task"]:checked').value;
    const model = document.querySelector('input[name="motion-model"]:checked').value;
    const epochs = Number(document.querySelector("#motion-epochs").value);
    runMotionPrototype({ task, model, epochs });
  });
}

function runMotionPrototype({ task, model, epochs }) {
  const meter = document.querySelector("#motion-meter");
  const copy = document.querySelector("#motion-training-copy");
  meter.style.width = "0";
  copy.textContent = `Preparing simulated ${task} features for a ${model === "rnn" ? "sequence" : "skeleton graph"} model...`;

  window.setTimeout(() => {
    meter.style.width = "42%";
    copy.textContent =
      model === "rnn"
        ? "The model reads ordered frames, so timing and movement shape matter."
        : "The model reads body landmarks as connected nodes, so anatomical structure matters.";
  }, 450);

  window.setTimeout(() => {
    meter.style.width = "76%";
    copy.textContent = "Validation checks whether the model recognizes abnormal movement without overclaiming diagnosis.";
  }, 1000);

  window.setTimeout(() => {
    meter.style.width = "100%";
    copy.textContent = `${epochs}-epoch prototype run complete. Open the Test step to inspect simulated metrics.`;
    labState.motion = { task, model, epochs };
    updateMotionTest();
  }, 1550);
}

function updateMotionTest() {
  if (!labState.motion) return;
  const { task, model, epochs } = labState.motion;
  const modelLabel = model === "rnn" ? "RNN/LSTM" : "Skeleton graph";
  const taskLabel = {
    posture: "static posture risk",
    rom: "joint range of motion",
    gait: "gait sequence",
  }[task];
  const metricSeed = task === "posture" ? 0 : task === "rom" ? 1 : 2;
  const modelBoost = model === "graph" && task === "posture" ? 0.04 : model === "rnn" && task !== "posture" ? 0.05 : 0;
  const accuracy = Math.min(0.91, 0.76 + epochs * 0.004 + metricSeed * 0.015 + modelBoost);
  const recall = Math.min(0.88, 0.65 + epochs * 0.005 + metricSeed * 0.02 + modelBoost);
  const error = Math.max(3.2, 9.8 - epochs * 0.13 - modelBoost * 20);

  document.querySelector("#motion-test-run-pill").textContent = `${modelLabel} · ${epochs} epochs`;
  document.querySelector("#motion-test-run-pill").classList.add("ready");
  document.querySelector("#motion-run-summary").innerHTML = `
    <strong>Your simulated ${taskLabel} run</strong>
    <p>Model: ${modelLabel}. This prototype uses generated coordinates and angle curves until real device exports are available.</p>
  `;
  document.querySelector("#motion-accuracy").textContent = accuracy.toFixed(2);
  document.querySelector("#motion-recall").textContent = recall.toFixed(2);
  document.querySelector("#motion-error").textContent = `${error.toFixed(1)}°`;
  document.querySelector("#motion-curve-copy").textContent =
    task === "posture"
      ? "A static posture task can still be represented as landmark offsets, but it has less temporal information than gait or ROM."
      : "This sequence is the kind of frame-by-frame signal an RNN can use to learn movement patterns.";
  drawMotionCurve(task);
}

const resistanceBaseMaps = {
  2: [
    [1.2, 2.4],
    [1.7, 3.1],
  ],
  3: [
    [2, 2, 2],
    [2, 2, 2],
    [110, 2, 2],
  ],
  4: [
    [1.0, 1.5, 2.1, 2.8],
    [1.4, 2.6, 3.8, 2.5],
    [1.9, 2.4, 4.5, 3.2],
    [2.3, 3.0, 3.6, 4.2],
  ],
};

let activeResistanceSize = 3;

const resistanceRealMetrics = {
  mae: 4.86,
  rmse: 14.61,
  cellAccuracy: 0.976,
  highRecall: 0.879,
  highPrecision: 1.0,
  exactMapAccuracy: 0.84,
};

function cloneResistanceMap(size) {
  return resistanceBaseMaps[size].map((row) => row.slice());
}

function summarizeResistanceMap(map) {
  const rowTotals = map.map((row) => row.reduce((sum, value) => sum + value, 0));
  const colTotals = map[0].map((_, colIndex) => map.reduce((sum, row) => sum + row[colIndex], 0));
  const diagonal = map.reduce((sum, row, index) => sum + row[index], 0);
  const antiDiagonal = map.reduce((sum, row, index) => sum + row[row.length - 1 - index], 0);
  return { rowTotals, colTotals, diagonal, antiDiagonal };
}

function renderResistanceGrid(container, map) {
  if (!container) return;
  if (!map) {
    container.innerHTML = '<div class="placeholder-cell">Run model</div>';
    container.style.setProperty("--array-size", 1);
    return;
  }
  const values = map.flat();
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(max - min, 0.1);
  container.style.setProperty("--array-size", map.length);
  container.innerHTML = map
    .flatMap((row, rowIndex) =>
      row.map((value, colIndex) => {
        const intensity = (value - min) / range;
        const alpha = 0.18 + intensity * 0.72;
        return `
          <div class="resistance-cell" style="--cell-alpha:${alpha.toFixed(2)}">
            <strong>${value.toFixed(1)}</strong>
            <span>R${rowIndex + 1}C${colIndex + 1}</span>
          </div>
        `;
      })
    )
    .join("");
}

function renderResistanceMeasurements(size) {
  const map = cloneResistanceMap(size);
  const table = document.querySelector("#measurement-table");
  if (!table) return;
  const { rowTotals, colTotals, diagonal, antiDiagonal } = summarizeResistanceMap(map);
  const rows = [
    ...rowTotals.map((value, index) => [`Row ${index + 1} total`, value]),
    ...colTotals.map((value, index) => [`Column ${index + 1} total`, value]),
    ["Main diagonal path", diagonal],
    ["Anti-diagonal path", antiDiagonal],
  ];
  table.innerHTML = rows
    .map(
      ([label, value]) => `
        <div>
          <span>${label}</span>
          <strong>${value.toFixed(2)} kOhm</strong>
        </div>
      `
    )
    .join("");
}

function setResistanceSize(size) {
  activeResistanceSize = Number(size);
  const map = cloneResistanceMap(activeResistanceSize);
  renderResistanceGrid(document.querySelector("#resistance-grid"), map);
  renderResistanceMeasurements(activeResistanceSize);
  renderResistanceGrid(document.querySelector("#true-resistance-grid"), map);
  renderResistanceGrid(document.querySelector("#predicted-resistance-grid"), null);
  document.querySelectorAll("[data-array-size]").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.arraySize) === activeResistanceSize);
  });
}

document.querySelectorAll("[data-array-size]").forEach((button) => {
  button.addEventListener("click", () => setResistanceSize(button.dataset.arraySize));
});

setResistanceSize(activeResistanceSize);

const resistanceNoise = document.querySelector("#resistance-noise");
if (resistanceNoise) {
  resistanceNoise.addEventListener("input", () => {
    document.querySelector("#resistance-noise-label").textContent = `${resistanceNoise.value}%`;
  });
}

const resistanceSamples = document.querySelector("#resistance-samples");
if (resistanceSamples) {
  resistanceSamples.addEventListener("input", () => {
    document.querySelector("#resistance-samples-label").textContent = resistanceSamples.value;
  });
}

const resistanceButton = document.querySelector("#run-resistance-sim");
if (resistanceButton) {
  resistanceButton.addEventListener("click", () => {
    const model = document.querySelector('input[name="resistance-model"]:checked').value;
    const noise = Number(document.querySelector("#resistance-noise")?.value || 0);
    const samples = Number(document.querySelector("#resistance-samples")?.value || 10000);
    runResistancePrototype({ size: activeResistanceSize, model, noise, samples });
  });
}

function runResistancePrototype({ size, model, noise, samples }) {
  const meter = document.querySelector("#resistance-meter");
  const copy = document.querySelector("#resistance-training-copy");
  meter.style.width = "0";
  copy.textContent = "Loading the trained MLP conductance-regression result from the held-out test set...";

  window.setTimeout(() => {
    meter.style.width = "38%";
    copy.textContent = "Input: 9 current measurements. Output: 9 predicted resistance values reshaped into a 3x3 map.";
  }, 450);

  window.setTimeout(() => {
    meter.style.width = "72%";
    copy.textContent = "The best current setting uses conductance transformation because current is physically closer to 1/R.";
  }, 1000);

  window.setTimeout(() => {
    meter.style.width = "100%";
    copy.textContent = "Trained result loaded. Open the Test step to inspect reconstruction metrics and error heatmaps.";
    labState.resistance = { size, model, noise, samples };
    updateResistanceTest();
  }, 1550);
}

function makePredictedResistanceMap(map, model, noise, samples) {
  const modelPenalty = model === "linear" ? 0.28 : model === "mlp" ? 0.16 : 0.1;
  const noisePenalty = noise / 100;
  const sampleBoost = Math.min(samples / 2000, 1) * 0.08;
  const errorScale = Math.max(0.04, modelPenalty + noisePenalty - sampleBoost);
  return map.map((row, rowIndex) =>
    row.map((value, colIndex) => {
      const direction = (rowIndex + colIndex) % 2 === 0 ? 1 : -1;
      const offset = direction * errorScale * (0.8 + rowIndex * 0.15 + colIndex * 0.1);
      return Math.max(0.4, value + offset);
    })
  );
}

function updateResistanceTest() {
  if (!labState.resistance) return;
  const trueMap = cloneResistanceMap(3);
  const predictedMap = [
    [2, 2, 2],
    [2, 2, 2],
    [100, 2, 2],
  ];

  document.querySelector("#resistance-test-run-pill").textContent = "MLP + conductance target";
  document.querySelector("#resistance-test-run-pill").classList.add("ready");
  document.querySelector("#resistance-run-summary").innerHTML = `
    <strong>Held-out test result</strong>
    <p>Input: 9 measured current values. Output: 3x3 local resistance map. The model predicts concrete resistance values, then we evaluate high/low localization.</p>
  `;
  document.querySelector("#resistance-mae").textContent = `${resistanceRealMetrics.mae.toFixed(2)} ohm`;
  document.querySelector("#resistance-pattern").textContent = resistanceRealMetrics.highRecall.toFixed(3);
  document.querySelector("#resistance-uncertainty").textContent = resistanceRealMetrics.exactMapAccuracy.toFixed(3);
  document.querySelector("#resistance-test-copy").textContent =
    "The model usually finds high-resistance cells with high precision, but some high cells are still underestimated when several high-resistance regions appear in the same map.";
  renderResistanceGrid(document.querySelector("#true-resistance-grid"), trueMap);
  renderResistanceGrid(document.querySelector("#predicted-resistance-grid"), predictedMap);
}

function syncNotes() {
  const wfmNote = document.querySelector("#wfm-reflection").value.trim();
  const ecgNote = document.querySelector("#ecg-reflection").value.trim();
  const motionNote = document.querySelector("#motion-reflection")?.value.trim() || "";
  const resistanceNote = document.querySelector("#resistance-reflection")?.value.trim() || "";
  document.querySelector("#saved-wfm-note").textContent = wfmNote || "No note yet. Write one in Module 1 Reflect.";
  document.querySelector("#saved-ecg-note").textContent = ecgNote || "No note yet. Write one in Module 2 Reflect.";
  document.querySelector("#saved-motion-note").textContent = motionNote || "No note yet. Write one in Module 3 Reflect.";
  document.querySelector("#saved-resistance-note").textContent =
    resistanceNote || "No note yet. Write one in Module 4 Reflect.";
}

document.querySelector("#wfm-reflection").addEventListener("input", syncNotes);
document.querySelector("#ecg-reflection").addEventListener("input", syncNotes);
document.querySelector("#motion-reflection")?.addEventListener("input", syncNotes);
document.querySelector("#resistance-reflection")?.addEventListener("input", syncNotes);
