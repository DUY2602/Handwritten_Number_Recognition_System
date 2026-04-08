const elements = {
    dropZone: document.getElementById("drop-zone"),
    fileInput: document.getElementById("file-input"),
    previewContainer: document.getElementById("preview-container"),
    imagePreview: document.getElementById("image-preview"),
    fileName: document.getElementById("file-name"),
    fileSize: document.getElementById("file-size"),
    replaceBtn: document.getElementById("replace-btn"),
    clearBtn: document.getElementById("clear-btn"),
    analyzeBtn: document.getElementById("analyze-btn"),
    analyzeCanvasBtn: document.getElementById("analyze-canvas"),
    clearCanvasBtn: document.getElementById("clear-canvas"),
    canvas: document.getElementById("drawing-canvas"),
    tabs: Array.from(document.querySelectorAll(".tab")),
    uploadPanel: document.getElementById("upload-panel"),
    drawPanel: document.getElementById("draw-panel"),
    emptyState: document.getElementById("empty-state"),
    loading: document.getElementById("loading"),
    resultContent: document.getElementById("result-content"),
    expression: document.getElementById("res-expr"),
    answer: document.getElementById("res-ans"),
    lineResults: document.getElementById("line-results"),
    error: document.getElementById("res-error"),
    note: document.getElementById("res-note"),
    bboxImage: document.getElementById("bbox-img"),
    predictionSummary: document.getElementById("prediction-summary"),
    feedbackSummary: document.getElementById("feedback-summary"),
    characterStrip: document.getElementById("character-strip"),
    feedbackEditor: document.getElementById("feedback-editor"),
    feedbackPreview: document.getElementById("feedback-preview"),
    feedbackSelectedChar: document.getElementById("feedback-selected-char"),
    feedbackSelectedMeta: document.getElementById("feedback-selected-meta"),
    feedbackTopK: document.getElementById("feedback-topk"),
    feedbackInput: document.getElementById("feedback-input"),
    classPicker: document.getElementById("class-picker"),
    applyCorrectionBtn: document.getElementById("apply-correction"),
    removeSegmentBtn: document.getElementById("remove-segment"),
    resetCorrectionBtn: document.getElementById("reset-correction"),
    submitFeedbackBtn: document.getElementById("submit-feedback"),
    feedbackError: document.getElementById("feedback-error"),
    feedbackNote: document.getElementById("feedback-note"),
};

const state = {
    mode: "upload",
    selectedFile: null,
    sourcePreviewUrl: "",
    drawing: false,
    lastX: 0,
    lastY: 0,
    analysisId: "",
    characters: [],
    selectedCharacterId: "",
};

const CORRECTABLE_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "(", ")"];
const CORRECTION_ALIASES = {
    x: "*",
    X: "*",
    "\u00d7": "*",
    "\u00f7": "/",
    ":": "/",
};

const canvasContext = elements.canvas.getContext("2d");
const CANVAS_BRUSH_SIZE = 12;

function setDefaultResultFields() {
    elements.expression.textContent = "N/A";
    elements.answer.textContent = "N/A";
    elements.lineResults.innerHTML = "";
    elements.lineResults.classList.add("hidden");
    elements.bboxImage.src = "";
    elements.predictionSummary.textContent = "No data yet";
    resetFeedbackState();
}

function resetFeedbackState() {
    state.analysisId = "";
    state.characters = [];
    state.selectedCharacterId = "";
    elements.feedbackSummary.textContent = "No characters loaded";
    elements.characterStrip.innerHTML = "";
    elements.feedbackEditor.classList.add("hidden");
    elements.feedbackPreview.src = "";
    elements.feedbackSelectedChar.textContent = "-";
    elements.feedbackSelectedMeta.textContent = "Select a segmented character to correct it.";
    elements.feedbackTopK.textContent = "Model top-k will appear here.";
    elements.feedbackInput.value = "";
    elements.feedbackError.textContent = "";
    elements.feedbackError.classList.add("hidden");
    elements.feedbackNote.textContent = "";
    elements.feedbackNote.classList.add("hidden");
}

function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) {
        return "-";
    }

    const units = ["B", "KB", "MB"];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex += 1;
    }

    return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function showState(name) {
    elements.emptyState.classList.toggle("hidden", name !== "empty");
    elements.loading.classList.toggle("hidden", name !== "loading");
    elements.resultContent.classList.toggle("hidden", name !== "result");
}

function resetMessages() {
    elements.error.textContent = "";
    elements.error.classList.add("hidden");
    elements.note.textContent = "";
    elements.note.classList.add("hidden");
}

function resetFeedbackMessages() {
    elements.feedbackError.textContent = "";
    elements.feedbackError.classList.add("hidden");
    elements.feedbackNote.textContent = "";
    elements.feedbackNote.classList.add("hidden");
}

function resetResults() {
    showState("empty");
    resetMessages();
    setDefaultResultFields();
}

function showResultError(message) {
    resetMessages();
    setDefaultResultFields();
    showState("result");
    elements.error.textContent = message;
    elements.error.classList.remove("hidden");
}

function setMode(mode) {
    state.mode = mode;
    elements.tabs.forEach((tab) => {
        tab.classList.toggle("active", tab.dataset.mode === mode);
    });
    elements.uploadPanel.classList.toggle("hidden", mode !== "upload");
    elements.drawPanel.classList.toggle("hidden", mode !== "draw");
}

function updatePreview(file, previewUrl) {
    state.selectedFile = file;
    state.sourcePreviewUrl = previewUrl;
    elements.imagePreview.src = previewUrl;
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = formatBytes(file.size);
    elements.dropZone.classList.add("hidden");
    elements.previewContainer.classList.remove("hidden");
}

function clearSelectedFile() {
    state.selectedFile = null;
    state.sourcePreviewUrl = "";
    elements.fileInput.value = "";
    elements.imagePreview.src = "";
    elements.fileName.textContent = "No image selected";
    elements.fileSize.textContent = "-";
    elements.previewContainer.classList.add("hidden");
    elements.dropZone.classList.remove("hidden");
    resetResults();
}

function clearCanvas() {
    canvasContext.fillStyle = "#ffffff";
    canvasContext.fillRect(0, 0, elements.canvas.width, elements.canvas.height);
    canvasContext.beginPath();
}

function getCanvasPoint(event) {
    const rect = elements.canvas.getBoundingClientRect();
    const scaleX = elements.canvas.width / rect.width;
    const scaleY = elements.canvas.height / rect.height;
    return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY,
    };
}

function startDrawing(event) {
    state.drawing = true;
    const point = getCanvasPoint(event);
    state.lastX = point.x;
    state.lastY = point.y;
}

function draw(event) {
    if (!state.drawing) {
        return;
    }

    event.preventDefault();
    const point = getCanvasPoint(event);
    canvasContext.strokeStyle = "#111111";
    canvasContext.lineWidth = CANVAS_BRUSH_SIZE;
    canvasContext.lineCap = "round";
    canvasContext.lineJoin = "round";
    canvasContext.beginPath();
    canvasContext.moveTo(state.lastX, state.lastY);
    canvasContext.lineTo(point.x, point.y);
    canvasContext.stroke();
    state.lastX = point.x;
    state.lastY = point.y;
}

function stopDrawing() {
    state.drawing = false;
    canvasContext.beginPath();
}

function buildPredictionSummary(characters = []) {
    if (!characters.length) {
        return "No confidence data";
    }

    const average = Math.round(
        (characters.reduce((sum, item) => sum + Number(item.conf || 0), 0) / characters.length) * 100
    );
    const lowConfidence = characters.filter((item) => Number(item.conf || 0) < 0.8).length;

    if (lowConfidence > 0) {
        return `${characters.length} chars - avg ${average}% - ${lowConfidence} low-confidence`;
    }

    return `${characters.length} chars - avg ${average}%`;
}

function buildTopKSummary(entries = []) {
    if (!Array.isArray(entries) || !entries.length) {
        return "unavailable";
    }

    return entries
        .slice(0, 5)
        .map((entry) => `${String(entry.char || "?")} ${Math.round(Number(entry.conf || 0) * 100)}%`)
        .join(" · ");
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function normalizeCorrectionLabel(value) {
    const raw = String(value ?? "").trim();
    if (!raw) {
        return "";
    }
    return CORRECTION_ALIASES[raw] || raw;
}

function getSelectedCharacter() {
    return state.characters.find((item) => item.id === state.selectedCharacterId) || null;
}

function getCharacterDisplayValue(item) {
    if (item.rejected || item.saved_rejection) {
        return "Removed";
    }
    return item.corrected_char || item.saved_correction || item.char || "?";
}

function getPendingCorrections() {
    return state.characters.filter((item) => item.corrected_char && item.corrected_char !== item.char);
}

function getPendingRejections() {
    return state.characters.filter((item) => item.rejected);
}

function setFeedbackMessage(type, message) {
    const target = type === "error" ? elements.feedbackError : elements.feedbackNote;
    const other = type === "error" ? elements.feedbackNote : elements.feedbackError;

    other.textContent = "";
    other.classList.add("hidden");
    target.textContent = message;
    target.classList.toggle("hidden", !message);
}

function updateFeedbackSummary() {
    if (!state.characters.length) {
        elements.feedbackSummary.textContent = "No characters loaded";
        return;
    }

    const pending = getPendingCorrections().length;
    const pendingRemovals = getPendingRejections().length;
    const saved = state.characters.filter((item) => item.saved_correction && !item.corrected_char).length;
    const savedRemovals = state.characters.filter((item) => item.saved_rejection && !item.rejected).length;
    const parts = [`${state.characters.length} characters`];

    if (pending > 0) {
        parts.push(`${pending} pending`);
    }
    if (pendingRemovals > 0) {
        parts.push(`${pendingRemovals} pending remove`);
    }
    if (saved > 0) {
        parts.push(`${saved} saved`);
    }
    if (savedRemovals > 0) {
        parts.push(`${savedRemovals} removed`);
    }
    if (pending === 0 && pendingRemovals === 0 && saved === 0 && savedRemovals === 0) {
        parts.push("click one to correct it");
    }

    elements.feedbackSummary.textContent = parts.join(" - ");
}

function renderClassPicker() {
    const selected = getSelectedCharacter();
    const selectedValue = selected && !selected.rejected
        ? normalizeCorrectionLabel(selected.corrected_char || selected.saved_correction || selected.char)
        : "";

    elements.classPicker.innerHTML = CORRECTABLE_LABELS.map((label) => {
        const activeClass = label === selectedValue ? " active" : "";
        return `<button class="picker-btn${activeClass}" type="button" data-pick="${escapeHtml(label)}">${escapeHtml(label)}</button>`;
    }).join("");

    elements.classPicker.querySelectorAll("[data-pick]").forEach((button) => {
        button.addEventListener("click", () => {
            elements.feedbackInput.value = button.dataset.pick || "";
            applySelectedCorrection();
        });
    });
}

function renderFeedbackEditor() {
    const selected = getSelectedCharacter();
    if (!selected) {
        elements.feedbackEditor.classList.add("hidden");
        return;
    }

    elements.feedbackEditor.classList.remove("hidden");
    elements.feedbackPreview.src = selected.roi_image || "";
    elements.feedbackSelectedChar.textContent = `Predicted: ${selected.char || "?"}`;
    const status = selected.rejected
        ? "Pending removal"
        : selected.saved_rejection
            ? "Saved as removed"
            : selected.corrected_char
                ? `Pending label -> ${selected.corrected_char}`
                : selected.saved_correction
                    ? `Saved label -> ${selected.saved_correction}`
                    : "Ready";
    elements.feedbackSelectedMeta.textContent = `Line ${(Number(selected.line_index) || 0) + 1} - confidence ${Math.round(Number(selected.conf || 0) * 100)}% - ${status}`;
    elements.feedbackTopK.textContent = `Raw model top-k: ${buildTopKSummary(selected.top_k || [])}`;
    elements.feedbackInput.value = selected.rejected ? "" : (selected.corrected_char || selected.saved_correction || selected.char || "");
    renderClassPicker();
}

function renderCharacterReview(characters = [], analysisId = "") {
    state.analysisId = analysisId || "";
    state.selectedCharacterId = "";
    state.characters = Array.isArray(characters)
        ? characters.map((item) => ({
            ...item,
            corrected_char: "",
            saved_correction: "",
            rejected: false,
            saved_rejection: false,
        }))
        : [];

    renderCharacterStrip();
}

function renderCharacterStrip() {
    resetFeedbackMessages();

    if (!state.characters.length) {
        elements.characterStrip.innerHTML = '<p class="review-empty">No segmented characters are available for correction.</p>';
        elements.feedbackEditor.classList.add("hidden");
        updateFeedbackSummary();
        return;
    }

    elements.characterStrip.innerHTML = state.characters.map((item, index) => {
        const activeClass = item.id === state.selectedCharacterId ? " active" : "";
        const correctedClass = item.rejected || item.saved_rejection
            ? " rejected"
            : getCharacterDisplayValue(item) !== item.char
                ? " corrected"
                : "";
        const displayValue = getCharacterDisplayValue(item);
        const meta = item.rejected
            ? "Pending remove"
            : item.saved_rejection
                ? "Removed"
                : item.corrected_char
                    ? `Pending -> ${escapeHtml(displayValue)}`
                    : item.saved_correction
                        ? `Saved -> ${escapeHtml(displayValue)}`
                        : `${Math.round(Number(item.conf || 0) * 100)}%`;
        const topKTitle = `Raw model top-k: ${buildTopKSummary(item.top_k || [])}`;

        return `
            <button class="char-card${activeClass}${correctedClass}" type="button" data-char-id="${escapeHtml(item.id)}" title="${escapeHtml(topKTitle)}">
                <span class="char-card-index">#${index + 1} · L${(Number(item.line_index) || 0) + 1}</span>
                <img src="${item.roi_image || ""}" alt="Segmented character ${index + 1}">
                <strong>${escapeHtml(displayValue)}</strong>
                <small>${meta}</small>
            </button>
        `;
    }).join("");

    elements.characterStrip.querySelectorAll("[data-char-id]").forEach((button) => {
        button.addEventListener("click", () => {
            state.selectedCharacterId = button.dataset.charId || "";
            renderCharacterStrip();
            renderFeedbackEditor();
        });
    });

    updateFeedbackSummary();
    renderFeedbackEditor();
}

function applySelectedCorrection() {
    const selected = getSelectedCharacter();
    if (!selected) {
        setFeedbackMessage("error", "Select a character first.");
        return;
    }

    const normalized = normalizeCorrectionLabel(elements.feedbackInput.value);
    if (!CORRECTABLE_LABELS.includes(normalized)) {
        setFeedbackMessage("error", "Use only digits or +-*/().");
        return;
    }

    selected.corrected_char = normalized;
    selected.rejected = false;
    selected.saved_rejection = false;
    selected.saved_correction = "";
    elements.feedbackInput.value = normalized;
    renderCharacterStrip();
    setFeedbackMessage("note", `Correction staged for ${selected.id}. Press Save Feedback when you're done.`);
}

function rejectSelectedCharacter() {
    const selected = getSelectedCharacter();
    if (!selected) {
        setFeedbackMessage("error", "Select a character first.");
        return;
    }

    selected.corrected_char = "";
    selected.saved_correction = "";
    selected.rejected = true;
    selected.saved_rejection = false;
    elements.feedbackInput.value = "";
    renderCharacterStrip();
    setFeedbackMessage("note", `Segment ${selected.id} will be removed from the retraining dataset.`);
}

function resetSelectedCorrection() {
    const selected = getSelectedCharacter();
    if (!selected) {
        setFeedbackMessage("error", "Select a character first.");
        return;
    }

    selected.corrected_char = "";
    selected.saved_correction = "";
    selected.rejected = false;
    selected.saved_rejection = false;
    elements.feedbackInput.value = selected.char || "";
    renderCharacterStrip();
    setFeedbackMessage("note", `Reset ${selected.id} back to the model prediction.`);
}

async function submitFeedback() {
    resetFeedbackMessages();

    if (!state.analysisId) {
        setFeedbackMessage("error", "Analyze an image first before saving feedback.");
        return;
    }

    const corrections = getPendingCorrections().map((item) => ({
        character_id: item.id,
        corrected_char: item.corrected_char,
    }));
    const rejections = getPendingRejections().map((item) => ({
        character_id: item.id,
    }));

    if (!corrections.length && !rejections.length) {
        setFeedbackMessage("note", "No changed characters to save yet.");
        return;
    }

    elements.submitFeedbackBtn.disabled = true;
    try {
        const response = await fetch("/api/feedback", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                analysis_id: state.analysisId,
                corrections,
                rejections,
            }),
        });
        const data = await parseApiResponse(response);

        if (!response.ok || data.error) {
            setFeedbackMessage("error", data.error || "Could not save feedback.");
            return;
        }

        const correctionMap = new Map(corrections.map((item) => [item.character_id, item.corrected_char]));
        const rejectionIds = new Set(rejections.map((item) => item.character_id));
        state.characters = state.characters.map((item) => {
            const savedValue = correctionMap.get(item.id);
            if (savedValue) {
                return {
                    ...item,
                    corrected_char: "",
                    saved_correction: savedValue,
                    rejected: false,
                    saved_rejection: false,
                };
            }

            if (rejectionIds.has(item.id)) {
                return {
                    ...item,
                    corrected_char: "",
                    saved_correction: "",
                    rejected: false,
                    saved_rejection: true,
                };
            }

            return item;
        });

        renderCharacterStrip();
        setFeedbackMessage("note", data.message || "Feedback saved.");
    } catch (error) {
        setFeedbackMessage("error", "Could not reach the feedback endpoint.");
    } finally {
        elements.submitFeedbackBtn.disabled = false;
    }
}

function renderLineResults(lines = []) {
    if (!Array.isArray(lines) || lines.length <= 1) {
        elements.lineResults.innerHTML = "";
        elements.lineResults.classList.add("hidden");
        return;
    }

    elements.lineResults.innerHTML = lines.map((line, index) => {
        const expression = escapeHtml(line.expression || "N/A");
        const result = escapeHtml(line.result || "N/A");
        const confidence = escapeHtml(buildPredictionSummary(line.characters || []));
        const errorBlock = line.error
            ? `<div class="line-card-error">${escapeHtml(line.error)}</div>`
            : "";

        return `
            <article class="line-card">
                <div class="line-card-head">
                    <strong>Line ${index + 1}</strong>
                    <span>${confidence}</span>
                </div>
                <div class="line-card-row">
                    <span>Expression</span>
                    <strong>${expression}</strong>
                </div>
                <div class="line-card-row">
                    <span>Result</span>
                    <strong>${result}</strong>
                </div>
                ${errorBlock}
            </article>
        `;
    }).join("");

    elements.lineResults.classList.remove("hidden");
}

function exportTrimmedCanvas(callback) {
    const { width, height } = elements.canvas;
    const imageData = canvasContext.getImageData(0, 0, width, height);
    const { data } = imageData;

    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;

    for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
            const index = (y * width + x) * 4;
            const r = data[index];
            const g = data[index + 1];
            const b = data[index + 2];
            const isInk = r < 245 || g < 245 || b < 245;

            if (!isInk) {
                continue;
            }

            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        }
    }

    if (maxX < 0 || maxY < 0) {
        callback(null, null);
        return;
    }

    const padding = 24;
    const cropX = Math.max(0, minX - padding);
    const cropY = Math.max(0, minY - padding);
    const cropWidth = Math.min(width - cropX, maxX - minX + 1 + padding * 2);
    const cropHeight = Math.min(height - cropY, maxY - minY + 1 + padding * 2);

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = cropWidth;
    tempCanvas.height = cropHeight;
    const tempContext = tempCanvas.getContext("2d");
    tempContext.fillStyle = "#ffffff";
    tempContext.fillRect(0, 0, cropWidth, cropHeight);
    tempContext.drawImage(
        elements.canvas,
        cropX,
        cropY,
        cropWidth,
        cropHeight,
        0,
        0,
        cropWidth,
        cropHeight
    );

    tempCanvas.toBlob(
        (blob) => callback(blob, tempCanvas.toDataURL("image/png")),
        "image/png",
        1
    );
}

async function parseApiResponse(response) {
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
        return response.json();
    }

    const bodyText = await response.text();
    return {
        error: bodyText.trim() || "The server returned an unexpected response.",
    };
}

async function executeAnalysis(fileOrBlob, filename, inputMode = "upload") {
    showState("loading");
    resetMessages();
    elements.analyzeBtn.disabled = true;
    elements.analyzeCanvasBtn.disabled = true;

    const formData = new FormData();
    formData.append("image", fileOrBlob, filename);
    formData.append("input_mode", inputMode);

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            body: formData,
        });
        const data = await parseApiResponse(response);
        const modeUsed = data.input_mode || inputMode;

        showState("result");
        elements.expression.textContent = data.expression || "N/A";
        elements.answer.textContent = data.result || "N/A";
        renderLineResults(data.lines || []);
        elements.bboxImage.src = data.display_image || "";
        elements.predictionSummary.textContent = buildPredictionSummary(data.characters || []);
        renderCharacterReview(data.characters || [], data.analysis_id || "");

        if (!response.ok || data.error) {
            elements.error.textContent = data.error || "The server could not process the input image.";
            elements.error.classList.remove("hidden");
            return;
        }

        const lowConfidence = Array.isArray(data.characters)
            ? data.characters.filter((item) => Number(item.conf || 0) < 0.8).length
            : 0;

        if (lowConfidence > 0) {
            elements.note.textContent = `The model is uncertain about ${lowConfidence} character(s). Current pipeline: ${modeUsed}. If the result looks wrong, separate the characters more clearly or use a sharper image.`;
            elements.note.classList.remove("hidden");
        } else if (Array.isArray(data.lines) && data.lines.length > 1) {
            elements.note.textContent = `The image contains ${data.lines.length} lines. Current pipeline: ${modeUsed}. Each line is analyzed separately to reduce cross-line misreads.`;
            elements.note.classList.remove("hidden");
        } else {
            elements.note.textContent = `Result generated by the ${modeUsed} segmentation -> classifier -> parser pipeline.`;
            elements.note.classList.remove("hidden");
        }
    } catch (error) {
        showResultError("Could not reach the Flask server.");
    } finally {
        elements.analyzeBtn.disabled = false;
        elements.analyzeCanvasBtn.disabled = false;
    }
}

function handleFiles(fileList) {
    const files = Array.from(fileList || []);
    const file = files.find((item) => item.type && item.type.startsWith("image/"));
    if (!file) {
        showResultError("Please choose an image file.");
        return;
    }

    setMode("upload");
    const reader = new FileReader();
    reader.onload = (event) => {
        updatePreview(file, event.target.result);
        resetResults();
    };
    reader.readAsDataURL(file);
}

elements.tabs.forEach((tab) => {
    tab.addEventListener("click", () => setMode(tab.dataset.mode));
});

["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    elements.dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
    });
});

["dragenter", "dragover"].forEach((eventName) => {
    elements.dropZone.addEventListener(eventName, () => elements.dropZone.classList.add("dragover"));
});

["dragleave", "drop"].forEach((eventName) => {
    elements.dropZone.addEventListener(eventName, () => elements.dropZone.classList.remove("dragover"));
});

elements.dropZone.addEventListener("drop", (event) => {
    handleFiles(event.dataTransfer.files);
});

elements.fileInput.addEventListener("change", (event) => {
    handleFiles(event.target.files);
});

elements.replaceBtn.addEventListener("click", () => {
    elements.fileInput.click();
});

elements.clearBtn.addEventListener("click", clearSelectedFile);

elements.analyzeBtn.addEventListener("click", () => {
    if (!state.selectedFile) {
        showResultError("Please choose an image before analyzing.");
        return;
    }

    executeAnalysis(state.selectedFile, state.selectedFile.name, "upload");
});

document.addEventListener("paste", (event) => {
    const items = Array.from(event.clipboardData?.items || []);
    const imageItem = items.find((item) => item.type.startsWith("image/"));
    if (!imageItem) {
        return;
    }

    const blob = imageItem.getAsFile();
    if (!blob) {
        return;
    }

    const pastedFile = new File([blob], "clipboard-image.png", { type: blob.type || "image/png" });
    handleFiles([pastedFile]);
});

elements.clearCanvasBtn.addEventListener("click", () => {
    clearCanvas();
    resetResults();
});

elements.canvas.addEventListener("pointerdown", (event) => {
    elements.canvas.setPointerCapture(event.pointerId);
    startDrawing(event);
});
elements.canvas.addEventListener("pointermove", draw);
elements.canvas.addEventListener("pointerup", stopDrawing);
elements.canvas.addEventListener("pointerleave", stopDrawing);
elements.canvas.addEventListener("pointercancel", stopDrawing);

elements.analyzeCanvasBtn.addEventListener("click", () => {
    exportTrimmedCanvas((blob) => {
        if (!blob) {
            showResultError("The canvas is empty. Draw an expression before analyzing.");
            return;
        }

        executeAnalysis(blob, "canvas-expression.png", "draw");
    });
});

elements.applyCorrectionBtn.addEventListener("click", applySelectedCorrection);
elements.removeSegmentBtn.addEventListener("click", rejectSelectedCharacter);
elements.resetCorrectionBtn.addEventListener("click", resetSelectedCorrection);
elements.submitFeedbackBtn.addEventListener("click", submitFeedback);
elements.feedbackInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault();
        applySelectedCorrection();
    }
});

clearCanvas();
resetResults();
