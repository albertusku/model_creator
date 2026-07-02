const state = {
  projectPath: "",
  projectHandle: null,
  projectFiles: null,
  project: null,
  imageIndex: -1,
  image: null,
  boxes: [],
  selectedBoxId: null,
  activeClassId: 0,
  drag: null,
  undo: [],
  trainingJobId: null,
  trainingPoll: null,
  autoReviewJobId: null,
  autoReviewPoll: null,
  trackingJobId: null,
  trackingPoll: null,
  completedTrackingJobId: null,
  poseJobId: null,
  posePoll: null,
  trackingCandidates: [],
  trackingCandidateIndex: 0,
  selectedTrackingCandidate: null,
  rightPanel: "images",
  trajectory: null,
  trajectoryVideoUrl: null,
  manualReviewImageIds: new Set(),
  autoSuggestToken: 0,
  projectModels: [],
  classificationDatasetId: null,
  classificationColumns: [],
  classificationTrainJobId: null,
  classificationTrainPoll: null,
  classificationClusterPlot: null,
};

const el = (id) => document.getElementById(id);
const canvas = el("canvas");
const ctx = canvas.getContext("2d");
const trajectoryVideo = el("trajectoryVideo");
const trackingCandidateViewer = el("trackingCandidateViewer");
const trackingCandidateCanvas = el("trackingCandidateCanvas");
const trackingCtx = trackingCandidateCanvas.getContext("2d");
const canvasWrap = document.querySelector(".canvasWrap");
const img = new Image();
const trackingCandidateImg = new Image();
canvas.tabIndex = 0;

function setOptionsOpen(open) {
  document.body.classList.toggle("optionsOpen", open);
  el("drawerBackdrop").hidden = !open;
  el("toggleOptions").setAttribute("aria-expanded", String(open));
}

el("toggleOptions").onclick = () => setOptionsOpen(!document.body.classList.contains("optionsOpen"));
el("drawerBackdrop").onclick = () => setOptionsOpen(false);

el("autoSuggest").checked = localStorage.getItem("modelCreator.autoSuggest") === "true";
el("autoSuggest").onchange = () => {
  localStorage.setItem("modelCreator.autoSuggest", String(el("autoSuggest").checked));
};
el("showReviewed").checked = localStorage.getItem("modelCreator.showReviewed") === "true";
el("showReviewed").onchange = () => {
  localStorage.setItem("modelCreator.showReviewed", String(el("showReviewed").checked));
  renderImageList();
};
el("showImageQueue").onclick = () => {
  stopTrajectoryPlayback();
  setRightPanel("images");
  focusAnnotator();
};
el("showTrackingQueue").onclick = () => setRightPanel("tracking");
el("projectsBasePath").value = localStorage.getItem("modelCreator.projectsBasePath") || "projects";
el("projectType").onchange = () => {
  const isDetection = el("projectType").value === "object_detection";
  el("classesLabel").hidden = !isDetection;
};
el("projectType").dispatchEvent(new Event("change"));

function setStatus(text) {
  el("status").textContent = text;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || response.statusText);
  }
  return response.json();
}

function renderProjectOptions(projects) {
  const select = el("projectSelect");
  select.innerHTML = "";
  if (!projects.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No projects found";
    option.disabled = true;
    option.selected = true;
    select.appendChild(option);
    return;
  }
  for (const project of projects) {
    const option = document.createElement("option");
    option.value = project.path;
    option.textContent = project.project_name && project.project_name !== project.name
      ? `${project.name} - ${project.project_name}`
      : project.name;
    select.appendChild(option);
  }
}

async function refreshProjects() {
  const basePath = el("projectsBasePath").value.trim() || "projects";
  localStorage.setItem("modelCreator.projectsBasePath", basePath);
  const result = await api(`/api/projects/discover?base_path=${encodeURIComponent(basePath)}`);
  el("projectsBasePath").value = result.base_path || basePath;
  renderProjectOptions(result.projects || []);
}

function renderProjectModelOptions(models) {
  const select = el("projectModelSelect");
  select.innerHTML = "";
  if (!models.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = state.projectPath ? "No models found" : "Open a project first";
    option.disabled = true;
    option.selected = true;
    select.appendChild(option);
    return;
  }
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.path;
    option.textContent = model.name;
    select.appendChild(option);
  }
  const configuredPath = state.project?.model?.path || "";
  const configured = [...select.options].find((option) => option.value === configuredPath);
  if (configured) {
    select.value = configuredPath;
  }
}

function projectType() {
  return state.project?.project_type || "object_detection";
}

async function ensureProjectType(project, path = "") {
  if (project.project_type) return project;
  let selected = "";
  while (!["object_detection", "csv_classification"].includes(selected)) {
    selected = window.prompt("This project has no project_type. Enter object_detection or csv_classification:", "object_detection") || "";
  }
  project.project_type = selected;
  if (path) {
    return api("/api/projects/type", {
      method: "POST",
      body: JSON.stringify({ path, project_type: selected }),
    });
  }
  state.project = project;
  await saveBrowserProject();
  return project;
}

function setProjectMode() {
  const isClassification = projectType() === "csv_classification";
  document.body.classList.toggle("classificationMode", isClassification);
  document.querySelectorAll(".objectDetectionOnly").forEach((node) => {
    node.hidden = isClassification;
  });
  document.querySelectorAll(".classificationOnly").forEach((node) => {
    node.hidden = !isClassification;
  });
}

function selectedModelPath() {
  return el("projectModelSelect").value;
}

function projectPathFromName(name) {
  const slug = name
    .trim()
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "") || "untitled-project";
  const basePath = el("projectsBasePath").value.trim() || "projects";
  return `${basePath.replace(/\/+$/, "")}/${slug}`;
}

async function refreshProjectModels() {
  state.projectModels = [];
  if (!canUseBackendProject()) {
    renderProjectModelOptions([]);
    return;
  }
  const result = await api(`/api/model/files?project_path=${encodeURIComponent(state.projectPath)}`);
  state.projectModels = result.models || [];
  renderProjectModelOptions(state.projectModels);
  const selectedPath = selectedModelPath();
  if (selectedPath && selectedPath !== state.project?.model?.path) {
    await saveSelectedModelConfig({ silent: true });
  }
}

function currentImage() {
  if (!state.project || state.imageIndex < 0) return null;
  return state.project.images[state.imageIndex] || null;
}

function currentAnnotation() {
  const image = currentImage();
  if (!image) return { boxes: [], reviewed: false };
  return state.project.annotations[image.id] || { boxes: [], reviewed: false };
}

function loadProject(project, path) {
  project.images ||= [];
  project.videos ||= [];
  project.annotations ||= {};
  project.classes ||= [];
  project.classification ||= { datasets: [], runs: [], last_run_id: null };
  state.project = project;
  state.projectPath = path;
  state.projectHandle = null;
  state.projectFiles = null;
  state.manualReviewImageIds = new Set();
  state.imageIndex = project.images.length ? 0 : -1;
  state.selectedBoxId = null;
  state.completedTrackingJobId = null;
  state.poseJobId = null;
  state.classificationDatasetId = project.classification.datasets?.at(-1)?.id || null;
  state.classificationColumns = project.classification.datasets?.at(-1)?.columns || [];
  setProjectMode();
  renderClasses();
  renderTrackingControls();
  renderModelConfig();
  renderClassificationProject();
  renderImageList();
  setRightPanel("images");
  loadCurrentImage();
  setStatus(projectType() === "csv_classification" ? `${project.name} (CSV classification)` : `${project.name} (${project.images.length} images)`);
  if (projectType() === "object_detection") {
    refreshProjectModels().catch((error) => setStatus(error.message));
  } else {
    renderProjectModelOptions([]);
  }
}

function loadBrowserProject(project, handle, files = null) {
  project.images ||= [];
  project.videos ||= [];
  project.annotations ||= {};
  project.classes ||= [];
  project.classification ||= { datasets: [], runs: [], last_run_id: null };
  state.project = project;
  state.projectPath = "";
  state.projectHandle = handle;
  state.projectFiles = files;
  state.manualReviewImageIds = new Set();
  state.imageIndex = project.images.length ? 0 : -1;
  state.selectedBoxId = null;
  state.completedTrackingJobId = null;
  state.poseJobId = null;
  state.classificationDatasetId = project.classification.datasets?.at(-1)?.id || null;
  state.classificationColumns = project.classification.datasets?.at(-1)?.columns || [];
  setProjectMode();
  renderClasses();
  renderTrackingControls();
  renderModelConfig();
  renderClassificationProject();
  renderImageList();
  setRightPanel("images");
  loadCurrentImage();
  setStatus(projectType() === "csv_classification" ? `${project.name} (CSV classification, browser folder mode)` : `${project.name} (${project.images.length} images, browser folder mode)`);
  renderProjectModelOptions([]);
}

async function getProjectFileHandle(relativePath) {
  if (!state.projectHandle) return null;
  const parts = relativePath.split("/").filter(Boolean);
  let handle = state.projectHandle;
  for (const part of parts.slice(0, -1)) {
    handle = await handle.getDirectoryHandle(part);
  }
  return handle.getFileHandle(parts.at(-1));
}

async function saveBrowserProject() {
  if (!state.projectHandle || !state.project) return;
  state.project.updated_at = new Date().toISOString();
  const fileHandle = await state.projectHandle.getFileHandle("project.json");
  const writable = await fileHandle.createWritable();
  await writable.write(JSON.stringify(state.project, null, 2) + "\n");
  await writable.close();
}

function browserProjectFile(relativePath) {
  if (!state.projectFiles) return null;
  return state.projectFiles.get(relativePath.replace(/^\/+/, ""));
}

function focusAnnotator() {
  canvas.focus({ preventScroll: true });
}

function setRightPanel(panel) {
  if (projectType() === "csv_classification") return;
  state.rightPanel = panel;
  const showingImages = panel === "images";
  el("showImageQueue").classList.toggle("active", showingImages);
  el("showTrackingQueue").classList.toggle("active", !showingImages);
  el("showImageQueue").setAttribute("aria-selected", String(showingImages));
  el("showTrackingQueue").setAttribute("aria-selected", String(!showingImages));
  el("imageList").hidden = !showingImages;
  el("trackingCandidates").hidden = showingImages;
  document.querySelector(".imageFilter").hidden = !showingImages;
}

function renderModelConfig() {
  const model = state.project?.model;
  el("modelConfidence").value = model?.confidence ?? 0.25;
}

function renderClasses() {
  const select = el("activeClass");
  select.innerHTML = "";
  for (const item of state.project?.classes || []) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.name;
    select.appendChild(option);
  }
  state.activeClassId = Number(select.value || 0);
}

function renderTrackingControls() {
  const videoSelect = el("trackingVideo");
  const classSelect = el("trackingClass");
  videoSelect.innerHTML = "";
  classSelect.innerHTML = "";
  for (const video of state.project?.videos || []) {
    const option = document.createElement("option");
    option.value = video.id;
    option.textContent = video.source_name || video.stored_name || video.id;
    videoSelect.appendChild(option);
  }
  for (const item of state.project?.classes || []) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.name;
    classSelect.appendChild(option);
  }
  const confidence = state.project?.model?.confidence ?? 0.25;
  el("trackingConfidence").value = confidence;
  clearTrackingSelection();
}

function clearTrackingSelection() {
  state.trackingCandidates = [];
  state.trackingCandidateIndex = 0;
  state.selectedTrackingCandidate = null;
  state.trajectory = null;
  state.trajectoryVideoUrl = null;
  state.completedTrackingJobId = null;
  renderEmptyTrackingCandidates("No candidate frames. Generate candidates from the Tracking panel.");
  el("trackSelectedInstance").disabled = true;
  el("showTrajectory").disabled = true;
  el("generateOriginalPose").disabled = !el("trackingVideo").value;
  el("generateTrackingPose").disabled = true;
  trackingCandidateViewer.hidden = true;
}

function stopTrajectoryPlayback() {
  trajectoryVideo.pause();
  trajectoryVideo.hidden = true;
  trackingCandidateViewer.hidden = true;
  canvas.hidden = false;
  canvasWrap.classList.remove("candidateMode", "trajectoryMode");
}

function renderImageList() {
  const list = el("imageList");
  list.innerHTML = "";
  if (!state.project) return;
  state.project.images.forEach((image, index) => {
    const annotation = state.project.annotations[image.id] || { reviewed: false };
    if (!el("showReviewed").checked && annotation.reviewed) return;
    const button = document.createElement("button");
    button.className = "imageItem";
    if (index === state.imageIndex) button.classList.add("active");
    if (annotation.reviewed) button.classList.add("reviewed");
    if (state.manualReviewImageIds.has(image.id) && !annotation.reviewed) button.classList.add("needsReview");
    button.textContent = `${index + 1}. frame ${image.source_frame}`;
    button.onclick = () => {
      stopTrajectoryPlayback();
      setRightPanel("images");
      state.imageIndex = index;
      loadCurrentImage();
      renderImageList();
      focusAnnotator();
    };
    list.appendChild(button);
  });
}

function renderEmptyTrackingCandidates(message) {
  const container = el("trackingCandidates");
  container.innerHTML = "";
  const empty = document.createElement("p");
  empty.className = "emptyState";
  empty.textContent = message;
  container.appendChild(empty);
}

function drawCandidateThumbnail(canvasElement, candidate) {
  const width = candidate.width || 800;
  const height = candidate.height || 500;
  canvasElement.width = width;
  canvasElement.height = height;
  const thumbCtx = canvasElement.getContext("2d");
  const thumbImg = new Image();
  thumbImg.onload = () => {
    thumbCtx.clearRect(0, 0, width, height);
    thumbCtx.drawImage(thumbImg, 0, 0, width, height);
    for (const box of candidate.boxes || []) {
      thumbCtx.strokeStyle = "#f59e0b";
      thumbCtx.lineWidth = Math.max(3, Math.round(width / 220));
      thumbCtx.strokeRect(box.x, box.y, box.width, box.height);
    }
  };
  thumbImg.src = candidate.image;
}

function loadCurrentImage() {
  const image = currentImage();
  const token = ++state.autoSuggestToken;
  state.selectedBoxId = null;
  state.undo = [];
  if (!image) {
    canvas.width = 800;
    canvas.height = 500;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    el("imageInfo").textContent = "No image";
    return;
  }
  const ann = currentAnnotation();
  state.boxes = structuredClone(ann.boxes || []);
  img.onload = () => {
    canvas.width = image.width;
    canvas.height = image.height;
    draw();
    maybeAutoSuggest(token);
  };
  if (state.projectHandle) {
    getProjectFileHandle(image.file)
      .then((fileHandle) => fileHandle.getFile())
      .then((file) => {
        img.src = URL.createObjectURL(file);
      })
      .catch((error) => setStatus(error.message));
  } else if (state.projectFiles) {
    const file = browserProjectFile(image.file);
    if (file) {
      img.src = URL.createObjectURL(file);
    } else {
      setStatus(`Image file not found in selected folder: ${image.file}`);
    }
  } else {
    img.src = `/api/images/${image.id}?project_path=${encodeURIComponent(state.projectPath)}&t=${Date.now()}`;
  }
  el("imageInfo").textContent = `${state.imageIndex + 1}/${state.project.images.length} ${image.width}x${image.height}`;
}

function canUseBackendProject() {
  return !state.projectHandle && !state.projectFiles && Boolean(state.projectPath);
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  for (const box of state.boxes) {
    const selected = box.id === state.selectedBoxId;
    ctx.strokeStyle = selected ? "#ffb000" : "#1f5eff";
    ctx.lineWidth = selected ? 4 : 2;
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    const baseName = state.project.classes.find((item) => item.id === box.class_id)?.name || box.class_id;
    const className = box.confidence ? `${baseName} ${(box.confidence * 100).toFixed(0)}%` : baseName;
    ctx.fillStyle = selected ? "#ffb000" : "#1f5eff";
    ctx.fillRect(box.x, Math.max(0, box.y - 22), ctx.measureText(className).width + 14, 22);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(className, box.x + 7, Math.max(14, box.y - 7));
    if (selected) {
      for (const handle of boxHandles(box)) {
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = "#ffb000";
        ctx.lineWidth = 2;
        ctx.fillRect(handle.x - 4, handle.y - 4, 8, 8);
        ctx.strokeRect(handle.x - 4, handle.y - 4, 8, 8);
      }
    }
  }
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function boxAt(point) {
  return [...state.boxes].reverse().find((box) => {
    return point.x >= box.x && point.x <= box.x + box.width && point.y >= box.y && point.y <= box.y + box.height;
  });
}

function boxHandles(box) {
  return [
    { name: "nw", x: box.x, y: box.y },
    { name: "ne", x: box.x + box.width, y: box.y },
    { name: "sw", x: box.x, y: box.y + box.height },
    { name: "se", x: box.x + box.width, y: box.y + box.height },
  ];
}

function resizeHandleAt(point) {
  const box = state.boxes.find((item) => item.id === state.selectedBoxId);
  if (!box) return null;
  return boxHandles(box).find((handle) => Math.abs(point.x - handle.x) <= 8 && Math.abs(point.y - handle.y) <= 8);
}

function pushUndo() {
  state.undo.push(structuredClone(state.boxes));
  if (state.undo.length > 20) state.undo.shift();
}

function clampBox(box) {
  box.x = Math.max(0, Math.min(box.x, canvas.width));
  box.y = Math.max(0, Math.min(box.y, canvas.height));
  box.width = Math.max(1, Math.min(box.width, canvas.width - box.x));
  box.height = Math.max(1, Math.min(box.height, canvas.height - box.y));
}

function persistableBoxes() {
  return state.boxes.map((box) => ({
    id: box.id,
    class_id: box.class_id,
    x: box.x,
    y: box.y,
    width: box.width,
    height: box.height,
  }));
}

async function saveCurrent(reviewed = currentAnnotation().reviewed || false) {
  const image = currentImage();
  if (!image) return;
  if (state.projectHandle) {
    state.project.annotations[image.id] = { reviewed, boxes: persistableBoxes() };
    await saveBrowserProject();
    renderImageList();
    return;
  }
  if (state.projectFiles) {
    state.project.annotations[image.id] = { reviewed, boxes: persistableBoxes() };
    renderImageList();
    setStatus("Browser fallback folder mode is read-only here. Reopen by backend path to save project.json.");
    return;
  }
  const ann = await api("/api/annotations", {
    method: "POST",
    body: JSON.stringify({
      project_path: state.projectPath,
      image_id: image.id,
      boxes: persistableBoxes(),
      reviewed,
    }),
  });
  state.project.annotations[image.id] = ann;
  if (ann.reviewed) state.manualReviewImageIds.delete(image.id);
  renderImageList();
}

canvas.addEventListener("pointerdown", (event) => {
  if (!currentImage()) return;
  focusAnnotator();
  const point = canvasPoint(event);
  const handle = resizeHandleAt(point);
  const hit = boxAt(point);
  pushUndo();
  if (handle) {
    const box = state.boxes.find((item) => item.id === state.selectedBoxId);
    state.drag = { mode: "resize", id: box.id, handle: handle.name, original: { ...box } };
  } else if (hit) {
    state.selectedBoxId = hit.id;
    state.drag = { mode: "move", id: hit.id, start: point, original: { ...hit } };
  } else {
    const box = {
      id: crypto.randomUUID(),
      class_id: state.activeClassId,
      x: point.x,
      y: point.y,
      width: 1,
      height: 1,
    };
    state.boxes.push(box);
    state.selectedBoxId = box.id;
    state.drag = { mode: "create", id: box.id, center: point };
  }
  canvas.setPointerCapture(event.pointerId);
  draw();
});

canvas.addEventListener("pointermove", (event) => {
  if (!state.drag) return;
  const point = canvasPoint(event);
  const box = state.boxes.find((item) => item.id === state.drag.id);
  if (!box) return;
  if (state.drag.mode === "move") {
    box.x = state.drag.original.x + point.x - state.drag.start.x;
    box.y = state.drag.original.y + point.y - state.drag.start.y;
    clampBox(box);
  } else if (state.drag.mode === "resize") {
    const original = state.drag.original;
    const right = original.x + original.width;
    const bottom = original.y + original.height;
    let left = original.x;
    let top = original.y;
    let nextRight = right;
    let nextBottom = bottom;
    if (state.drag.handle.includes("w")) left = point.x;
    if (state.drag.handle.includes("e")) nextRight = point.x;
    if (state.drag.handle.includes("n")) top = point.y;
    if (state.drag.handle.includes("s")) nextBottom = point.y;
    box.x = Math.max(0, Math.min(left, nextRight));
    box.y = Math.max(0, Math.min(top, nextBottom));
    box.width = Math.min(canvas.width, Math.max(left, nextRight)) - box.x;
    box.height = Math.min(canvas.height, Math.max(top, nextBottom)) - box.y;
    clampBox(box);
  } else {
    const width = Math.abs(point.x - state.drag.center.x) * 2;
    const height = Math.abs(point.y - state.drag.center.y) * 2;
    box.x = state.drag.center.x - width / 2;
    box.y = state.drag.center.y - height / 2;
    box.width = width;
    box.height = height;
    clampBox(box);
  }
  draw();
});

canvas.addEventListener("pointerup", async () => {
  if (!state.drag) return;
  const box = state.boxes.find((item) => item.id === state.drag.id);
  if (box && (box.width < 2 || box.height < 2)) {
    state.boxes = state.boxes.filter((item) => item.id !== box.id);
    state.selectedBoxId = null;
  }
  state.drag = null;
  draw();
  await saveCurrent();
});

el("createProject").onclick = async () => {
  const name = el("projectName").value.trim();
  const path = projectPathFromName(name);
  const type = el("projectType").value;
  const classes = type === "object_detection" ? el("classes").value.split(",").map((item) => item.trim()).filter(Boolean) : [];
  const project = await api("/api/projects", {
    method: "POST",
    body: JSON.stringify({ path, name, classes, project_type: type }),
  });
  loadProject(project, path);
  refreshProjects().catch((error) => setStatus(error.message));
};

el("browseProject").onclick = async () => {
  setStatus("Choose the project folder that contains project.json...");
  if (window.showDirectoryPicker) {
    const handle = await window.showDirectoryPicker({ mode: "readwrite" });
    const projectFile = await handle.getFileHandle("project.json");
    const file = await projectFile.getFile();
    let project = JSON.parse(await file.text());
    state.projectHandle = handle;
    project = await ensureProjectType(project);
    loadBrowserProject(project, handle);
    return;
  }
  el("projectFolderInput").click();
};

el("projectFolderInput").onchange = async () => {
  const files = [...el("projectFolderInput").files];
  const fileMap = new Map();
  for (const file of files) {
    const relative = file.webkitRelativePath.split("/").slice(1).join("/");
    fileMap.set(relative, file);
  }
  const projectFile = fileMap.get("project.json");
  if (!projectFile) throw new Error("Selected folder does not contain project.json");
  let project = JSON.parse(await projectFile.text());
  project = await ensureProjectType(project);
  loadBrowserProject(project, null, fileMap);
};

el("refreshProjects").onclick = async () => {
  await refreshProjects();
};

el("projectsBasePath").onchange = async () => {
  await refreshProjects();
};

function requireBackendProject(actionName) {
  if (state.projectHandle || state.projectFiles) {
    throw new Error(`${actionName} requires opening the project by path because the backend needs filesystem access.`);
  }
  if (!state.projectPath) {
    throw new Error("Open or create a project first");
  }
}

el("openProject").onclick = async () => {
  const path = el("projectSelect").value;
  if (!path) throw new Error("Select an existing project first");
  let project = await api("/api/projects/open", { method: "POST", body: JSON.stringify({ path }) });
  project = await ensureProjectType(project, path);
  loadProject(project, path);
};

el("importVideo").onclick = async () => {
  requireBackendProject("Video extraction");
  const file = el("videoFile").files[0];
  if (!file) throw new Error("Choose a video file");
  const button = el("importVideo");
  const form = new FormData();
  form.append("project_path", state.projectPath);
  form.append("every_n_frames", el("everyFrames").value);
  form.append("file", file);
  button.disabled = true;
  el("importStatus").textContent = "Extracting frames. This can take a while for long videos.";
  setStatus("Extracting frames from video...");
  try {
    const result = await api("/api/videos/import", { method: "POST", body: form });
    loadProject(result.project, state.projectPath);
    el("importStatus").textContent = `Extracted ${result.frames} images from the selected video.`;
  } finally {
    button.disabled = false;
  }
};

async function saveSelectedModelConfig({ silent = false } = {}) {
  requireBackendProject("Model configuration");
  const modelPath = selectedModelPath();
  if (!modelPath) throw new Error("Select a project model first");
  const model = await api("/api/model/config", {
    method: "POST",
    body: JSON.stringify({
      project_path: state.projectPath,
      model_path: modelPath,
      confidence: Number(el("modelConfidence").value),
    }),
  });
  state.project.model = model;
  renderProjectModelOptions(state.projectModels);
  if (!silent) setStatus("Model configuration saved");
}

el("saveModelConfig").onclick = async () => {
  await saveSelectedModelConfig();
};

el("projectModelSelect").onchange = async () => {
  if (el("projectModelSelect").value && canUseBackendProject()) {
    await saveSelectedModelConfig();
  }
};

el("suggestBoxes").onclick = async () => {
  await suggestBoxesForCurrentImage({ skipIfHasBoxes: false, showStatus: true });
};

function renderAutoReviewJob(job) {
  const counts = job.counts || {};
  el("autoReviewStatus").textContent = [
    `Status: ${job.status}`,
    `Processed: ${counts.processed || 0}/${counts.total || 0}`,
    `Auto-reviewed: ${counts.approved || 0}`,
    `Insufficient: ${counts.insufficient || 0}`,
    `Skipped: ${counts.skipped || 0}`,
    `Failed: ${counts.failed || 0}`,
  ].join(" | ");
}

function applyAutoReviewJob(job) {
  for (const [imageId, annotation] of Object.entries(job.updated_annotations || {})) {
    state.project.annotations[imageId] = annotation;
    if (annotation.reviewed) state.manualReviewImageIds.delete(imageId);
  }
  for (const imageId of job.manual_review_image_ids || []) {
    if (!state.project.annotations[imageId]?.reviewed) state.manualReviewImageIds.add(imageId);
  }
  const image = currentImage();
  if (image && job.updated_annotations?.[image.id]) {
    state.boxes = structuredClone(job.updated_annotations[image.id].boxes || []);
    state.selectedBoxId = null;
    draw();
  }
  renderImageList();
}

async function pollAutoReviewJob(jobId) {
  const job = await api(`/api/model/auto-review/${jobId}`);
  renderAutoReviewJob(job);
  applyAutoReviewJob(job);
  if (job.status === "completed" || job.status === "failed") {
    clearInterval(state.autoReviewPoll);
    state.autoReviewPoll = null;
    state.autoReviewJobId = null;
    el("autoReviewPending").disabled = false;
    setStatus(job.status === "completed" ? "Auto-review completed" : "Auto-review failed");
  }
}

el("autoReviewPending").onclick = async () => {
  requireBackendProject("Auto-review");
  const button = el("autoReviewPending");
  button.disabled = true;
  setStatus("Starting auto-review...");
  const job = await api("/api/model/auto-review/start", {
    method: "POST",
    body: JSON.stringify({
      project_path: state.projectPath,
      confidence: Number(el("modelConfidence").value),
    }),
  });
  state.autoReviewJobId = job.id;
  renderAutoReviewJob(job);
  applyAutoReviewJob(job);
  clearInterval(state.autoReviewPoll);
  state.autoReviewPoll = setInterval(() => pollAutoReviewJob(job.id).catch((error) => setStatus(error.message)), 1500);
  setStatus(`Auto-review job started: ${job.id}`);
};

function selectedTrackingPayload() {
  const selected = state.selectedTrackingCandidate;
  if (!selected) throw new Error("Select a candidate detection first");
  return {
    project_path: state.projectPath,
    video_id: el("trackingVideo").value,
    class_id: Number(el("trackingClass").value),
    start_frame: selected.frame,
    start_box: selected.box,
    confidence: Number(el("trackingConfidence").value),
  };
}

function renderTrackingCandidates(candidates) {
  const container = el("trackingCandidates");
  container.innerHTML = "";
  state.trackingCandidates = candidates;
  state.trackingCandidateIndex = 0;
  state.selectedTrackingCandidate = null;
  el("trackSelectedInstance").disabled = true;
  setRightPanel("tracking");
  if (!candidates.length) {
    renderEmptyTrackingCandidates("No candidate frames for this class and confidence.");
    trackingCandidateViewer.hidden = true;
    canvas.hidden = false;
    canvasWrap.classList.remove("candidateMode");
    return;
  }
  for (const [index, candidate] of candidates.entries()) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "candidateJump";
    const boxCount = (candidate.boxes || []).length;
    const thumbnail = document.createElement("canvas");
    thumbnail.className = "candidateThumb";
    thumbnail.setAttribute("aria-hidden", "true");
    const meta = document.createElement("span");
    meta.className = "candidateMeta";
    meta.textContent = `${index + 1}. ${candidate.time_sec.toFixed(0)}s | frame ${candidate.frame} | ${boxCount} ${boxCount === 1 ? "box" : "boxes"}`;
    button.append(thumbnail, meta);
    button.onclick = () => showTrackingCandidate(index);
    container.appendChild(button);
    drawCandidateThumbnail(thumbnail, candidate);
  }
  trackingCandidateViewer.hidden = true;
  canvas.hidden = false;
  canvasWrap.classList.remove("candidateMode");
  el("trackingCandidateInfo").textContent = "";
}

function selectTrackingCandidateBox(candidate, box) {
  state.selectedTrackingCandidate = { frame: candidate.frame, time_sec: candidate.time_sec, box };
  el("trackSelectedInstance").disabled = false;
  el("trackingStatus").textContent = `Selected frame ${candidate.frame} at ${candidate.time_sec.toFixed(0)}s.`;
  drawTrackingCandidate();
}

function drawTrackingCandidate() {
  const candidate = state.trackingCandidates[state.trackingCandidateIndex];
  if (!candidate) return;
  trackingCtx.clearRect(0, 0, trackingCandidateCanvas.width, trackingCandidateCanvas.height);
  if (trackingCandidateImg.complete && trackingCandidateImg.naturalWidth) {
    trackingCtx.drawImage(trackingCandidateImg, 0, 0, trackingCandidateCanvas.width, trackingCandidateCanvas.height);
  }
  for (const box of candidate.boxes || []) {
    const selected = state.selectedTrackingCandidate?.frame === candidate.frame && state.selectedTrackingCandidate?.box?.id === box.id;
    trackingCtx.strokeStyle = selected ? "#d88906" : "#2563eb";
    trackingCtx.lineWidth = selected ? 5 : 3;
    trackingCtx.strokeRect(box.x, box.y, box.width, box.height);
    const label = `${Math.round((box.confidence || 0) * 100)}%`;
    trackingCtx.font = "16px system-ui, sans-serif";
    trackingCtx.fillStyle = selected ? "#d88906" : "#2563eb";
    trackingCtx.fillRect(box.x, Math.max(0, box.y - 24), trackingCtx.measureText(label).width + 14, 24);
    trackingCtx.fillStyle = "#ffffff";
    trackingCtx.fillText(label, box.x + 7, Math.max(17, box.y - 7));
  }
}

function showTrackingCandidate(index) {
  const candidates = state.trackingCandidates;
  if (!candidates.length) return;
  state.trackingCandidateIndex = (index + candidates.length) % candidates.length;
  const candidate = candidates[state.trackingCandidateIndex];
  trajectoryVideo.pause();
  trajectoryVideo.hidden = true;
  canvas.hidden = true;
  trackingCandidateViewer.hidden = false;
  canvasWrap.classList.add("candidateMode");
  canvasWrap.classList.remove("trajectoryMode");
  setRightPanel("tracking");
  trackingCandidateCanvas.width = candidate.width || 800;
  trackingCandidateCanvas.height = candidate.height || 500;
  trackingCtx.clearRect(0, 0, trackingCandidateCanvas.width, trackingCandidateCanvas.height);
  trackingCtx.fillStyle = "#111827";
  trackingCtx.fillRect(0, 0, trackingCandidateCanvas.width, trackingCandidateCanvas.height);
  trackingCandidateImg.onload = () => {
    drawTrackingCandidate();
    el("trackingCandidateInfo").textContent = `Candidate ${state.trackingCandidateIndex + 1}/${candidates.length} | ${candidate.time_sec.toFixed(0)}s | frame ${candidate.frame} | ${(candidate.boxes || []).length} boxes`;
  };
  trackingCandidateImg.onerror = () => {
    el("trackingCandidateInfo").textContent = `Could not load candidate frame ${candidate.frame}`;
    el("trackingStatus").textContent = `Could not load candidate frame ${candidate.frame}`;
  };
  trackingCandidateImg.src = candidate.image;
  if (trackingCandidateImg.complete && trackingCandidateImg.naturalWidth) drawTrackingCandidate();
  document.querySelectorAll(".candidateJump").forEach((item, itemIndex) => {
    item.classList.toggle("active", itemIndex === state.trackingCandidateIndex);
  });
  el("trackingCandidateInfo").textContent = `Candidate ${state.trackingCandidateIndex + 1}/${candidates.length} | ${candidate.time_sec.toFixed(0)}s | frame ${candidate.frame} | ${(candidate.boxes || []).length} boxes`;
}

function trackingCandidatePoint(event) {
  const rect = trackingCandidateCanvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * trackingCandidateCanvas.width,
    y: ((event.clientY - rect.top) / rect.height) * trackingCandidateCanvas.height,
  };
}

trackingCandidateCanvas.addEventListener("pointerdown", (event) => {
  const candidate = state.trackingCandidates[state.trackingCandidateIndex];
  if (!candidate) return;
  const point = trackingCandidatePoint(event);
  const box = [...(candidate.boxes || [])].reverse().find((item) => {
    return point.x >= item.x && point.x <= item.x + item.width && point.y >= item.y && point.y <= item.y + item.height;
  });
  if (!box) return;
  selectTrackingCandidateBox(candidate, box);
});

el("generateTrackingCandidates").onclick = async () => {
  requireBackendProject("Tracking candidates");
  if (!el("trackingVideo").value) throw new Error("Add or import a tracking video first");
  const button = el("generateTrackingCandidates");
  button.disabled = true;
  clearTrackingSelection();
  el("trackingStatus").textContent = "Generating candidate frames...";
  try {
    const result = await api("/api/tracking/candidates", {
      method: "POST",
      body: JSON.stringify({
        project_path: state.projectPath,
        video_id: el("trackingVideo").value,
        class_id: Number(el("trackingClass").value),
        confidence: Number(el("trackingConfidence").value),
      }),
    });
    renderTrackingCandidates(result.candidates || []);
    const detectionCount = (result.candidates || []).reduce((total, candidate) => total + (candidate.boxes || []).length, 0);
    el("trackingStatus").textContent = detectionCount
      ? `Generated ${result.candidates.length} frames with ${detectionCount} detections.`
      : "No detections for this class. Change class or confidence and regenerate.";
  } finally {
    button.disabled = false;
  }
};

el("addTrackingVideo").onclick = async () => {
  requireBackendProject("Tracking video");
  const file = el("trackingVideoFile").files[0];
  if (!file) throw new Error("Choose a video file for tracking");
  const button = el("addTrackingVideo");
  const form = new FormData();
  form.append("project_path", state.projectPath);
  form.append("file", file);
  button.disabled = true;
  el("trackingStatus").textContent = "Adding tracking video...";
  setStatus("Adding tracking video...");
  try {
    const result = await api("/api/tracking/videos", { method: "POST", body: form });
    loadProject(result.project, state.projectPath);
    el("trackingVideo").value = result.video_id;
    clearTrackingSelection();
    el("trackingStatus").textContent = "Tracking video added. Generate candidate frames when ready.";
    setStatus("Tracking video added");
  } finally {
    button.disabled = false;
  }
};

function renderTrackingJob(job) {
  const processed = job.progress?.processed ?? 0;
  const count = job.trajectory?.length ?? 0;
  const parts = [`Status: ${job.status}`, `points: ${count}`, `processed: ${processed}`];
  if (job.selected_track_id != null) parts.push(`track_id: ${job.selected_track_id}`);
  if (job.error) parts.push(`Error: ${job.error}`);
  el("trackingStatus").textContent = parts.join(" | ");
}

async function pollTrackingJob(jobId) {
  let job;
  try {
    job = await api(`/api/tracking/${jobId}`);
  } catch (error) {
    clearInterval(state.trackingPoll);
    state.trackingPoll = null;
    state.trackingJobId = null;
    el("trackSelectedInstance").disabled = !state.selectedTrackingCandidate;
    const message = error.message.includes("tracking job not found")
      ? "Tracking job was lost. The server likely restarted; start tracking again."
      : error.message;
    el("trackingStatus").textContent = message;
    setStatus(message);
    return;
  }
  renderTrackingJob(job);
  if (job.status === "completed" || job.status === "failed") {
    clearInterval(state.trackingPoll);
    state.trackingPoll = null;
    state.trackingJobId = null;
    el("trackSelectedInstance").disabled = !state.selectedTrackingCandidate;
    if (job.status === "completed") {
      state.trajectory = {
        video_id: job.video_id,
        class_id: job.class_id,
        track_id: job.selected_track_id,
        points: job.trajectory || [],
      };
      state.trajectoryVideoUrl = job.video_url ? `${job.video_url}?t=${Date.now()}` : null;
      el("showTrajectory").disabled = !state.trajectoryVideoUrl;
      state.completedTrackingJobId = job.id;
      el("generateTrackingPose").disabled = !state.trajectoryVideoUrl;
      setStatus(`Tracking video ready with ${state.trajectory.points.length} tracked points`);
    } else {
      state.completedTrackingJobId = null;
      el("generateTrackingPose").disabled = true;
      setStatus(job.error || "Tracking failed");
    }
  }
}

el("trackSelectedInstance").onclick = async () => {
  requireBackendProject("Tracking");
  const button = el("trackSelectedInstance");
  button.disabled = true;
  el("trackingStatus").textContent = "Starting tracking...";
  const job = await api("/api/tracking/start", {
    method: "POST",
    body: JSON.stringify(selectedTrackingPayload()),
  });
  state.trackingJobId = job.id;
  state.completedTrackingJobId = null;
  state.trajectoryVideoUrl = null;
  el("showTrajectory").disabled = true;
  el("generateTrackingPose").disabled = true;
  renderTrackingJob(job);
  clearInterval(state.trackingPoll);
  state.trackingPoll = setInterval(() => pollTrackingJob(job.id).catch((error) => setStatus(error.message)), 1500);
  setStatus(`Tracking job started: ${job.id}`);
};

el("showTrajectory").onclick = async () => {
  if (!state.trajectoryVideoUrl) {
    setStatus("Run tracking first to generate a trajectory video");
    return;
  }
  canvas.hidden = true;
  trackingCandidateViewer.hidden = true;
  trajectoryVideo.hidden = false;
  canvasWrap.classList.remove("candidateMode");
  canvasWrap.classList.add("trajectoryMode");
  trajectoryVideo.src = state.trajectoryVideoUrl;
  trajectoryVideo.loop = true;
  try {
    await trajectoryVideo.play();
  } catch (error) {
    stopTrajectoryPlayback();
    setStatus(`Could not play trajectory video: ${error.message}`);
  }
};

function renderPoseJob(job) {
  const processed = job.progress?.processed ?? 0;
  const total = job.progress?.total;
  const parts = [`Pose: ${job.status}`, `processed: ${processed}${total ? `/${total}` : ""}`];
  if (job.error) parts.push(`Error: ${job.error}`);
  el("poseStatus").textContent = parts.join(" | ");
}

async function showRenderedVideo(videoUrl) {
  canvas.hidden = true;
  trackingCandidateViewer.hidden = true;
  trajectoryVideo.hidden = false;
  canvasWrap.classList.remove("candidateMode");
  canvasWrap.classList.add("trajectoryMode");
  trajectoryVideo.src = `${videoUrl}?t=${Date.now()}`;
  trajectoryVideo.loop = true;
  await trajectoryVideo.play().catch((error) => {
    stopTrajectoryPlayback();
    throw error;
  });
}

async function pollPoseJob(jobId) {
  const job = await api(`/api/pose/${jobId}`);
  renderPoseJob(job);
  if (job.status === "completed" || job.status === "failed") {
    clearInterval(state.posePoll);
    state.posePoll = null;
    state.poseJobId = null;
    el("generateOriginalPose").disabled = !el("trackingVideo").value;
    el("generateTrackingPose").disabled = !state.completedTrackingJobId;
    if (job.status === "completed" && job.video_url) {
      await showRenderedVideo(job.video_url);
      setStatus("Human pose video ready");
    } else {
      setStatus(job.error || "Human pose failed");
    }
  }
}

async function startPoseForVideo(source) {
  requireBackendProject("Human pose");
  if (!el("trackingVideo").value) throw new Error("Add or import a tracking video first");
  if (source === "tracking" && !state.completedTrackingJobId) {
    throw new Error("Generate a tracking video before running pose on it");
  }
  const originalDisabled = el("generateOriginalPose").disabled;
  const trackingDisabled = el("generateTrackingPose").disabled;
  el("generateOriginalPose").disabled = true;
  el("generateTrackingPose").disabled = true;
  el("poseStatus").textContent = "Starting human pose...";
  setStatus("Starting human pose...");
  try {
    const job = await api("/api/pose/start", {
      method: "POST",
      body: JSON.stringify({
        project_path: state.projectPath,
        video_id: el("trackingVideo").value,
        source,
        tracking_job_id: source === "tracking" ? state.completedTrackingJobId : null,
        confidence: Number(el("trackingConfidence").value),
      }),
    });
    state.poseJobId = job.id;
    renderPoseJob(job);
    clearInterval(state.posePoll);
    state.posePoll = setInterval(() => pollPoseJob(job.id).catch((error) => setStatus(error.message)), 1500);
    setStatus(`Human pose job started: ${job.id}`);
  } catch (error) {
    el("generateOriginalPose").disabled = originalDisabled;
    el("generateTrackingPose").disabled = trackingDisabled;
    throw error;
  }
}

el("generateOriginalPose").onclick = () => startPoseForVideo("original").catch((error) => setStatus(error.message));
el("generateTrackingPose").onclick = () => startPoseForVideo("tracking").catch((error) => setStatus(error.message));

el("trackingVideo").onchange = () => {
  stopTrajectoryPlayback();
  clearTrackingSelection();
};
el("trackingClass").onchange = () => {
  stopTrajectoryPlayback();
  clearTrackingSelection();
};

async function suggestBoxesForCurrentImage({ skipIfHasBoxes = false, showStatus = false } = {}) {
  requireBackendProject("Model suggestions");
  const image = currentImage();
  if (!image) throw new Error("Select an image first");
  if (skipIfHasBoxes && state.boxes.length > 0) return;
  pushUndo();
  if (showStatus) setStatus("Running model...");
  el("suggestStatus").textContent = "Running model suggestion...";
  const result = await api("/api/model/suggest", {
    method: "POST",
    body: JSON.stringify({
      project_path: state.projectPath,
      image_id: image.id,
      confidence: Number(el("modelConfidence").value),
    }),
  });
  const suggestions = result.boxes || [];
  state.boxes = state.boxes.concat(suggestions);
  state.selectedBoxId = suggestions.at(-1)?.id || state.selectedBoxId;
  draw();
  await saveCurrent(false);
  el("suggestStatus").textContent = `Added ${suggestions.length} suggested boxes`;
  if (showStatus) setStatus(`Added ${suggestions.length} suggested boxes`);
}

async function maybeAutoSuggest(token) {
  if (!el("autoSuggest").checked || !canUseBackendProject() || !currentImage()) return;
  if (state.boxes.length > 0) return;
  const imageId = currentImage().id;
  try {
    await suggestBoxesForCurrentImage({ skipIfHasBoxes: true });
  } catch (error) {
    el("autoSuggest").checked = false;
    el("suggestStatus").textContent = error.message;
    setStatus(error.message);
    return;
  }
  if (token !== state.autoSuggestToken || currentImage()?.id !== imageId) return;
  focusAnnotator();
}

function renderClassificationProject() {
  const datasets = state.project?.classification?.datasets || [];
  const latest = datasets.at(-1);
  if (latest) {
    state.classificationDatasetId = latest.id;
    state.classificationColumns = latest.columns || [];
    renderClassificationColumns(state.classificationColumns, latest.summary);
    el("classificationInfo").textContent = `${latest.source_name || latest.id}: ${latest.rows} rows, ${(latest.columns || []).length} columns`;
  } else {
    state.classificationDatasetId = null;
    state.classificationColumns = [];
    el("classificationTarget").innerHTML = "";
    el("classificationFeatures").innerHTML = "";
    el("classificationInfo").textContent = "No CSV dataset imported.";
    el("classificationMetrics").innerHTML = "";
    clearClassificationClusterPlot();
  }
  renderClassificationRuns();
}

function renderClassificationColumns(columns, summary = null) {
  const target = el("classificationTarget");
  target.innerHTML = "";
  for (const column of columns) {
    const option = document.createElement("option");
    option.value = column;
    option.textContent = column;
    target.appendChild(option);
  }
  if (columns.length) {
    target.value = columns.at(-1);
  }
  renderClassificationFeatures();
  if (summary) {
    const columnCount = Object.keys(summary.columns || {}).length;
    const preview = columns.slice(0, 6).join(", ");
    const suffix = columns.length > 6 ? ", ..." : "";
    el("classificationStatus").textContent = `Imported ${columns.length} columns: ${preview}${suffix}. Choose target and features before training.`;
    el("classificationInfo").textContent = `Dataset ready: ${columnCount} columns`;
  }
}

function renderClassificationFeatures() {
  const container = el("classificationFeatures");
  const target = el("classificationTarget").value;
  container.innerHTML = "";
  for (const column of state.classificationColumns) {
    if (column === target) continue;
    const label = document.createElement("label");
    label.className = "inlineControl featureItem";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = column;
    checkbox.checked = true;
    label.append(checkbox, document.createTextNode(column));
    container.appendChild(label);
  }
}

function selectedFeatureColumns() {
  return [...el("classificationFeatures").querySelectorAll("input[type='checkbox']:checked")].map((item) => item.value);
}

function renderClassificationRuns() {
  const select = el("classificationRun");
  const runs = state.project?.classification?.runs || [];
  select.innerHTML = "";
  if (!runs.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No trained runs";
    option.disabled = true;
    option.selected = true;
    select.appendChild(option);
    return;
  }
  for (const run of runs) {
    const option = document.createElement("option");
    option.value = run.id;
    option.textContent = `${run.id.slice(0, 8)} - ${run.target_column}`;
    select.appendChild(option);
  }
  select.value = state.project.classification.last_run_id || runs.at(-1).id;
}

function renderClassificationMetrics(metrics) {
  const container = el("classificationMetrics");
  container.innerHTML = "";
  if (!metrics || !Object.keys(metrics).length) return;
  const values = [
    ["Accuracy", metrics.accuracy],
    ["Macro F1", metrics.macro_f1],
    ["Weighted F1", metrics.weighted_f1],
    ["Train rows", metrics.train_rows],
    ["Test rows", metrics.test_rows],
  ];
  for (const [label, value] of values) {
    const item = document.createElement("div");
    item.className = "metricCard";
    const title = document.createElement("span");
    title.textContent = label;
    const number = document.createElement("strong");
    number.textContent = typeof value === "number" && value <= 1 ? value.toFixed(3) : String(value ?? "-");
    item.append(title, number);
    container.appendChild(item);
  }
  if (metrics.class_distribution) {
    const item = document.createElement("pre");
    item.className = "jsonBlock";
    item.textContent = `Class distribution\n${JSON.stringify(metrics.class_distribution, null, 2)}`;
    container.appendChild(item);
  }
  if (metrics.confusion_matrix) {
    const item = document.createElement("pre");
    item.className = "jsonBlock";
    item.textContent = `Confusion matrix\nLabels: ${(metrics.labels || []).join(", ")}\n${metrics.confusion_matrix.map((row) => row.join("  ")).join("\n")}`;
    container.appendChild(item);
  }
}

function clearClassificationClusterPlot(message = "Import a CSV and choose target/features to plot clusters.") {
  const canvasElement = el("classificationClusterCanvas");
  if (!canvasElement) return;
  const clusterCtx = canvasElement.getContext("2d");
  clusterCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  clusterCtx.fillStyle = "#f7f9fc";
  clusterCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);
  clusterCtx.fillStyle = "#5d6b7c";
  clusterCtx.font = "16px system-ui, sans-serif";
  clusterCtx.textAlign = "center";
  clusterCtx.fillText("No cluster plot", canvasElement.width / 2, canvasElement.height / 2 - 8);
  clusterCtx.font = "13px system-ui, sans-serif";
  clusterCtx.fillText(message, canvasElement.width / 2, canvasElement.height / 2 + 18);
  el("classificationClusterStatus").textContent = message;
}

function renderClassificationClusterPlot(plot) {
  state.classificationClusterPlot = plot;
  const canvasElement = el("classificationClusterCanvas");
  const clusterCtx = canvasElement.getContext("2d");
  const width = canvasElement.width;
  const height = canvasElement.height;
  const padding = { left: 52, right: 190, top: 28, bottom: 46 };
  const points = plot.points || [];
  clusterCtx.clearRect(0, 0, width, height);
  clusterCtx.fillStyle = "#ffffff";
  clusterCtx.fillRect(0, 0, width, height);
  if (!points.length) {
    clearClassificationClusterPlot("No rows available to plot.");
    return;
  }
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const colors = ["#2563eb", "#d88906", "#1f7a42", "#dc3454", "#7c3aed", "#0f766e", "#be123c", "#475569"];
  const labels = plot.labels || [...new Set(points.map((point) => point.label))].sort();
  const colorByLabel = new Map(labels.map((label, index) => [label, colors[index % colors.length]]));
  const toCanvasX = (value) => padding.left + ((value - minX) / rangeX) * plotWidth;
  const toCanvasY = (value) => padding.top + plotHeight - ((value - minY) / rangeY) * plotHeight;

  clusterCtx.strokeStyle = "#d7dfeb";
  clusterCtx.lineWidth = 1;
  clusterCtx.strokeRect(padding.left, padding.top, plotWidth, plotHeight);
  clusterCtx.fillStyle = "#5d6b7c";
  clusterCtx.font = "12px system-ui, sans-serif";
  clusterCtx.textAlign = "center";
  clusterCtx.fillText("PCA 1", padding.left + plotWidth / 2, height - 12);
  clusterCtx.save();
  clusterCtx.translate(16, padding.top + plotHeight / 2);
  clusterCtx.rotate(-Math.PI / 2);
  clusterCtx.fillText("PCA 2", 0, 0);
  clusterCtx.restore();

  for (const point of points) {
    clusterCtx.beginPath();
    clusterCtx.arc(toCanvasX(point.x), toCanvasY(point.y), 4, 0, Math.PI * 2);
    clusterCtx.fillStyle = colorByLabel.get(point.label) || "#2563eb";
    clusterCtx.globalAlpha = 0.72;
    clusterCtx.fill();
  }
  clusterCtx.globalAlpha = 1;

  clusterCtx.textAlign = "left";
  clusterCtx.font = "13px system-ui, sans-serif";
  clusterCtx.fillStyle = "#172033";
  clusterCtx.fillText("Target classes", width - padding.right + 28, padding.top + 2);
  labels.forEach((label, index) => {
    const y = padding.top + 28 + index * 22;
    clusterCtx.fillStyle = colorByLabel.get(label);
    clusterCtx.fillRect(width - padding.right + 28, y - 10, 12, 12);
    clusterCtx.fillStyle = "#172033";
    clusterCtx.fillText(String(label), width - padding.right + 48, y);
  });
  const variance = (plot.explained_variance || []).map((value) => `${(value * 100).toFixed(1)}%`).join(" / ");
  el("classificationClusterStatus").textContent = `Plotted ${points.length} rows with PCA variance ${variance || "n/a"}.`;
}

function renderClassificationJob(job) {
  const lines = [`Status: ${job.status}`];
  if (job.run_path) lines.push(`Run: ${job.run_path}`);
  if (job.model_path) lines.push(`Model: ${job.model_path}`);
  if (job.error) lines.push(`Error: ${job.error}`);
  el("classificationStatus").textContent = lines.join("\n");
  renderClassificationMetrics(job.metrics || {});
}

async function pollClassificationTraining(jobId) {
  const job = await api(`/api/classification/train/${jobId}`);
  renderClassificationJob(job);
  if (job.status === "completed" || job.status === "failed") {
    clearInterval(state.classificationTrainPoll);
    state.classificationTrainPoll = null;
    state.classificationTrainJobId = null;
    el("startClassificationTraining").disabled = false;
    if (job.status === "completed") {
      setStatus("Classification training completed");
      const project = await api("/api/projects/open", { method: "POST", body: JSON.stringify({ path: state.projectPath }) });
      loadProject(project, state.projectPath);
      renderClassificationMetrics(job.metrics || {});
    } else {
      setStatus(job.error || "Classification training failed");
    }
  }
}

function renderPredictionRows(rows) {
  const container = el("classificationPredictions");
  container.innerHTML = "";
  if (!rows?.length) {
    container.textContent = "No prediction rows returned.";
    return;
  }
  const table = document.createElement("table");
  const columns = Object.keys(rows[0]);
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const column of columns) {
    const th = document.createElement("th");
    th.textContent = column;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const column of columns) {
      const td = document.createElement("td");
      const value = row[column];
      td.textContent = value == null ? "" : column.startsWith("prob_") && typeof value === "number" ? `${(value * 100).toFixed(1)}%` : String(value);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.append(thead, tbody);
  container.appendChild(table);
}

el("classificationTarget").onchange = renderClassificationFeatures;

el("plotClassificationClusters").onclick = async () => {
  requireBackendProject("Cluster plot");
  if (!state.classificationDatasetId) throw new Error("Import a CSV dataset first");
  const featureColumns = selectedFeatureColumns();
  if (!featureColumns.length) throw new Error("Select at least one feature column");
  const button = el("plotClassificationClusters");
  button.disabled = true;
  el("classificationClusterStatus").textContent = "Calculating PCA cluster plot...";
  try {
    const plot = await api("/api/classification/clusters", {
      method: "POST",
      body: JSON.stringify({
        project_path: state.projectPath,
        dataset_id: state.classificationDatasetId,
        target_column: el("classificationTarget").value,
        feature_columns: featureColumns,
      }),
    });
    renderClassificationClusterPlot(plot);
    setStatus("Cluster plot ready");
  } finally {
    button.disabled = false;
  }
};

el("importClassificationCsv").onclick = async () => {
  requireBackendProject("CSV import");
  if (projectType() !== "csv_classification") throw new Error("Open a CSV classification project first");
  const file = el("classificationCsvFile").files[0];
  if (!file) throw new Error("Choose a CSV file");
  const form = new FormData();
  form.append("project_path", state.projectPath);
  form.append("file", file);
  el("importClassificationCsv").disabled = true;
  el("classificationStatus").textContent = "Importing CSV...";
  try {
    const result = await api("/api/classification/datasets/import", { method: "POST", body: form });
    state.classificationDatasetId = result.dataset_id;
    state.classificationColumns = result.columns || [];
    renderClassificationColumns(state.classificationColumns, result.summary);
    const project = await api("/api/projects/open", { method: "POST", body: JSON.stringify({ path: state.projectPath }) });
    loadProject(project, state.projectPath);
    setStatus(`Imported CSV with ${result.rows} rows`);
  } finally {
    el("importClassificationCsv").disabled = false;
  }
};

el("startClassificationTraining").onclick = async () => {
  requireBackendProject("Classification training");
  if (!state.classificationDatasetId) throw new Error("Import a CSV dataset first");
  const featureColumns = selectedFeatureColumns();
  if (!featureColumns.length) throw new Error("Select at least one feature column");
  const button = el("startClassificationTraining");
  button.disabled = true;
  setStatus("Starting classification training...");
  const job = await api("/api/classification/train/start", {
    method: "POST",
    body: JSON.stringify({
      project_path: state.projectPath,
      dataset_id: state.classificationDatasetId,
      target_column: el("classificationTarget").value,
      feature_columns: featureColumns,
      test_size: Number(el("classificationTestSize").value),
    }),
  });
  state.classificationTrainJobId = job.id;
  renderClassificationJob(job);
  clearInterval(state.classificationTrainPoll);
  state.classificationTrainPoll = setInterval(() => pollClassificationTraining(job.id).catch((error) => setStatus(error.message)), 1500);
  setStatus(`Classification job started: ${job.id}`);
};

el("predictClassificationCsv").onclick = async () => {
  requireBackendProject("Classification prediction");
  const file = el("classificationPredictFile").files[0];
  if (!file) throw new Error("Choose a CSV file to predict");
  const form = new FormData();
  form.append("project_path", state.projectPath);
  form.append("file", file);
  if (el("classificationRun").value) form.append("run_id", el("classificationRun").value);
  el("predictClassificationCsv").disabled = true;
  el("classificationPredictionStatus").textContent = "Predicting...";
  try {
    const result = await api("/api/classification/predict", { method: "POST", body: form });
    renderPredictionRows(result.rows || []);
    const link = el("classificationDownload");
    link.href = `${result.download_url}?project_path=${encodeURIComponent(state.projectPath)}`;
    link.hidden = false;
    el("classificationPredictionStatus").textContent = `Generated predictions from run ${result.run_id}.`;
    setStatus("Classification predictions ready");
  } finally {
    el("predictClassificationCsv").disabled = false;
  }
};

el("exportDataset").onclick = async () => {
  requireBackendProject("Export");
  const result = await api("/api/export", {
    method: "POST",
    body: JSON.stringify({
      project_path: state.projectPath,
      format: el("exportFormat").value,
      train: Number(el("splitTrain").value),
      val: Number(el("splitVal").value),
      test: Number(el("splitTest").value),
    }),
  });
  setStatus(`Exported: ${result.path}`);
};

function splitPayload() {
  return {
    train: Number(el("splitTrain").value),
    val: Number(el("splitVal").value),
    test: Number(el("splitTest").value),
  };
}

function renderTrainingJob(job) {
  const lines = [`Status: ${job.status}`];
  if (job.run_path) lines.push(`Run: ${job.run_path}`);
  if (job.best_model_path) lines.push(`best.pt: ${job.best_model_path}`);
  if (job.last_model_path) lines.push(`last.pt: ${job.last_model_path}`);
  const metricKeys = Object.keys(job.metrics || {}).slice(0, 6);
  if (metricKeys.length) {
    lines.push(`Metrics: ${metricKeys.map((key) => `${key}=${job.metrics[key]}`).join(", ")}`);
  }
  if (job.assets?.length) lines.push(`Graphs: ${job.assets.join(", ")}`);
  if (job.error) lines.push(`Error: ${job.error}`);
  el("trainingStatus").textContent = lines.join("\n");
}

async function pollTrainingJob(jobId) {
  const job = await api(`/api/training/${jobId}`);
  renderTrainingJob(job);
  if (job.status === "completed" || job.status === "failed") {
    clearInterval(state.trainingPoll);
    state.trainingPoll = null;
    state.trainingJobId = null;
    el("startTraining").disabled = false;
    if (job.status === "completed") {
      setStatus("Training completed");
      refreshProjectModels().catch((error) => setStatus(error.message));
    }
  }
}

el("startTraining").onclick = async () => {
  requireBackendProject("Training");
  const button = el("startTraining");
  button.disabled = true;
  setStatus("Creating training snapshot...");
  const job = await api("/api/training/start", {
    method: "POST",
    body: JSON.stringify({
      project_path: state.projectPath,
      model: el("trainModel").value.trim() || "yolo11n.pt",
      epochs: Number(el("trainEpochs").value),
      image_size: Number(el("trainImageSize").value),
      batch: el("trainBatch").value.trim() || "auto",
      device: el("trainDevice").value.trim() || "auto",
      ...splitPayload(),
    }),
  });
  state.trainingJobId = job.id;
  renderTrainingJob(job);
  setStatus(`Training job started: ${job.id}`);
  clearInterval(state.trainingPoll);
  state.trainingPoll = setInterval(() => pollTrainingJob(job.id).catch((error) => setStatus(error.message)), 2000);
};

el("activeClass").onchange = () => {
  state.activeClassId = Number(el("activeClass").value);
  const box = state.boxes.find((item) => item.id === state.selectedBoxId);
  if (box) {
    pushUndo();
    box.class_id = state.activeClassId;
    saveCurrent();
    draw();
  }
};

el("deleteBox").onclick = async () => {
  if (!state.selectedBoxId) return;
  pushUndo();
  state.boxes = state.boxes.filter((item) => item.id !== state.selectedBoxId);
  state.selectedBoxId = null;
  draw();
  await saveCurrent();
};

el("undo").onclick = async () => {
  const previous = state.undo.pop();
  if (!previous) return;
  state.boxes = previous;
  state.selectedBoxId = null;
  draw();
  await saveCurrent();
};

el("reviewImage").onclick = async () => {
  await saveCurrent(true);
};

function goToPreviousImage() {
  if (!state.project || state.imageIndex <= 0) return;
  stopTrajectoryPlayback();
  state.imageIndex -= 1;
  loadCurrentImage();
  renderImageList();
}

function goToNextImage() {
  if (!state.project || state.imageIndex >= state.project.images.length - 1) return;
  stopTrajectoryPlayback();
  state.imageIndex += 1;
  loadCurrentImage();
  renderImageList();
}

async function saveAnnotatedAndGoToNextImage() {
  if (currentImage() && state.boxes.length > 0) {
    await saveCurrent(true);
  }
  goToNextImage();
}

function cycleClass() {
  const classes = state.project?.classes || [];
  if (!classes.length) return;
  const currentIndex = classes.findIndex((item) => item.id === state.activeClassId);
  const nextClass = classes[(currentIndex + 1) % classes.length];
  el("activeClass").value = String(nextClass.id);
  el("activeClass").dispatchEvent(new Event("change"));
}

async function markEmptyReviewedAndNext() {
  if (!currentImage()) return;
  pushUndo();
  state.boxes = [];
  state.selectedBoxId = null;
  draw();
  await saveCurrent(true);
  await saveAnnotatedAndGoToNextImage();
}

el("prevImage").onclick = () => {
  goToPreviousImage();
  focusAnnotator();
};

el("nextImage").onclick = async () => {
  await saveAnnotatedAndGoToNextImage();
  focusAnnotator();
};

function isTextEntryTarget(target) {
  if (target instanceof HTMLTextAreaElement) return true;
  if (target instanceof HTMLSelectElement) return false;
  if (!(target instanceof HTMLInputElement)) return target?.isContentEditable || false;
  return !["button", "checkbox", "radio", "range", "file", "submit"].includes(target.type);
}

document.addEventListener("keydown", async (event) => {
  if (isTextEntryTarget(event.target)) return;
  const key = event.key.toLowerCase();
  let handled = true;
  if (event.key === "ArrowLeft") goToPreviousImage();
  else if (event.key === "ArrowRight" || key === "n") await saveAnnotatedAndGoToNextImage();
  else if (event.key === "Delete" || event.key === "Backspace") el("deleteBox").click();
  else if (key === "d") await markEmptyReviewedAndNext();
  else if (key === "c") cycleClass();
  else if (key === "z") el("undo").click();
  else handled = false;
  const numeric = Number(event.key);
  if (numeric >= 1 && numeric <= (state.project?.classes.length || 0)) {
    el("activeClass").value = String(state.project.classes[numeric - 1].id);
    el("activeClass").dispatchEvent(new Event("change"));
    handled = true;
  }
  if (handled) {
    event.preventDefault();
    focusAnnotator();
  }
}, { capture: true });

window.addEventListener("error", (event) => setStatus(event.message));
window.addEventListener("unhandledrejection", (event) => setStatus(event.reason?.message || String(event.reason)));

renderProjectModelOptions([]);
refreshProjects().catch((error) => setStatus(error.message));
