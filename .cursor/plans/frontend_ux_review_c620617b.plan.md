---
name: Frontend UX Review
overview: A comprehensive usability review of the frontend from the perspective of a non-technical crane business user, with prioritized improvement recommendations.
todos: []
isProject: false
---

# Frontend UX Review: Non-Technical Crane Business User Perspective

The following improvements are ordered roughly by impact. Each addresses a genuine usability barrier for someone who knows cranes, not machine learning.

---

## 1. No Guided Workflow -- Users Don't Know What to Do Next

**Where:** Project page, and the app overall

**Problem:** A crane operations manager opens a new project and sees three equal sections (Videos, Images, Classes) plus two action buttons (Train, Run Inference). Nothing tells them the order of operations: upload data, define classes, annotate, train, then run detection. The app treats these as peer features when they are actually a sequential pipeline. A first-time user will click "Train" before annotating anything, get an error toast, and feel lost.

**Rationale:** This is the single biggest barrier to adoption. Without a mental model of the workflow, every other feature is confusing. Tools like Roboflow succeed partly because they make the pipeline steps explicit.

**Suggestion:** Add a visual step indicator or checklist at the top of the project page (e.g., "1. Upload data -- 2. Define classes -- 3. Label objects -- 4. Train model -- 5. Run detection") with completion states. Highlight the current recommended next step. This doesn't have to block navigation -- just provide orientation.

---

## 2. ML Jargon Presented Without Explanation

**Where:** Training page (primarily), Inference page, Annotation pages

**Problem:** The training form immediately confronts users with "Epochs", "Batch Size", "Learning Rate", "Image Size", "Patience", "Grad Accumulation", "mAP50", "IoU threshold", and "RF-DETR". These terms are completely meaningless to someone in the crane industry. Even "Model Training" as a page title is somewhat abstract -- they think in terms of "teaching the system to recognize things."

The inference page uses "Inference" throughout. A crane operator would naturally say "test my model," "run detection," or "scan my videos." The word "inference" has no real-world parallel for them.

**Rationale:** When users encounter terms they don't understand, they either avoid the feature entirely or guess randomly. Both outcomes lead to poor model performance and abandonment.

**Suggestion:**

- Add short, plain-English tooltips on every technical parameter: e.g., Epochs becomes "Training rounds -- How many times the AI reviews all your images. More rounds = longer training but often better results." 
- Rename "Inference" throughout to "Run Detection" or "Test Model."
- Move Batch Size, Learning Rate, Patience, and Grad Accumulation entirely into Advanced (they currently split across main and advanced). Only expose Model Size and Epochs on the primary form.

---

## 3. Training Configuration Exposes Too Many Knobs by Default

**Where:** [TrainingPage.tsx](frontend/src/pages/TrainingPage.tsx), lines 431-480

**Problem:** The primary training form shows four technical parameters in a row: Epochs, Batch Size, Learning Rate, Image Size. A crane operations user doesn't know what any of these mean and shouldn't need to. The defaults are reasonable (50 epochs, auto batch size, 1e-4 lr, 640 image size), so most users would be best served by just clicking "Start Training."

**Rationale:** Exposing too many controls creates decision paralysis and a false impression that the tool requires expertise. This is the classic "expert vs. novice" UX trap. The 80% use case is: pick a model size, click train.

**Suggestion:** Collapse Epochs, Batch Size, Learning Rate, and Image Size into the "Advanced Options" section. The main form should only show Model Size (which is already well-presented with visual cards and plain descriptions) and perhaps a single "Training Duration" preset (Quick / Standard / Thorough) that maps to epoch counts internally.

---

## 4. Model Size Cards Lack Practical Guidance

**Where:** [TrainingPage.tsx](frontend/src/pages/TrainingPage.tsx), lines 46-52

**Problem:** The model cards show "RF-DETR Nano -- Fastest" through "RF-DETR Large -- Most accurate" but don't tell users: (a) what "RF-DETR" even is, (b) how much longer Large takes vs. Nano, (c) what's recommended for their data size, or (d) whether their GPU can even handle larger models. A crane business user choosing between these is essentially guessing.

**Rationale:** Model selection is the most consequential decision a user makes, and it's presented without enough context to decide. Choosing too large a model wastes time and may fail; too small may miss critical safety-relevant objects.

**Suggestion:** Remove "RF-DETR" from the card labels (just say "Nano", "Small", etc. -- the architecture name adds nothing for end users). Add a recommended badge based on dataset size (e.g., "Recommended for your 200 images"). Add approximate training time estimates based on past runs. Consider defaulting to "Base" with a note like "Good starting point for most projects."

---

## 5. CLI References and Filesystem Paths Exposed in UI

**Where:** [ProjectPage.tsx](frontend/src/pages/ProjectPage.tsx), lines 344-346 and 481-592

**Problem:** The Videos card description says: `Same as: python -m cli.videos add --project ... <file.mp4>`. The Images card tells users to "Place images in project_root/manual_data/ folder" and the upload help text says "Subfolder name goes to manual_data/your_name/." Delete confirmation mentions `data/.trash`. These are implementation details that leak the developer's mental model into the user's interface.

**Rationale:** A crane manager using the web UI will never run CLI commands. Seeing code snippets signals "this tool wasn't built for me." The `manual_data/` folder references imply a filesystem workflow that conflicts with the drag-and-drop UI right above it.

**Suggestion:** Remove all CLI references from the web UI. Replace filesystem paths with user-friendly descriptions. The "Subfolder" concept for image upload should be reframed as "Group" or "Category" (e.g., "crane closeups", "wide angle") so users understand the organizational purpose without knowing about folder structures.

---

## 6. "SAM3" and "Auto-label" Are Opaque Names

**Where:** [AnnotatePage.tsx](frontend/src/pages/AnnotatePage.tsx) line 457, [VideoAnnotatePage.tsx](frontend/src/pages/VideoAnnotatePage.tsx) line 501

**Problem:** The button says "Auto-label with SAM3." A non-technical user doesn't know what SAM3 is (it's Meta's Segment Anything Model 3). The modal title repeats "Auto-label with SAM3" and asks users to edit "class descriptions (SAM prompts)" -- another term that means nothing outside ML.

**Rationale:** This is arguably the most powerful feature of the tool (AI-assisted labeling saves hours), but it's hidden behind jargon. Users who don't understand what it does will skip it entirely and manually label hundreds of images.

**Suggestion:** Rename to "Auto-Detect Objects" or "AI-Assisted Labeling." In the modal, replace "Class descriptions (SAM prompts)" with "Describe what each object looks like" with a help note like "The more specific your descriptions, the better the AI will find objects. E.g., instead of 'hook', try 'metal crane hook hanging from cable'."

---

## 7. No Readiness Summary Before Training

**Where:** Training page, before the submit button

**Problem:** There's no pre-flight check or summary telling the user: "You have 150 annotated images, 3 classes (crane_hook: 89, person: 45, load: 16), using 2 videos and manual images." If they have zero annotations, they only discover this via an error toast after clicking "Start Training." There's no warning about class imbalance (e.g., "crane_hook has 5x more annotations than load -- model may underperform on rare classes").

**Rationale:** Training a model costs time (and potentially GPU credits). Users should have confidence they're starting from a reasonable dataset before committing. Class imbalance is one of the most common reasons models fail in practice, and it's entirely preventable with a warning.

**Suggestion:** Add a data summary card above the submit button showing: total annotations, per-class counts with a visual bar chart, a warning if any class has very few samples, and a green/yellow/red "readiness" indicator.

---

## 8. Annotation Page Has No Onboarding Cues

**Where:** [AnnotatePage.tsx](frontend/src/pages/AnnotatePage.tsx), [VideoAnnotatePage.tsx](frontend/src/pages/VideoAnnotatePage.tsx)

**Problem:** A first-time user lands on the annotation page and sees an image with a sidebar showing class names. The only hint is "Click and drag to draw" in tiny text at the bottom of the sidebar. There's no indication of what "drawing" means (bounding boxes), no visual demonstration, and the keyboard shortcuts (arrow keys, 1-9, Del, Ctrl+Z) are listed in 10px text in the bottom bar where they're easy to miss.

**Rationale:** Annotation is the most time-intensive task in the entire workflow. If users find it confusing for the first 5 minutes, they'll associate the whole tool with difficulty. A brief overlay or tooltip tour on first visit would save significant frustration.

**Suggestion:** On first visit (or when annotations list is empty), show a brief inline guide: "Select a class from the sidebar, then click and drag on the image to mark where objects are." Consider a pulsing visual hint on the first class in the sidebar. Make the keyboard shortcut reference more prominent or add a "?" help button.

---

## 9. "Regions" Is Unintuitive Terminology for Annotations

**Where:** Annotation sidebar in both [AnnotatePage.tsx](frontend/src/pages/AnnotatePage.tsx) line 687 and [VideoAnnotatePage.tsx](frontend/src/pages/VideoAnnotatePage.tsx) line 693

**Problem:** The sidebar labels the list of drawn boxes as "Regions (3)." In the context of labeling crane hooks and people, "regions" is abstract. A crane business user thinks in terms of "objects I've marked" or "labels."

**Rationale:** Small terminology mismatches accumulate into cognitive load. Every time the user mentally translates "regions" to "my markings," they spend attention that should be on the task.

**Suggestion:** Rename to "Labels" or "Marked Objects." This aligns with the "Label" header already used for the class list above it.

---

## 10. Confidence and Detection Interval Sliders Lack Context

**Where:** Inference run panel in [InferencePage.tsx](frontend/src/pages/InferencePage.tsx) lines 651-694

**Problem:** The confidence slider shows "Confidence: 50%" and the detection interval shows "Detection Interval: every 1 frames" -- but neither explains what these mean practically. A crane safety manager setting confidence to 10% will get flooded with false detections; setting it to 95% will miss real objects. They have no frame of reference.

The "Detection Interval" concept (running detection every N frames) is an optimization tradeoff that most users don't need to think about.

**Rationale:** These settings directly affect whether the model catches a person in a danger zone or misses them. The stakes are real in a crane business context. Users need to understand the tradeoff in plain terms.

**Suggestion:** Add descriptive labels: "Confidence: 50% -- Balanced (detects most objects, some false alarms)" with presets like "Strict (fewer detections, very reliable)" / "Balanced" / "Sensitive (catch everything, more noise)." For detection interval, default to 1 and hide it in Advanced unless the user is processing very long videos.

---

## 11. Hardcoded Singapore Timezone in Inference Results

**Where:** [InferencePage.tsx](frontend/src/pages/InferencePage.tsx) lines 342 and 382-387

**Problem:** Timestamps display as `toLocaleString('en-SG', { timeZone: 'Asia/Singapore' })`, which means every user everywhere sees times in SGT. A crane operator in Houston, Texas will see timestamps 13+ hours off from their local time, leading to confusion about when runs actually happened.

**Rationale:** This is a straightforward bug that erodes trust. When timestamps don't match the user's wall clock, they start second-guessing the tool.

**Suggestion:** Use the browser's local timezone: `new Date(timestamp).toLocaleString()` without specifying a timezone.

---

## 12. Browser `confirm()` Dialogs for Destructive Actions

**Where:** Throughout: [ProjectPage.tsx](frontend/src/pages/ProjectPage.tsx) lines 272, 448, 719; annotation pages for clearing

**Problem:** Most delete/clear operations use the native browser `confirm()` dialog, which is an ugly, unstyled OS popup that looks out of place in a modern web app. Notably, the project delete dialog IS a well-designed custom modal -- but other destructive actions are inconsistent.

**Rationale:** For a crane business spending real resources on this tool, visual polish signals reliability. Native `confirm()` popups feel like the app is unfinished. More importantly, they're easy to accidentally click through (hitting Enter dismisses them), increasing the risk of accidental data loss.

**Suggestion:** Replace all `confirm()` calls with consistent custom confirmation modals that match the app's design system. For high-stakes operations (delete class with annotations, clear all annotations), require an explicit action like typing the name or clicking a clearly labeled destructive button.

---

## 13. No Way to Compare Training Runs

**Where:** Training page sidebar, [TrainingPage.tsx](frontend/src/pages/TrainingPage.tsx) lines 768-801

**Problem:** Training runs are listed individually showing mAP50 for completed runs. After a user trains 3-4 models (which they will, iterating on data and settings), there's no way to compare: which run is best? How did mAP change between runs? Did adding more data help? They have to mentally track numbers across cards.

**Rationale:** The entire point of iterative model training is improvement over time. Without comparison, users can't tell if their efforts (more annotations, different settings) are actually helping. This makes the training loop feel like guesswork.

**Suggestion:** Add a simple comparison table or chart at the top of the runs list showing mAP50 trends across completed runs. Even a sparkline next to the run name would help. Highlight the "best run" with a badge.

---

## 14. Image Thumbnails Don't Show Annotation Count

**Where:** [ProjectPage.tsx](frontend/src/pages/ProjectPage.tsx) lines 639-658

**Problem:** The image grid on the project page shows a tiny green dot on annotated images, but no indication of how many annotations each image has. An image with 1 annotation looks the same as one with 20. This matters because a crane scene might have multiple objects (hook, person, load, safety equipment) and an image with only 1 of 4 labeled is incomplete.

**Rationale:** During the labeling phase, users need to quickly identify under-annotated images. Without annotation counts, they can't efficiently prioritize their review.

**Suggestion:** Show the annotation count as a small badge on each thumbnail (e.g., a colored number overlay in the corner). In the annotation page filmstrip, already-annotated thumbnails could show the count as well.