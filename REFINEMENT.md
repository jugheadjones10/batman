# Refinement Log

Running log of iterative fixes to the stacking-distance / side-view-schematic system.
Each entry records: the issue, what was done to investigate, the root cause, and the
attempted fix. Treat this file as context for all follow-up work.

---

## Entry 1 — 2026-08-28 — Empty-spreader pickup (and post-put-down) mis-tracked

### Current issue

At the start of a stacking video the spreader is **empty**, hovering a few feet above
the container it is about to pick up. It descends, grabs the container, hoists, and
moves it elsewhere. The side-view schematic does not handle this phase: it never
recognises the spreader as descending toward the container directly below it.
Instead it measures against ("locks") the container at the **center of the screen**,
which is a different container entirely.

Symmetrically, after the spreader **puts a container down** and hoists back up empty,
the schematic should keep measuring the distance between the just-placed container
and the departing spreader — it doesn't; the carried state never releases and the
empty phase again falls back to the center-of-screen container.

### What I did

Traced the full data path: `analyzeStacking` in `frontend/src/lib/stackingDistance.ts`
(runs over all smoothed ByteTrack tracks) → per-frame state consumed by
`SideViewSchematic.tsx` and `buildStackingOverlayBoxes` in `trackingPresentation.ts`.
Separately, the schematic's non-stacking container slot comes from
`pickPrimaryTrackPerClassFrames` / `pickCenterDetection`, which always collapses the
container class to the box nearest the **frame center**.

### What's causing it

Three compounding problems:

1. **False "carried" acquisition while hovering.** The carried-container cues
   (bbox overlap with spreader + velocity lockstep + depth agreement) all pass for a
   container sitting a few feet *below* a static empty spreader: overlap holds, both
   tracks are static so lockstep is trivially satisfied, and a ~1 m gap is inside the
   `CARRIED_MAX_Z_DELTA_MM = 1000` depth threshold. After 0.7 s the pickup target is
   flagged as *carried*. Nothing in the cues requires the pair to actually **move
   together** — attachment was never proven.
2. **Phantom placement lock.** Once descent starts (background still, spreader
   moving), the lock trigger fires — because a "carried" container exists — and locks
   the non-carried container nearest the *frame center* as the placement target.
   This is exactly the reported symptom: the empty pickup descent gets rendered as a
   loaded placement onto the center-screen container.
3. **No empty-spreader targeting and no put-down release.** The state machine is
   one-shot (once locked, locked forever) and the carried flag only releases when the
   track *disappears*, not when the spreader physically detaches. In every empty
   phase (before pickup, after put-down) the schematic just uses the center-of-screen
   container for the gap measurement.

### Attempted fix

Restructured `stackingDistance.ts` into a cyclic state machine
(`idle → pickup → carrying → locked → idle → …`) with per-episode lock/freeze:

1. **Attachment must be proven by joint motion.** Carried qualification time only
   accumulates while the spreader is actually moving, and lockstep now also requires
   the *relative bbox-scale rates* to agree (`CARRIED_SCALE_RATE_EPS`) — rigidly
   attached boxes scale identically, while a static container under a descending
   spreader does not. A static hover can no longer acquire the carried flag, and
   `CARRIED_MAX_Z_DELTA_MM` was tightened 1000 → 800 mm as an extra guard.
2. **Detach (put-down) detection.** If the carried track stays visible but fails the
   carried cues for a sustained `CARRIED_DETACH_SECONDS`, the flag releases. If a
   placement lock was active it ends there (an *episode* boundary), and the released
   container is seeded as the empty-spreader reference target — so the schematic
   measures spreader ↔ just-placed container as the spreader hoists away.
3. **Empty-spreader target = container nearest the spreader** (not the frame
   center), selected per frame with switch hysteresis (`EMPTY_TARGET_SWITCH_RATIO`)
   so it doesn't flicker between stacks. Its live Z drives the schematic's container
   slot during `idle`.
4. **Pickup lock (same mechanism as placement lock).** When the spreader is empty,
   the background has been still ≥ 1 s and the spreader has been *descending*
   ≥ 0.5 s, the current empty target is locked, its Z is frozen (median around the
   lock frame), and the same constrained-trajectory mapping runs with the contact
   plane at `targetTop` (instead of `targetTop − containerHeight`). This makes the
   pickup gap immune to the growing occlusion as the spreader covers the container.
5. **Direction awareness.** Locks (pickup and placement) now require sustained
   *descent*, detected from the signed spreader bbox-scale rate (shrinking = moving
   away from the overhead camera). This prevents the placement lock from firing
   during the post-pickup hoist, which the old code would have done.
6. **Per-episode analysis.** `lockFrameIndex/targetTrackId/targetZMm` moved from
   one-shot globals to an `episodes` list plus per-frame fields; frozen-Z and the
   constrained descent run per episode over `[lock, episodeEnd]`. The schematic and
   overlay were updated to read the per-frame fields, draw the pickup/nearest target,
   and show pickup status lines.

### Known trade-offs / watch for in the next iteration

- Carried acquisition now needs joint motion, so a video that *starts* mid-carry with
  a static spreader shows `idle` (with the carried box as the "nearest" empty target,
  gap ≈ 0) until hoist/lower movement begins. Acceptable for now.
- Descent direction comes from the signed bbox-scale rate with a 0.6 s neutral hold;
  if scale noise defeats it, the fallback is a Z-trend over a ~0.5 s baseline.
- Empty-target selection uses plain center-to-center distance to the spreader. If
  parallax makes the wrong stack win, weight the horizontal offset (line-of-movement)
  more heavily.
- Thresholds to tune live at the top of `stackingDistance.ts`:
  `CARRIED_SCALE_RATE_EPS = 0.02`, `CARRIED_DETACH_SECONDS = 1.0`,
  `CARRIED_MAX_Z_DELTA_MM = 800`, `EMPTY_TARGET_SWITCH_RATIO = 0.8`.

---

## Entry 2 — 2026-08-28 — "Stacking 1": first pickup still targets the wrong container; closed the feedback loop offline

### Current issue

After Entry 1, "Stacking 1" (project *Final testing*) still failed at the start:
the schematic never identified the container the spreader picks up first. User
asked to inspect the video + bounding boxes directly and iterate until fixed.

### What I did — offline feedback loop

Built a reproduction harness that runs the EXACT frontend pipeline outside the
browser, so results can be checked frame-by-frame against the real video:

1. `src.core.inference.compute_bytetrack_frames` over the run's `result.json`
   (same params the frontend requests: 0.25 / 30 / 0.8) → tracked frames JSON.
2. `frontend/scripts/debugStacking.ts` — bundles with esbuild
   (`--alias:@=./src`), applies the real `smoothFramesPerTrack` (One Euro
   1.0/0.007) and `analyzeStacking`, prints the state timeline + episodes, and
   dumps per-frame diagnostics.
3. `frontend/scripts/render_stacking_debug.py` — draws every detection and the
   analysis roles (carried/target/empty) onto real video frames for visual
   ground-truthing.

### What's causing it (three findings from the data)

1. **The pickup container is barely detected.** It sits directly under the
   spreader; the detector emits it as `container` only sporadically
   (conf 0.24–0.36), often with a bbox IDENTICAL to the `spreader`-class box
   (one merged blob). Around the old lock moment (~25 s) there was NO container
   detection at the pickup location for ~15 s, so the "nearest container" was
   the green NEIGHBOR stack (track 5) — which got locked.
2. **Every container bbox is edge-clipped in this camera geometry** (containers
   are longer than the field of view), so pinhole Z from container boxes is
   garbage: ~15.2 m for everything, constant. Only the calibrated `round`
   feature yields real Z (~1 m hoisted → ~2 m at first touchdown, ~6.2 m deep
   in the slot). Every container-derived gap number was physically meaningless.
3. **A carried container is INVISIBLE in nadir views.** It hangs directly under
   the spreader. Cue-based carried detection (overlap+lockstep+depth) cannot
   work for most of the cycle; during travel/second move the state machine fell
   apart (68 s bogus "pickup" episode, no placement ever detected).

### Attempted fix (all in `stackingDistance.ts`; UI untouched)

1. **Under-spreader gating.** New `resolveGeometrySpreaderClass` finds the
   dedicated `spreader` class (the Z class can be a small off-center proxy like
   `round`). Pickup/placement targets and the empty-spreader reference must
   overlap the spreader's geometry box (`TARGET_OVERLAP_MIN_FRAC = 0.2` of the
   smaller box). No candidate → target goes null; a wrong neighbor is never
   locked again.
2. **No carried acquisition while descending.** A load is physically acquired
   by HOISTING. Qualify time now only accumulates while moving and not
   descending — kills the false carry of the merged under-spreader blob during
   the pickup approach.
3. **Sequence (Z-profile) cycle inference.** `carrying` is now a STATE, not a
   track: entered by cue-based acquisition OR by the Z profile (pickup episode
   + touchdown + sustained re-ascent ≥ `TOUCHDOWN_BACKOFF_MM = 300`); losing
   the container track no longer ends the carry. Put-down requires the carry to
   have actually hoisted (`CARRY_HOIST_MIN_MM = 800` above the pickup plane)
   and re-descended before detach/ascent counts — the false put-down that
   previously fired during the post-pickup hoist is impossible now.
4. **Blind locks.** A lock may have `targetTrackId = null` when no container is
   detected under the spreader (target occluded by spreader/load). The target
   plane comes from touchdown contact; a visible target is adopted if one
   appears mid-episode.
5. **Contact-plane knowledge (`knownTopZByTrack`).** Touchdowns are the only
   reliable depth source in this geometry. Frozen target Z now prefers:
   episode's own touchdown contact → previously learned contact plane for the
   track → unclipped bbox medians (clipped boxes are excluded outright). After
   a placement the track's plane is the NEW top (contact); after a pickup it is
   the exposed container below (contact + container height). Idle-phase
   `emptyTargetZMm` prefers contact knowledge over live pinhole reads.

### Verified result on "Stacking 1" (offline harness + rendered frames)

- idle (empty=blue container when visible, honest null 19–25 s while undetected)
- pickup lock 24.9 s on the CORRECT blue container, plane 1987 mm inferred from
  touchdown, gap counts ~940 → 0 mm
- carrying 41 s → 140 s (survives track loss at 89 s, hoist + travel)
- placement locked 139.9 s, target plane 8922 mm (= contact 6331 + ISO height)
- put-down 202.7 s; idle then measures spreader ↔ placed plane 6331 mm — the
  post-put-down measurement Entry 1 wanted
- tail re-grab adjustments (209–284 s) all read the placed plane consistently

### Core problem statement / data needs (user asked)

- The detector genuinely struggles with the under-spreader container in this
  nadir view (tarped tops, merged spreader+container blobs, class confusion).
  More labeled frames of exactly this configuration would raise the floor.
- Container pinhole Z is unusable whenever the box is frame-clipped — that is
  ALL containers in this camera. Nothing model-side fixes that; contact-plane
  inference (implemented) or an additional calibration (e.g. known tier
  heights) are the options.

### Known trade-offs / watch for in the next iteration

- Contact planes learned per ByteTrack id can go stale if an id is reused
  across different stacks; post-processing fallbacks may read a plane learned
  later in the video (time travel). Static-yard assumption for now.
- A false cue-based acquisition now has no vanish-release escape (carrying
  persists until a put-down sequence). Acquisition gating makes this unlikely,
  but a long wrong carry is the failure mode to watch.
- Blind placement locks during travel wiggles can create short aborted
  episodes (e.g. 108.5–114.7 s) shown as `locked` with no gap. Cosmetic.
- New thresholds: `TARGET_OVERLAP_MIN_FRAC = 0.2`, `PICKUP_MIN_DESCENT_MM =
  300`, `TOUCHDOWN_BACKOFF_MM = 300`, `CARRY_HOIST_MIN_MM = 800`.

---

## Entry 3 — 2026-08-28 — Making the debug harness reproducible (tooling, typing, docs)

### Current issue

Entry 2's findings all came from an ad-hoc offline harness. Two loose ends made
it non-reproducible for the next iteration:

1. `frontend/scripts/*.ts` sits outside the app `tsconfig.json`'s `include`
   (`["src"]`), so the editor flagged 12 phantom errors in `debugStacking.ts`
   (unresolved `@/*` aliases, missing `node:fs` / `process` types, implicit
   `any` params) even though the esbuild bundle ran fine.
2. The exact command sequence to regenerate the analysis existed only in the
   chat, not in the repo or the docs.

### What's causing it

The harness is a Node script inside a browser-targeted Vite project. The app's
`tsconfig.json` intentionally scopes to `src` and does not declare `node`
types, so anything under `scripts/` is unowned by any TS project — hence
unresolved aliases and missing globals. The `@types/node` dependency was
already present; nothing was actually missing, only unconfigured.

### What I did / attempted fix

1. Added `frontend/scripts/tsconfig.json`: extends the app config, sets
   `types: ["node"]`, re-declares `baseUrl: ".."` + the `@/*` path mapping, and
   includes both `scripts/**/*.ts` and `../src/**/*.ts`. Type-checked as its
   own project, leaving the app config untouched.
2. Documented the harness in the mkdocs guide
   (`docs/guides/z-axis-height-estimation.md`) instead of a new markdown file,
   per the repo rule: an "Offline debugging harness" paragraph with the exact
   esbuild/bundle command, plus both scripts added to the source map.
3. Rewrote that guide's "Stacking distance across the crane cycle" section to
   describe the Entry 2 behaviour: the Z-class vs geometry-class split,
   under-spreader gating, carrying-as-a-state with Z-profile inference, blind
   locks, hoist-gated put-down, and contact-plane knowledge.

### Did it work

Yes. `tsc --noEmit -p tsconfig.json` → clean (app) and
`tsc --noEmit -p scripts/tsconfig.json` → clean (harness); `npm run build`
passes. The 12 editor errors are gone. One residual editor-only warning in
`stackingDistance.ts` ("This expression is always nullish" on the frozen-Z
fallback chain) is not reproduced by `tsc` on either project and was left
alone — the chain is intentionally source-ordered by physical trust.

### Reproduction (for the next iteration)

```bash
# 1. tracked frames, exactly as the frontend requests them (0.25 / 30 / 0.8)
.venv/bin/python -c "..."  # src.core.inference.compute_bytetrack_frames

# 2. run the real analysis pipeline offline
cd frontend
node_modules/.bin/esbuild scripts/debugStacking.ts --bundle --platform=node \
  --format=esm --alias:@=./src --outfile=/tmp/debugStacking.mjs
node /tmp/debugStacking.mjs "<path to result.json>"

# 3. draw detections + analysis roles on real video frames
.venv/bin/python frontend/scripts/render_stacking_debug.py \
  "<video>" /tmp/stacking1_analysis.json /tmp/out 26 41 150 204
```

### Watch for in the next iteration

- Only "Stacking 1" has been run through the loop. Stacking 2 and 3 are
  untested against the Entry 2 state machine and are the obvious next check.
- The harness hardcodes the One Euro params (1.0 / 0.007) and ByteTrack params;
  if `DEFAULT_OEF` / `DEFAULT_TRACKER_PARAMS` change in the frontend, the
  script drifts out of sync silently.

---

## Entry 4 — 2026-08-28 — Placement descent: target measured against the load itself

### Current issue

During the loaded descent onto another container, the container at the bottom
of the slot is "completely not detected", and the initial part of the descent
was unusable in the schematic.

### What I did

Ran the Entry-2/3 harness on "Stacking 1" and dumped, per frame, every
container box together with its overlap against the `spreader`-class box.

### What's causing it — one root cause, three symptoms

The detector emits **one merged box** covering the spreader and whatever is in
line with it, and reports that same box under BOTH the `spreader` and
`container` classes. In the placement descent that container track (#16) had a
box identical to the spreader's to four decimal places (`dx = 0.0000`),
shrinking in lockstep as the spreader descended. Consequences:

1. **The load was used as its own target.** `#16` overlapped the spreader
   fully, so it won "nearest under-spreader container" and became the
   placement target. The schematic measured the load against itself.
2. **The descent was fragmented into nothing.** `#16` was also counted as
   static *background*; its large apparent motion (cx 0.246 → 0.435) read as
   "the trolley moved" and aborted the lock episode repeatedly. The initial
   descent (114.7–128.7 s and 133.0–139.9 s) had no episode at all — exactly
   the reported symptom.
3. **The real target is genuinely invisible.** It sits under the carried
   container. No detector tuning recovers it; it must be inferred.

Two further defects surfaced while fixing the above (each caused a regression
mid-iteration, caught by the harness):

4. **Lockstep was judged against the wrong box.** Velocity/scale agreement
   compared containers to the *Z-class* box (the small off-center `round`
   casting), whose image motion differs from the spreader body's — so a
   rigidly attached load looked detached.
5. **Spurious `spreader` detections.** A second `spreader` box appears on
   container stacks seen end-on; its confidence sometimes exceeds the real
   spreader's, so confidence-based selection made the body box jump across the
   frame between consecutive frames (carried identity flapped 14↔16).
6. **Noise-driven episode aborts.** A relative bbox-scale-rate direction test
   with a 0.6 s neutral hold fakes sustained "descending"/"ascending" streaks
   while hovering, and the ascent-timer abort killed valid episodes during the
   landing plateau (twist-lock engagement holds the spreader at the touchdown
   plane for ~29 s here).

### Attempted fix (`stackingDistance.ts`, plus a UI honesty label)

1. **Merged blobs are explicitly modelled** (`isMergedWithSpreader`,
   `COINCIDENT_OVERLAP_FRAC = 0.8`, `COINCIDENT_SIZE_RATIO = 1.3`). Such a
   track carries NO information about attachment — its box is the spreader's
   box — so it is barred from cue-based acquisition and from detach evidence.
   Attachment for these comes only from the Z-profile sequence. It is still
   excluded from background, still excluded from placement targets while
   carrying, and still adopted as the carried track for display.
2. **State-dependent target candidates.** While carrying, anything moving with
   the spreader is removed from the target pool (it is the load); while empty,
   merged blobs are kept (during a pickup descent the blob is the best handle
   on the container underneath).
3. **Lockstep now compares against the spreader BODY** (`geometryVel`).
4. **`pickGeometrySpreader` anchors the body box to the Z-reference feature**:
   the round casting is physically mounted on the spreader, so the body is the
   candidate whose box contains it. Containment outranks any confidence
   margin.
5. **Locks require real depth travel** (`LOCK_MIN_DESCENT_MM = 100`) since the
   descent streak began, not merely a shrinking bbox.
6. **Removed the ascent-timer abort.** The Z-profile branch already resolves
   both outcomes from the same trigger (real rise off the deepest plane),
   routing to `aborted` when the approach never really descended.
7. **`targetKnownZMm` is snapshotted at lock/adopt time**, so a later
   episode's contact plane can no longer rewrite what an earlier episode was
   measured against (this closes the "time travel" trade-off flagged in
   Entry 2).
8. **UI honesty**: new `targetZInferred` flag; the schematic labels such a
   target `(inferred)` and the status line says "not detected — plane from
   touchdown", so a target with no bounding box does not look like a bug.

### Did it work — verified on "Stacking 1"

The whole cycle is now coherent, and the placement descent is covered from its
onset:

| phase | window | target | plane |
| --- | --- | --- | --- |
| pickup lock | 25.8 → 68.4 s | track 6 | 2021 mm |
| carrying | 68.4 → 129.2 s | — | — |
| **place lock** | **129.2 → 202.7 s** | **blind (undetected)** | **8922 mm** |
| put-down → idle | 202.7 s | placed track 16 | 6331 mm |

- Placement gap counts down monotonically from **5063 mm at lock** through
  1242 mm at 145 s to **0 mm at 197.6 s** (touchdown), instead of being pinned
  near zero against the load.
- Carried identity is stable (no 14↔16 flapping).
- **Independent confirmation of the inferred plane:** at t = 198 s the
  detector produced an *unclipped* box for track #20 (the container in the
  slot) reading **z = 8915 mm**, against the contact-inferred target plane of
  **8922 mm** — 7 mm apart, from two fully independent methods.

### Regression protection added

`debugStacking.ts` now asserts physical invariants and prints pass/fail:
placement target may never be the spreader blob; a completed episode's gap must
actually close (< 250 mm); a placement lock requires a carrying state; carried
identity must not flap. All hold on "Stacking 1", and on a smoke test of
`Phase 2` (pickup 41.6 → 128.9 s, put-down 151.2 s) and the `reverse video`
run (no episodes; that project has only 2 classes and no spreader class, so
class resolution degenerates — pre-existing, untouched).

### Watch for in the next iteration

- Track #20 was *sometimes* a valid, unclipped observation of the real target
  but sat just under `TARGET_OVERLAP_MIN_FRAC` (0.16 vs 0.20), so it is never
  adopted. Lowering the gate would draw a target box here but risks adopting
  side-neighbour stacks — the original Entry-1 bug. Left alone deliberately;
  the contact plane already agrees with it to 7 mm.
- The put-down instant (202.7 s) comes from the cue path while the spreader is
  still down in the slot; it then re-descends at 209.8 s and 223.2 s. Whether
  the true release is 202.7 s or later is not resolvable from the Z profile
  alone.
- `COINCIDENT_SIZE_RATIO` assumes the merged blob is close in size to the
  spreader box. A detector that merges the spreader with a much larger region
  would slip past it.

---

## Entry 5 — 2026-08-28 — "Stacking 2" starts loaded: the load state is not in the boxes

### Current issue

"Stacking 2" begins with the spreader already holding a container, and the
system does not detect it. The whole first move is mis-read.

### What I did

Ran the harness on the `video_2` run, dumped per-frame container/spreader
geometry, and rendered frames at the start, during the descent, and after the
move.

### What's causing it — a data limitation, not a logic bug

The state machine can only learn `carrying` from a **completed pickup**
(Entry 2's sequence inference) or from cue-proven attachment. "Stacking 2"
starts mid-cycle, so there is no pickup to infer from, and it assumes empty.
Consequences: the first descent (17.9 → 71.9 s) is labelled `pickup` although
it is physically a **placement**, no carried container is drawn, and the target
plane is one ISO height (2591 mm) too shallow because the placement contact
offset is not applied.

The deeper finding is why no cue can rescue this. **The load state has no
bounding-box signature at all:**

| time | video shows | container track vs spreader box |
| --- | --- | --- |
| 0–16 s | spreader **loaded** | `#3` overlap **1.00** |
| 120–136 s | spreader **empty** (visibly see-through) | `#3` overlap **1.00 / 0.99** |

The detector emits a `container` box coincident with the spreader body in
essentially every frame whether or not a load hangs from it — track `#3` is
just a duplicate of the spreader box. And no container track in the clip is
ever both distinct from the spreader box *and* moving in lockstep with it, so
there is no cue-based attachment evidence anywhere in the video. Per Entry 4's
principle, a merged blob carries no attachment information — so the state is
genuinely unobservable. The only visual difference is appearance: an empty
spreader frame is see-through. A box cannot express that.

Two other things I ruled out along the way:

- **Union-size plateau test** (a static container below should stop the merged
  box shrinking, while an attached load keeps shrinking with the spreader):
  dead end — the union box is clipped by the frame (size ≈ 1.0) throughout, so
  it carries no size information.
- **Container pinhole Z** as an independent target measurement: unusable here.
  The calibration is fitted from two labels at 1000 / 2000 mm on the `round`
  feature, and container boxes are edge-clipped, so yard containers read
  16–21 m while the spreader reads 1–7 m.

### Attempted fix

Asked which way to go; the decision was the **model-level fix**, so I did not
add a per-video override. Instead the model contract is now specified and the
consumption path is implemented:

1. **Model/labelling contract** (documented in the Training guide): split the
   spreader body class into `spreader_loaded` / `spreader_empty`. Labelling
   guidance included — switch to `loaded` only once the load is actually held
   (not when the target merely fills the footprint, which would end pickups
   early), and include clips that *start* loaded, since those are precisely the
   unresolvable cases.
2. **`loadStateOfClass`** reads the state from the body class name
   (`loaded|laden|full` vs `empty|unladen|bare`).
3. **Body-class resolution is now a set** (`resolveGeometrySpreaderClasses`),
   so all spreader-body variants are used interchangeably for spatial
   reasoning; splitting the class does not disturb the geometry gates.
4. **The class signal is authoritative** when present: the first reading seeds
   the state immediately (anchoring a mid-cycle start), later disagreements
   flip it after `LOAD_CLASS_CONFIRM_SECONDS = 1 s` of sustained evidence,
   `loaded` starts a carry and completes any open pickup, and `empty` releases
   through the normal put-down path.
5. **`carryHoisted` is now a tracked latch** instead of a derived expression. A
   carry declared by the class was hoisted before the clip began and has no
   pickup plane to compare against, so the old derived `hoisted` would have
   been permanently false and the put-down could never have fired. (The latch
   is equivalent to the old expression for motion-inferred carries: `carryMin`
   only decreases and `carryStart` is fixed, so the condition was already
   monotonic.)
6. **`StackingAnalysis.loadStateSource`** reports `'class'` vs `'inferred'`.

### Did it work

**Not yet verified.** My shell session wedged partway through this entry (a
heredoc left bash waiting on stdin), so I could not run the harness. What I do
have:

- The IDE TypeScript server reports no errors across `frontend/src`, and the
  running dev server HMR-ed the changes without a build error.
- **Stacking 1 cannot regress by construction:** every new behaviour is behind
  `hasLoadStateClasses`, which is false for a plain `spreader` class, and the
  `carryHoisted` latch is provably equivalent to the expression it replaced
  (see 5 above). The single-element class list makes `pickGeometrySpreader`
  behave exactly as before.

### Still to do

- Run the harness on Stacking 1 (regression) and on Stacking 2 via
  `frontend/scripts/simulate_load_classes.py`, which relabels a tracked dump
  into `spreader_loaded` / `spreader_empty` against a known release time so the
  new path can be exercised before a retrained model exists. Expected on
  Stacking 2: one `place` episode locking near 18 s with a contact-inferred
  target near 9479 mm (touchdown 6888 + 2591), and a put-down near 116 s.
- Retrain with the split class, then confirm `loadStateSource === 'class'` on a
  real run.
- Harness updated while waiting on the shell: it now prints the load-state
  source, asserts "a clip that starts loaded must not begin with a pickup"
  (checkable only when the model reports the state), and its
  spreader-blob check matches any body class rather than the literal
  `spreader`, so it keeps working once the class is split.

---

## Entry 6 — Relabelling the split class was impractical in the UI

### The issue

Entry 5 settled the fix: the model must emit `spreader_loaded` /
`spreader_empty`. That puts the next move on the labelling side, and the chosen
route was to relabel in the annotation UI. But the UI had no way to apply a
class change over a span of frames, which is the shape this particular relabel
takes.

### What's causing it

Two separate gaps, and I initially misread the first one:

- **Per-box reassignment already existed** and I reported it as missing. The
  number-key handler routes through `handleSelectClass`, which reassigns the
  selected annotation's class in addition to setting the draw class. Only the
  keyboard hint line advertises it, which is why it reads as absent.
- **Bulk reassignment genuinely did not exist.** The only multi-frame
  annotation operation was `clear-frames` (delete). So splitting a class over
  thousands of frames meant either one box at a time, or deleting and redrawing
  every box.

The second gap is what made the task impractical. Load state is constant across
long runs and changes once or twice per clip, so the natural unit of work is a
frame range, not a frame.

### The fix

1. **`POST /projects/{project}/annotations/reassign-class`**, mirroring the
   existing `clear-frames` shape: `frame_ids`, `to_class_label_id`, and an
   optional `from_class_label_id` (omitted means every class on those frames).
   It validates the target class, skips no-op annotations, and reports how many
   annotations and frames it touched.
2. **Reassign class dialog** in the video annotation toolbar: from/to class
   pickers plus three scopes — frame-number range, filmstrip selection, or
   current frame. It previews how many frames are in scope and how many carry
   annotations, and confirms before writing since there is no undo.
3. **The range resolves against every extracted frame**, not `filteredFrames`.
   The filmstrip shows every Nth frame, so scoping to visible frames would have
   silently skipped 4 in 5 at the default interval — exactly the kind of
   partial relabel that produces a subtly poisoned dataset.
4. **The current frame number is shown in the playback bar.** The range is
   expressed in frame numbers and the page previously displayed only a filmstrip
   index, so the range was unusable without it.
5. **Global number-key shortcuts are suppressed while a modal is open.** Digits
   typed over a `<select>` are not caught by the existing input guard, so they
   would have silently reassigned the selected box's class from inside the
   dialog. This also closes the same latent hazard for the SAM3 modal.

Deleting the emptied class afterwards is safe — `delete_class` already shifts
`class_label_id` for classes after the removed one. The cheaper route avoids the
question entirely: rename `spreader` → `spreader_empty` (index-preserving, so it
touches no boxes), add `spreader_loaded`, then reassign the loaded ranges. This
is written up in the Training guide next to the class contract.

### Did it work

**Not verified by running.** No linter or type errors across the touched files,
but my shell is still wedged (even `echo` fails to spawn), so I could not
exercise the endpoint. The dialog is inert until used, and the two changes that
touch existing behaviour are the keyboard suppression (strictly narrowing, and
only while a modal is open) and the added frame-number label. Still to check
once the backend is restarted: a range reassignment reports the expected count,
and the reassigned frames come back with the new class on reload.

---

## Entry 7 — Retrained load-state classes still declare an empty approach loaded

### The issue

The latest `Final testing` inference
(`rfdetr_local_20260901_151755/video_1/20260901_144813`) changes from empty to
carrying near the beginning of Stacking 1, while the spreader is still
approaching the pickup container.

### What's causing it

The relabel and checkpoint selection worked: the run's `class_info.json` has
`spreader_empty` and `spreader_loaded`, and frame 0 is correctly predicted
`spreader_empty`. The failure is that the detector does not produce a mutually
exclusive load state:

- In the first 20 seconds, `spreader_empty` appears on 491/500 frames,
  `spreader_loaded` appears on 285/500, and both appear on 283/500.
- The exact frontend harness changes to `carrying` at 11.28 s. The annotations
  still call the spreader empty at 16, 30, 38, and 44 s and do not first call it
  loaded until 56 s.
- At labeled-empty 30, 38, and 44 s, the model's loaded confidence is 0.551,
  0.479, and 0.509, respectively, versus empty confidence 0.206, 0.216, and
  0.220. Raising the inference threshold alone therefore cannot fix it.

Two factors create this:

1. Load state was encoded as two RF-DETR object classes, although it is one
   mutually exclusive attribute of the same physical spreader. Different DETR
   queries can emit overlapping loaded and empty boxes. The RF-DETR inference
   path does not apply cross-class suppression, and its `iou_threshold`
   parameter is unused.
2. The dataset is too small and sparse for this distinction. Only 78 training
   images contain a spreader-state annotation (34 empty, 44 loaded), only
   videos 1 and 2 are labeled, and the critical empty approach/contact region
   has large label gaps. Bulk reassignment changed existing annotations; it did
   not annotate the many missing spreader instances. The proportional split
   also randomly mixes adjacent frames from the same two videos across
   train/validation/test, so validation does not measure generalization to a
   new operation.

The frontend amplifies the detector error by treating the chosen class as
authoritative: one second of loaded observations changes the cycle state even
when empty and loaded boxes compete for the same object.

### Recommended fix

1. Collapse overlapping `spreader_empty` / `spreader_loaded` detections into
   one physical-body observation before tracking. If confidence is low or the
   two classes are close, report load state as unknown and preserve the prior
   state.
2. Fuse class evidence with cycle physics. Permit empty-to-loaded after a
   pickup contact/hoist cue; permit loaded-to-empty after placement contact.
   For a clip that begins mid-cycle, seed from a stronger multi-frame class
   vote so Stacking 2 can still begin loaded.
3. Fully annotate the spreader state on every training image where it is
   visible, concentrating on hard negatives immediately before pickup and
   immediately after put-down; add Stacking 3 and more independent clips; split
   evaluation by video/operation rather than adjacent frame.
4. Longer term, detect one `spreader` class and classify `loaded | empty` with
   a dedicated crop-based, preferably temporal, classifier. Load state is an
   attribute and attachment is partly a temporal event, so this matches the
   problem better than two detector object classes.

### Did it work

Diagnosis only; no production fix has been applied yet. The exact harness
reproduces the false transition at 11.28 s, so it can be used as the regression
case for the recommended fusion and suppression changes.


---

## Entry 8 — 2026-09-15 — Simplify inference by removing frame extraction and tracking comparison

### Current issue

The inference step includes Extract Frames and Tracking Comparison workflows
that the user wants removed to reduce the application to its essentials.

### What's causing it

Inference results link to two dedicated pages: a frame-selection/ZIP-download
viewer and a side-by-side tracking comparison tool. Frame downloads also have
an inference-only API client method and backend endpoint; the comparison page
has an exclusive video-synchronization hook.

### What I did to fix it

- Removed both action cards, page routes/imports, and dedicated page files.
- Removed the unused video-synchronization hook and inference frame-download
  client method, request model, endpoint, and exclusive imports.
- Updated the inference and tracking guides to describe the remaining workflow.
- Kept the shared ByteTrack/smoothing pipeline used by inference playback,
  Z-axis calibration, and the separate annotation frame-extraction workflow.
- Preserved the existing uncommitted annotation and stacking changes.

### Whether it worked

The frontend production build (`npm run build`, including TypeScript checking)
passed. Backend inference syntax compilation and `git diff --check` passed.
A reference scan found no remaining links, imports, or calls to the removed
pages and frame-download endpoint in frontend source, backend, or docs.
Browser interaction and live backend requests were not exercised. The build
reports warnings about stale Browserslist data and bundle size.


---

## Entry 9 — 2026-09-15 — Remove segmentation throughout the application

### Current issue

The user wants to remove the Segmentation training option and completely drop
its support, including the backend.

### What's causing it

Segmentation is threaded through the training UI, API schemas, local and GPU
launchers, CLI, model dispatch, and inference. It also adds polygon annotation
editing/storage, COCO mask exports, mask rendering, and a mask-based skew
estimator/readout. Removing only the training toggle would leave these paths
available through other entry points.

### What I did to fix it

- Removed the task selector and segmentation-only XLarge model. Training now
  exposes only nano, small, base, medium, and large detection models.
- Removed task arguments from the CLI, core trainer, inference loader, local
  launcher, and GPU script generation. API validation rejects obsolete task
  fields and unsupported model sizes instead of silently ignoring them.
- Removed RF-DETR-Seg dispatch, preview fallbacks, and native-resolution logic.
  Known non-detection checkpoint metadata is rejected before weights load;
  the legacy YOLO loader also requires a detection model.
- Removed polygon drawing, vertex editing, polygon undo state, annotation API
  fields, and polygon persistence from auto-labeling. Existing annotations
  retain their bounding boxes. Both COCO exporters now emit box annotations.
- Removed inference mask conversion/serialization, polygon rendering controls,
  the skew estimator/readout, and the segmentation spike script. Also removed
  unused rerender/comparison endpoints and helpers that exposed mask rendering.
- Kept SAM as a bounding-box auto-labeler, including its internal mask-to-box
  fallback. No polygon or mask labels are returned or persisted by that flow.
- Updated documentation, dependency comments, and the model-resolution rule;
  removed the segmentation/skew guide and links. Cleaned unused code and lint
  issues in touched files. Preserved existing stacking and annotation changes.
- Added regression coverage in `tests/test_detection_only.py`. Existing data
  and checkpoints were not deleted; old segmentation runs need retraining for
  detection to be loaded again.

### Whether it worked

- `npm run build` passed, including TypeScript checking.
- ESLint passed with zero warnings across the changed frontend source files.
- `pytest -q tests/test_detection_only.py tests/test_z_estimator.py`: 17 passed.
  Covers request validation, CLI rejection, local/GPU command construction,
  legacy checkpoint metadata rejection, both COCO exporters, inference output,
  SAM mask-to-box fallback, and existing Z-estimation behavior.
- In-process FastAPI requests to both training submit routes with the obsolete
  segmentation task returned HTTP 422 before any training was launched.
- Python compilation, Ruff's F checks on changed Python files, and
  `git diff --check` passed. No segmentation/polygon/skew support references
  remain in application source or active guides.
- Browser interaction and real GPU training/inference were not exercised.
  Existing build warnings (Browserslist data and bundle size) and a Pydantic
  configuration deprecation warning remain.

---

## Entry 10 — 2026-09-15 — Default inference confidence to 20%

### Current issue

The inference step starts at 50% confidence; the user wants 20% instead.

### What's causing it

The local and GPU form states each hardcode 0.5. The local slider, label, and
submission also fall back to 0.5 using `||`, which overrides an explicit 0%.

### What I did to fix it

Added one shared 0.2 default for both inference form states and the local
slider, label, and submission fallback. Changed the fallback to `??` so an
explicit 0% remains valid. Both local and GPU submissions send the selected
confidence to the backend.

### Whether it worked

`npm run build` passed, including TypeScript checking. The reference check
confirms all five form default/fallback locations use the shared value;
`git diff --check` passed. Browser interaction and actual inference were not
run. Existing Browserslist and bundle-size build warnings remain.

---

## Entry 11 — 2026-09-15 — Reduce Distance Calibration to reference and known distances

### Current issue

Distance Calibration presents measurement method, container length, feature
 diameter, reference class, target classes, and distance points as a long setup.
The intended output is the vertical clearance from the empty spreader or the
bottom of its load to the target container.

### What's causing it

The UI gives ordinary calibration and optional cross-object scale transfer equal
prominence. The estimator's whole-object fit only uses reference pixel sizes and
known camera distances: one point fits `k/s`; two or more fit `m/s + c`. Physical
length is not used by that fit. Round-feature-to-container transfer does require
the length/diameter ratio. Load state and target-plane inference are separate
tracking responsibilities, and currently use a fixed 2,591 mm load height.

Frame sampling also cleared entered points and filtered out saved labels on
reload. Saved reference initialization could race automatic selection, and the
page always submitted detection index zero and omitted an existing feature offset.

### What I did to fix it

- Reduced the active calibration page to two primary sections: object to measure
  and known camera distances. One distance is sufficient; the UI explains what
  adding a second point enables and distinguishes camera distance from clearance.
- Moved measurement method, physical dimensions, and target-class overrides into
  Advanced; collapsed the saved model details. New whole-spreader calibrations
  automatically include spreader/container classes, with the existing equal-size
  assumption documented. Existing target lists remain intact.
- Kept round-feature size transfer available, automatically exposed missing
  dimensions, and disabled incomplete submissions. Replaced the long help popup
  with guidance about required inputs, loaded/empty clearance, and visibility.
- Preserved calibration points across sampling changes and loaded saved points
  from all frames. Clicking a point outside the filmstrip switches to every-frame
  browsing. Preserved saved detection indices and feature offsets, gave saved
  reference selection priority, and required a matching reference detection and
  finite positive distance for each submitted point.
- Updated the active UI workflow in the distance-calibration guide. Changes are
  limited to the active calibration page, guide, and this log; concurrent changes
  elsewhere in the workspace were preserved.

### Whether it worked

- Frontend production build passed, including TypeScript checking. Existing
  Browserslist-age and bundle-size warnings remain.
- ESLint on `ZCalibrationPage.tsx` passed with zero warnings.
- All five existing `tests/test_z_estimator.py` tests passed.
- Headless Chromium checks with mocked API responses passed: minimum-input
  submission without physical dimensions, automatic target classes, correct
  reference detection index, point retention across sampling, reopening a saved
  round-feature calibration, navigation to an unsampled saved point, preservation
  of dimensions/target classes/feature offset, and blocked submission with missing
  round dimensions. No browser runtime errors occurred; inspected the rendered
  main form screenshot. The temporary harness is `/tmp/calibration-ui-check/check.cjs`.
- `git diff --check` passed. Live calibration persistence and accuracy on crane
  footage were not retested; the fitting and stacking algorithms are unchanged.
- Existing load-classification limitations, hidden targets inferred from touchdown,
  and the fixed container-height assumption remain. One point is the minimum for
  the camera-distance fit, not a guarantee of accurate clearance throughout a cycle.

---

## Entry 12 — 2026-09-15 — Use round-feature calibration by default

### Current issue

The user intends to use the round feature for distance measurement throughout
and wants the whole-bbox measurement choice removed from Advanced settings.

### What's causing it

The active page still defaults to whole-bbox measurement and the spreader class,
exposing a method selector for two workflows even though this setup only needs
round-feature measurement. Simply changing the method on saved whole-object
labels could reuse distances measured to a different reference plane.

### What I did to fix it

- Removed the active page's measurement-method selector and state. All new
  submissions explicitly use `round_feature_equivalent_length`.
- Automatically select `round` (then a round-like class name); leave the reference
  unselected with guidance when none exists. Custom feature class names can still
  be selected explicitly. New target defaults include the round reference plus
  spreader/container classes.
- Kept the required container length and round diameter with validation in
  Advanced, and updated field labels, feature highlighting, help, and the guide.
- Preserved saved round calibration points, detection indices, targets, dimensions,
  and feature offset. Historical whole-object models remain readable in inference;
  editing starts fresh round points and keeps only the reusable container length.
  An inline explanation states that the saved calibration remains active until
  the replacement is applied. No stored calibration is changed merely by opening it.

### Whether it worked

Frontend production build and page ESLint passed; all five existing estimator
tests passed. Headless Chromium checks with mocked API responses verified the
removed selector, round default and submitted method, dimensions, automatic
classes, retained points across sampling, saved round data/offset preservation,
blocked incomplete submissions, safe whole-object replacement setup, and the
missing-round case. No browser runtime errors occurred. The temporary check is
`/tmp/calibration-ui-check/round-default.cjs`. `git diff --check` passed.
The estimator and tracking algorithms were not changed; live crane-distance
accuracy was not retested. Existing build warnings remain.

---

## Entry 13 — 2026-09-15 — Document project folders under data

### Current issue

The user needs a Markdown document showing the projects under `data/`, with
explanations below the tree describing what belongs in each folder.

### What's causing it

Storage details are spread across workflow guides and implementation files, with
no single guide combining the current project folders and their nested contents.

### What I did to fix it

- Added `docs/guides/project-folder-structure.md` with a dated snapshot of all five
  projects, a representative nested layout, and a folder-purpose table below it.
- Explained optional artifacts, import sources, label iterations, COCO splits,
  training outputs, timestamped and legacy inference layouts, and calibration storage.
- Added the page to the Guides navigation in `mkdocs.yml`.

### Whether it worked

Verified the snapshot programmatically against the actual immediate project
folders. Checked the descriptions against storage code and existing artifacts;
all local Markdown links and the navigation target exist. Code fences and new
document whitespace passed checks; `git diff --check -- mkdocs.yml` passed.
This was a documentation-only change; application tests and a site build were
not needed for the folder inventory and were not run.

---

## Entry 14 — 2026-09-15 — Translate the edited folder guide into Korean

### Current issue

The user needs an exact Korean translation of their edited project-folder guide,
with actual folder names, filenames, paths, and identifiers kept in English.

### What's causing it

The guide is available only in English, including its folder explanations and
practical notes.

### What I did to fix it

- Created `docs/guides/project-folder-structure.ko.md` from the current edited
  English document, translating headings, explanations, tree comments, table
  descriptions, practical notes, and link labels.
- Preserved folder trees, technical identifiers, and link destinations, and kept
  the English source unchanged.
- Added the Korean guide to the documentation navigation.

### Whether it worked

Checked that the English source was unchanged; both trees match after accounting
for translated comments; all inline identifiers are preserved; heading structure,
table row counts, bullet counts, and link targets match. All local links resolve,
the Korean document has no trailing whitespace, and `git diff --check -- mkdocs.yml`
passed. Application tests were not run for this translation-only change.

---

## Entry 15 — 2026-09-15 — Remove current-project listings from both folder guides

### Current issue

The user wants the Current projects section removed from both the English and
Korean Markdown guides.

### What's causing it

Both guides include a dated inventory and tree of the current local projects.

### What I did to fix it

- Removed the Current projects / 현재 프로젝트 heading, introduction, and tree
  from both guides.
- Removed the imports-table sentence referring to the deleted five-project
  snapshot, preserving the remaining project template and explanations.

### Whether it worked

Verified that both sections and dangling snapshot references are removed, the
remaining content is preserved, and each guide retains one complete fenced
project-layout tree. No application tests were needed for this documentation edit.

---

## Entry 16 — 2026-09-22 — Keep Advanced settings open while editing dimensions

### Current issue

Entering the round feature diameter closes Advanced settings during calibration.

### What's causing it

The details element's `open` property depends on whether required dimensions are
missing. Once the selected length and entered diameter produce a valid size ratio,
React removes `open`, collapsing the panel while the user is still typing.

### What I did to fix it

- Gave Advanced settings independent open/closed state, synchronized with the
  details element's toggle event so manual opening and closing are preserved.
- Kept the panel initially open for new or incomplete calibrations and collapsed
  when loading saved round calibrations with valid dimensions. Subsequent edits
  update validation without changing the panel's open state.

### Whether it worked

Frontend production build (including TypeScript checking) and ESLint for
`ZCalibrationPage.tsx` passed. Existing Browserslist-age and bundle-size warnings
remain. The panel's open state no longer depends on live dimension validation.
Browser interaction was not retested. `git diff --check` for the page passed.


---

## Entry 17 — 2026-09-22 — Prepare accumulated changes for commit

### Current issue

The accumulated changes need committing, and the staged whitespace check flags
trailing whitespace in the newly added `AGENTS.md`.

### What's causing it

The final instruction line has a trailing space that was not checked while the
file was untracked.

### What I did to fix it

Removed the trailing space without changing the instructions and staged all
pending changes, including the existing refinement history.

### Whether it worked

The frontend production build passed and the focused detection-only and
Z-estimator tests passed (17 tests). Existing Browserslist, bundle-size, and
Pydantic deprecation warnings remain. Browser and GPU workflows were not run.
