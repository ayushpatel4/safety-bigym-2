# Demo runbook & presentation logistics

Everything you need to run the talk from your own Mac. Read this once the day
before, then again the morning of.

---

## 0. Deliverables in this folder

| File | What it is |
|---|---|
| `safety_bigym_presentation.pptx` | **The deck** (open in Keynote or PowerPoint). 12 main + 7 Q&A appendix slides, speaker notes embedded. |
| `safety_bigym_presentation.pdf` | **Static backup** of the deck (videos show as still frames). Use only if Keynote/PowerPoint won't run. |
| `speaker_script.md` | Timed master script (to 16:30) + Q&A bank + delivery tips. |
| `demo_runbook.md` | This file. |
| `assets/clips/*.mp4` | Standalone demo clips (open in QuickTime as ultimate fallback). |
| `build_deck.py`, `build_clips.py` | Regenerators (edit + re-run to tweak). |

---

## 1. The live demo (the benchmark)

Runs on your Mac, **CPU only, no GPU, no trained policy, no AMASS** — robust.

**Primary command** (coworker parked at the robot, reaching at its end-effector):
```bash
cd /Users/ayushpatel/Documents/FYP3/safety_bigym
export AMASS_DATA_DIR=/Users/ayushpatel/Documents/FYP3/CMU/CMU
./venv/bin/mjpython scripts/demo_coworker.py --human g1 \
    --spawn in_place --reach-target ee --stage train --task saucepan --seed 0
```
- The coworker stands in the robot's working volume and **reaches at its end-effector on a cycle** — separation drops to ~0.15 m (verified: min sep 0.14 m, inside 0.3 m ~100% of steps).
- The **terminal prints** each reach phase (extend / hold / retract), the target, and the reach gate — keep it visible beside the viewer.
- Narrate per `speaker_script.md` slide 5. Key line: *"this is the persistent regime."*

> **Why `in_place --reach-target ee` (not `patrol`)?** With `--spawn patrol` the coworker
> walks in/out and only reaches in a later near-loiter phase, so the first ~10 s can look
> like it never reaches (separation ~1.4 m — the behaviour you saw). `in_place` parks it
> beside the robot and `--reach-target ee` aims every reach at the robot's arm, so it
> reaches **immediately and continuously**. For the full walk-in/depart rhythm instead,
> use `--spawn patrol --reach-target ee` and allow ~30 s.

**Optional 30 s second beat** (ISO monitor on a clean minimal scene; shows the
closest human-joint ↔ robot-link SSM margin and PFL on contact, printed live):
```bash
./venv/bin/mjpython scripts/demo_safety_visual.py
```

**Window arrangement (set up before the talk):**
1. Launch the primary command; drag the MuJoCo viewer to fill ~right two-thirds of the screen.
2. Put the launching **terminal** on the left third so the printed metrics are visible.
3. Orbit the camera once to a 3/4 view where both robot and coworker are clearly framed; leave it there.
4. Practise the toggle: viewer ↔ terminal ↔ Keynote.

**If the live demo misbehaves — do NOT debug on stage.** Switch to the backup:
- The deck's **slide 5 has the disruption recording embedded** — click it and narrate over it.
- Or open `assets/clips/clip_benchmark_disruption.mp4` in QuickTime.

---

## 2. Embedded result videos (woven into the slides)

These play **inside the deck** (no terminal needed). All four are **scripted
reconstructions** (real env / G1 coworker / ISO-15066 monitor, scripted robot;
HUD measured from sim) rendered by `render_demo_videos.py` — the *video* analogues
of report figures E.2–E.4:

| Slide | Clip | Shows |
|---|---|---|
| 5 (demo backup) | `clip_benchmark_disruption.mp4` | coworker **walks in → reaches at the robot (min 0.14 m, VIOLATION) → departs (away, 2.2 m) → returns** (seed 9 patrol) |
| 9 (RQ1) | `clip_avoid_compare.mp4` | baseline VIOLATION (works through) vs constrained yields & waits |
| 10 (RQ1) | `clip_veto_compare.mp4` | reactive veto: FREEZE (dwells, violation) vs FLEE (retreats, task lost) |
| 11 (RQ2) | `clip_speedscale_compare.mp4` | NO FILTER (SSM violation at speed) vs ISO-SSM speed-scaling (SSM-OK) |

> **Two flavours of the slides 9/10/11 clips exist.** The embedded ones are
> *scripted reconstructions* (robot motion is scripted; clean, deterministic).
> For **real trained-policy** rollouts (robot actually performs the task, as in
> the report's E.2–E.4), regenerate them on the GPU box — see §7. Drop the
> resulting `*_sidebyside_ep*.mp4` onto the matching slide.

Extra single-panel clips in `assets/clips/` for Q&A (open in QuickTime if asked):
`clip_avoid_baseline/constrained.mp4`, `clip_veto_freeze/flee.mp4`,
`clip_speedscale_on.mp4`. (The older `clip_persistent_*`, `clip_reactive_filter`,
`clip_hybrid` from the raw eval rollouts are still present but unused — the raw
rollouts don't show the contrast in 10 s, which is why the scripted clips replace them.)

> **Say once (slide 9):** these clips are *scripted reconstructions* — the trained
> snapshots live on the lab GPU box (~450 GPU-h to train), so the robot motion is
> scripted in the real environment, exactly as the report's figures are made; every
> HUD value is measured from the sim, and the *quoted numbers* come from the full
> benchmark (180 / 60 episodes).

---

## 3. Opening the deck in Keynote (do this once at home)

1. Double-click `safety_bigym_presentation.pptx` → Keynote offers to import → **Import**.
2. **Verify the four embedded videos play** (slides 5, 9, 10, 11): click each, press play.
   - If a video shows a "missing media" icon, drag the matching file from
     `assets/clips/` back onto the slide and resize to fill the placeholder.
3. Check fonts rendered (Helvetica Neue / Menlo are macOS built-ins — they will).
4. **Re-export your own PDF backup from Keynote** for perfect fidelity:
   `File → Export To → PDF…` (the included `.pdf` is a LibreOffice render and is
   already fine, but a Keynote export matches exactly what you'll present).
5. Use **Presenter mode**: `Play → In Window` (rehearse) or `Play → Slideshow`;
   speaker notes appear on your screen, slides on the projector.

> Prefer PowerPoint? It opens the `.pptx` natively with videos intact — same checks.

---

## 4. Optional: host clips online & add QR codes

The department suggests **linking** to videos rather than embedding. The deck
already embeds the two key clips (safer for offline rooms), but if you also want
shareable links on the slides:

1. Upload the clips to **YouTube (unlisted)** or **Imperial OneDrive / Google Drive (anyone-with-link)**.
2. Make a QR code per link, e.g.:
   ```bash
   ./venv/bin/python -m pip install qrcode
   ./venv/bin/python -c "import qrcode; qrcode.make('PASTE_LINK').save('presentation/assets/qr_rq1.png')"
   ```
3. Drop the QR PNG in a slide corner (Keynote: drag the image in).

Either way the **presentation recording captures whatever plays on screen**, so
embedded playback is fully sufficient.

---

## 5. Room / equipment checklist (the morning of)

- [ ] Laptop **charged + charger**; disable sleep/screensaver (System Settings → Lock Screen → Never), and Do Not Disturb on.
- [ ] **Display adapter** for the room (USB-C → HDMI/VGA); test mirroring/extend.
- [ ] Set resolution so 16:9 fills the screen; hide the dock; close Slack/Mail.
- [ ] **Mic check** — speak to the room, use the provided microphone.
- [ ] Pre-launch the **MuJoCo demo** and arrange windows (§1) *before* you start.
- [ ] Open the deck in **Presenter mode**; confirm notes show on your screen only.
- [ ] Confirm both **embedded videos play** on the room's machine/your laptop.
- [ ] **USB stick** with `.pptx`, `.pdf`, and `assets/clips/` as a hard backup.
- [ ] Have appendix slides 13–19 paged-to quickly for Q&A.
- [ ] Glass of water; visible **timer/phone** for the 16:30 pace markers.

---

## 6. Regenerating the assets (if you tweak anything)

```bash
cd /Users/ayushpatel/Documents/FYP3/safety_bigym
export AMASS_DATA_DIR=/Users/ayushpatel/Documents/FYP3/CMU/CMU
# (re)render the scripted demo videos (E.2-E.4 analogues + patrol backup).
# Needs the macOS window server for offscreen GL, so run it normally (not headless):
./venv/bin/python presentation/render_demo_videos.py --figure all --every 1 --cam-azimuth 210
# the slide-5 backup is the patrol (walk-in/away/return) clip on seed 9:
./venv/bin/python presentation/render_demo_videos.py --figure disruption --spawn patrol --seed 9 --cam-azimuth 210
# fast restyle of the 3 compare clips (re-overlay HUD on saved frames, no MuJoCo):
./venv/bin/python presentation/render_demo_videos.py --recompose --figure all
# then rebuild the .pptx and refresh the PDF backup:
./venv/bin/python presentation/build_deck.py
/Applications/LibreOffice.app/Contents/MacOS/soffice --headless \
    --convert-to pdf --outdir presentation presentation/safety_bigym_presentation.pptx
```
- `render_demo_videos.py --figure {avoid,veto,speedscale,disruption}` re-renders one clip.
  Poster frames (slide thumbnails) are extracted with `ffmpeg -ss <t> -frames:v 1`
  (avoid 12 s, veto 13 s, speedscale 30.1 s, disruption 40 s).
- `build_clips.py` (the older labelled raw-rollout clips) is retained but no longer feeds the deck.
Edit text/layout in `build_deck.py` (each slide is a clearly-commented block) or
just edit directly in Keynote after import.

---

## 7. Real trained-policy videos for slides 9 / 10 / 11 (run on the GPU box)

The embedded clips for slides 9–11 are *scripted reconstructions*. To show the
**actual trained policies performing the task** (as the report's E.2–E.4 do),
render them on the GPU box (where the snapshots live) with `scripts/render_policy_hud.py`.
It drives the real benchmark runner (same as `benchmark_policy.py`), reads each
step's separation / SSM / filter `Q` / scale live, burns in the **same report-style
HUD** as the other clips, runs both arms over the same seeds, auto-picks the
clearest episode, and writes a frame-aligned side-by-side.

```bash
cd ~/Documents/safety_bigym && source venv/bin/activate
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export AMASS_DATA_DIR=/path/to/CMU/CMU      # harness replays demos for action stats

BASE=exp_local/cqn_as_base_curriculum/base_g1_30k_30k_40k_20260529_124749/stage2_full/snapshot_28203.pt
# LAGR = your fixed-lambda=0.1 BASIN checkpoint (NOT snapshot_best.pt — peak success
# picks against the constraint). Find it with: scripts/pick_best_snapshot.py --by safety
LAGR=exp_local/.../fixlam0p1_seed0/.../snapshot_<basin_step>.pt
SVF=checkpoints/svf_coworker_train_g1_0p3_v3.pt
DISH=exp_local/curriculum/dish_rung1_a/stage2_full/snapshot_best.pt   # dishwasher baseline

# Slide 9 — RQ1: baseline vs constrained policy (saucepan, noisy)
python scripts/render_policy_hud.py --mode rq1 --baseline "$BASE" --lagrangian "$LAGR" \
    --episodes 10 --out-dir results/pres_hud/rq1

# Slide 10 — RQ1 filter: SVF veto FREEZE vs FLEE on the baseline (saucepan)
python scripts/render_policy_hud.py --mode veto --policy "$BASE" --svf-critic "$SVF" --R 2.25 \
    --episodes 10 --out-dir results/pres_hud/veto

# Slide 11 — RQ2: no-filter vs ISO-SSM speed-scaling (dishwasher)
python scripts/render_policy_hud.py --mode speedscale --policy "$DISH" \
    --d-slow 0.5 --d-stop 0.15 --episodes 10 --out-dir results/pres_hud/speedscale
```
Copy each `results/pres_hud/<mode>/<mode>_sidebyside_ep*.mp4` back to your Mac,
drop it onto the matching slide (replace the scripted clip), and extract a poster
with `ffmpeg -ss <t> -frames:v 1`. Tune `--R` / `--d-slow` if an arm over/under-fires.

### What each video should show (visual verification)

**Slide 9 — `rq1` (baseline | constrained).** Both robots do the saucepan task.
- LEFT (baseline): when the coworker reaches in, the robot **keeps working at the
  counter** — `min sep` turns **red**, the **VIOLATION** chip + red border fire
  repeatedly. It does not move away.
- RIGHT (constrained): as the coworker closes in, the robot **eases its base back /
  pauses** so `min sep` holds at **NEAR (amber)** rather than red, then **returns to
  the saucepan** once the coworker departs — and still completes the task.
- ✔ Verify: left shows clearly more red (VIOLATION) frames than right; right stays
  amber/green and still succeeds. The printed table shows `L_prox > R_prox` and `R_ok True`.

**Slide 10 — `veto` (freeze | flee).** Same baseline policy, SVF veto, two fallbacks.
- Both: when `Q(s,a) < R` the top-centre chip turns **red `… → VETO`**.
- LEFT (freeze): on veto the robot **stops** (robot vel ≈ 0) but the coworker keeps
  approaching, so `min sep` **still goes red** — the robot **dwells in the danger zone**.
- RIGHT (flee): on veto the robot **retreats** (base moves away); `min sep` recovers to
  **green/SAFE** but the robot **leaves the workspace and abandons the task**.
- ✔ Verify: the VETO chip fires on both; left stays in violation while frozen; right
  escapes to green but does **not** succeed. Table: both high `interv`; `L_prox` high,
  `R_prox` low with `R_ok False`.

**Slide 11 — `speedscale` (no filter | ISO-SSM scaling).** Same dishwasher policy.
- LEFT (no filter): robot does the task at speed; as the coworker passes, **robot vel
  stays high** and **SSM VIOLATION (red)** fires near the coworker.
- RIGHT (scaling): near the coworker the **`scale =` chip drops toward 0** and robot vel
  falls, so **SSM OK (green)** holds even at small separation; away from the coworker
  `scale = 1.00` (full speed) and the task still completes.
- ✔ Verify: left flashes red SSM VIOLATION near the coworker; right shows `scale < 1` +
  low robot vel + SSM OK at those same moments, full speed otherwise. Table: `L_ssm > R_ssm`.

> Notes: run on `--obs-mode noisy` (the SVF critic's native distribution; its Q collapses
> on oracle). The auto-pick prints a per-episode table; if the chosen episode isn't ideal,
> re-render with a different `--seed-base` or pick another `ep` clip by hand from the
> per-arm folders. `--max-steps` caps clip length (default 1000 ≈ 50 s at 20 fps).
