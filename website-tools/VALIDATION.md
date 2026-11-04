# Project page validation

Validated 26 October 2026.

## Content and assets

- Read the supplied current manuscript in full, including the appendices. Scientific page copy, author affiliations, metric values, and figure mappings follow that version.
- All 11 supplied examples appear under their actual instructions and operations. The author confirmed `pitch2/target.wav` is a model output; `replace2` is grouped under removal because its instruction requests removal.
- All 20 published WAV files match the supplied originals by SHA-256. No audio levels or samples are changed.
- Nine high-resolution paper figures, the lab logo, social preview card, local font, and font license are present.
- Internal file links and fragment targets resolve; IDs are unique. JavaScript syntax and `git diff --check` pass.
- Static site size is approximately 6.8 MiB, dominated by the original WAV files. Audio loads on demand.
- No fabricated arXiv ID, conference acceptance, public paper URL, or citation is present.

## Browser verification

Chrome desktop and Chrome's responsive device mode were used. The browser had per-origin zoom settings, which made automated pointer coordinates unreliable; keyboard and native accessibility interactions were used to verify the actual controls.

- All 20 gallery audio elements were started successfully; no media errors occurred. Each start left exactly one audio element playing.
- Featured source/output switching preserved the current playback position and the playing state.
- Task switching, tab arrow/Home navigation, and audio seek controls work.
- Figures open in a modal and close with Escape; focus returns to the trigger.
- Overall capability remains visible as an 11-model table. The separate STEER-effectiveness block switches between Selective and Complex settings, with three backbone cards in each.
- Desktop page width and document scroll width match (no page-level horizontal overflow).
- In a 400 CSS-pixel device viewport, the page width and scroll width both equal 400. Mobile audio playback reached readyState 4 without errors.
- In the same mobile viewport, the Complex results table is 640 pixels wide inside a 362-pixel scrolling container; it does not widen the page.
- No application console warnings or errors were captured during the desktop interaction checks.

The website does not run the model locally. The online inference service and Hugging Face resources are external links with availability determined by their respective hosts.

## Delivery and hosting

- The completed source and all website assets were committed and pushed to the repository's new `gh-pages` branch.
- GitHub Pages enablement returned HTTP 422: “Your current plan does not support GitHub Pages for this repository.” The initial deployment workflow consequently failed at hosting configuration; this is not a page compilation or asset error.
- The author explicitly chose to preserve the private repository and receive the website branch without public deployment. Repository visibility and `main` were not changed.

## Local revision following author review

- Reorganized results around overall capability and STEER effectiveness. Dataset switches are inside the effectiveness block; they no longer mix datasets with a cross-model comparison at the same navigation level.
- Checked 11 capability rows, all three backbone cards in each setting, nine detailed training-stage rows, and the exact Table 2 / Table 3 aggregate values in the browser. No application warnings/errors or page-level desktop overflow were observed.
- Paper images now declare their intrinsic dimensions to reserve layout space during lazy loading.
- The author requested local review before any further push, with no deployment after pushing. The revision is left in the working tree for review; the automatic deployment workflow is deleted locally and will be removed remotely only with the next approved push.

## HTTP inference entry

- Reproduced the author's issue by clicking the actual Online inference resource card in the in-app browser. The DOM href is HTTP, but the destination tab becomes HTTPS and shows the tunnel error. Current-tab navigation also becomes HTTPS, so changing the target alone does not fix this browser behavior.
- Plain HTTP responds with 200 and no redirect; HTTPS responds with 404. Chrome opens the HTTP interface successfully. All project inference links remain exactly `http://powersound.natapp1.cc/`.
- Added a visible HTTP address and Copy URL fallback beside the resource cards. This does not claim to prevent the in-app browser's protocol change.
- Automatic approval review initially denied observing the resulting HTTPS page because the author explicitly requires HTTP. The author then authorized read-only inspection of that error page. A direct HTTP navigation in the same in-app browser also becomes HTTPS, independently of the project-page link. No browser security settings were changed.

## Review revision: original model and both STEER stages

- Cross-checked the new three-checkpoint comparisons against manuscript Tables 2–3. Each of the three model cards contains an explicit original-model row, an Emphasis row, and a Recovery/full-STEER row in both settings.
- Complex cards show both TBG reference values (0.00 for the original model), with an adjacent explanation of the relative metric. Source-removal LSD shows actual task scores at all three checkpoints: our editor 1.35 / 1.22 / 1.16; UNISON 1.58 / 1.39 / 1.36; SAM Audio 1.51 / 1.43 / 1.27.
- The Complex details table has nine checkpoint rows and ten columns, including all six primary task metrics at Table 2's reported precision. The Selective cards show all success/error values, including both Emphasis and Recovery.
- Checked desktop and a 400-pixel browser viewport override (385-pixel document area after scrollbar). No page-level or card overflow occurred. The wide task table scrolls within its own region. No browser application warnings or errors were captured.
- Changes remain local for author review. Nothing was committed, pushed, or deployed in this revision.

## Review revision: architecture frame and page cleanup

- Added a white, rounded frame around the architecture image, matching the other method figures. The existing figure viewer now also enlarges this diagram and closes correctly.
- Removed the colon after STEER in the paper title, preserving its intended verb-like reading.
- Removed the bottom HTTP-only notice, URL-copy control, and Paper & citation section at the author's request after HTTPS became available. Removed the associated clipboard handler and unused styles; audio players still load successfully.
- Checked desktop rendering and the image frame in a 400-pixel browser viewport override. No page-level horizontal overflow or application warnings/errors were observed. Changes remain local.

## Review revision: percentage gains without a zero baseline

- Replaced Complex-scene checkpoint cards with Emphasis and Recovery/full-STEER percentage gains. All 12 stage/metric values match paper Table 2, with an explicit plus sign and percent unit. Percentages remain cumulative relative to the corresponding original model.
- Removed the original-model zero row, zero-baseline explainer, and extra removal-score strip from the Complex cards. The expanded table retains nine training-stage rows and six original-reference gain cells, without displaying baseline zeros.
- Checked desktop and 800-/400-pixel viewport overrides. Percentages fit their containers at all sizes; document width and scroll width match. No application warnings or errors were captured. Temporary viewport overrides were reset.
- This revision remains local, pending author review; nothing was committed, pushed, or deployed.

## Desktop preview scale

- The narrow-looking desktop screenshot was reproduced in Chrome: its native toolbar reported 50% zoom, with a 3420-pixel CSS viewport around the 1140-pixel content column.
- Restored the site's Chrome zoom to the normal 100% through the native browser shortcut. The CSS viewport became 1710 pixels and devicePixelRatio changed from 1 to 2. The rendered content and text returned to their normal display size.
- No page layout or typography changes were needed for this issue. This adjustment affects the local browser preview; the website source does not force or override visitor zoom.

## Task-wise detail tables in both evaluation settings

- Added all task-wise Selective results from paper Table 3. Automatically compared all nine model/stage rows against the TeX source: all 90 task-level cells, including unsupported-task markers, and all 18 aggregate cells match.
- Both expanded tables now group columns by Generate, Add, Replace, Remove, Pitch, and Volume. Selective has 14 body columns, including task success, selective error for the four selective tasks, and overall summaries. Complex has 10 body columns with primary task metrics and aggregate gains.
- Verified both settings in the browser. The full Selective table fits the normal desktop content width; in a 400-pixel viewport it scrolls inside its region without widening the page. No application warnings or errors occurred.
- The Selective task table is left expanded for local author review. No changes have been committed, pushed, or deployed.

## Review revision: remove supplementary page copy

- Simplified the research-question section to a centered heading and the overview figure, removing its three introductory paragraphs.
- Removed visible waveform provenance/playback notes, figure captions, and supplementary benchmark/method notes. Scientific provenance remains documented here and in the research notes; result tables retain accessible captions and visible metric units in their column headings.
- Verified the updated section on desktop and in a 400-pixel viewport override. The mobile document width and scroll width both equal 385 pixels after the scrollbar; no page overflow occurred. Reset the viewport after checking.
- The audio gallery still initializes, metric tables render, and no application warnings or errors were captured. The local preview is left on the revised research section. No commit, push, or deployment was performed.

## Review revision: scale before source composition

- Replaced the combined data graphic with two sequential explanatory sections. Three crops from the original PDF separate scale, source-count distribution, and reference-audio examples without changing their content. Asset regeneration retains the same crop bounds.
- Checked the exact Selective training size against the manuscript's construction appendix: 61,440. The approximately one-million Complex figure is explicitly labeled as a raw audio pool used for online construction. Both settings retain 6,000 validation and 6,000 test examples.
- Verified desktop and 400-pixel viewport layouts; all three images load, and the mobile document width equals its 385-pixel scroll width. The cropped composition chart opens in the existing figure viewer. No changes were committed, pushed, or deployed.

## Review correction: integrate each dataset's example with its scale

- Replaced the split scale/composition sections with two complete dataset columns. Every dataset card contains its own scale, source composition, and reference example in sequence; corresponding rows align across the desktop comparison. Mobile presents one complete dataset card at a time.
- Scale counts are retained as HTML, with Complex's raw audio pool distinguished from fixed Selective training pairs. Source-count percentages remain 16.8 / 62.1 / 15.5 / 5.5 for Complex and 100% two-source references for Selective, scoped explicitly to Replace, Remove, Pitch, and Volume.
- Cropped the original PDF spectrograms into one image per dataset, preserving the source annotations. Both images load and open correctly in the existing figure viewer; the obsolete three intermediate crops are removed and asset regeneration now produces the two current examples.
- Checked the desktop row positions, responsive stacking, image dimensions, local asset paths, and browser logs. At a 400-pixel viewport override, document width and scroll width are both 385 pixels, with no card overflow. Viewport overrides were reset; no application warnings or errors were captured. Changes remain local, uncommitted, and unpushed.

## Review correction: parallel data settings

- Unified the data cards as Complex setting and Selective setting, with parallel audio-editing-data subtitles. Updated the Data navigation label, data-section title, result-setting tabs, summary copy, accessible descriptions, and metadata consistently.
- Cross-checked the Complex construction appendix: its nominal online training length is approximately 600,000 samples, from a raw pool of approximately one million entries. The main scale comparison now uses ~600K nominal training samples versus 61,440 fixed training samples; each setting has its own 6,000 validation and 6,000 test examples. The source pool is described separately from training samples.
- Verified desktop and mobile rendering, no horizontal overflow, and the renamed Complex setting tab still rendering all three model cards. Reset the mobile viewport override after checking. Source changes remain local, uncommitted, and unpushed.

## Research-led data redraw

- Completed a reference review and wrote `DATA_REDESIGN_PLAN.md` before editing. Data facts now live in `website/data/settings.json`; `build_data_figures.py` generates the static comparison and editable SVGs reproducibly.
- Checked the manuscript relationship between both source pools and both settings. The class-labeled and fine-captioned pool counts sum to 1,015,184 entries. The shared source node feeds two parallel setting columns with matching scale, composition and example rows.
- Verified all six split counts and eight source-count percentages against the supplied manuscript/figure. The main page uses responsive HTML for numbers, distributions and role labels. No measured audio, model outputs, plot values, or timestamps were created.
- Extracted original JPEG images 0 and 2 from the data PDF. Both the website JPEGs and JPEG payloads embedded in the SVGs are byte-for-byte copies of those PDF objects. The rebuilt vector labels sit outside the measured images.
- Verified SVG XML, local assets and script syntax. The standalone overview rendered correctly in Chrome; a figure-only crop of its actual browser screenshot was saved for review. Both annotated reference SVGs open in the existing figure viewer without errors.
- Checked desktop layout and 800-/400-pixel viewport overrides. Corresponding scale/composition/example rows align in both desktop columns. Document width equals scroll width (785 and 385 pixels after the scrollbar); no card overflow occurs. Mobile role labels render at 11 pixels rather than shrinking with the image. Reset all overrides. No application warnings/errors were captured.
- The local review page remains available. No commit, push, deployment or repository-visibility change was performed.

## Review revision: consistent compact counts and clearer hierarchy

- Standardized the visible data counts to lowercase k: ~600k / 61k training, 6k validation/test, and 635k / 380k source-pool entries, with ~1M for the combined pool. The exact numeric facts remain unchanged in `settings.json`; native count tooltips retain exact sample/entry counts. The downloadable overview uses the same compact formatter.
- Removed redundant Fixed labels from validation/test, enlarged scale and section labels, and promoted May overlap / No temporal overlap from small badges to the main composition statements. Source counts and percentages are retained beneath these statements.
- Verified the rendered six split counts, matching desktop row positions, SVG text, and mobile layout. The 400-pixel viewport has a 385-pixel document/scroll width, with no card overflow. Viewport override reset. Changes remain local and unpushed.

## Review revision: explicit construction flow and source mapping

- Replaced the clipped fork dots with complete downward arrowheads and separate Construct Complex data / Construct Selective data labels. Both branches visibly connect the shared source pools to their setting columns. Mobile uses a compact construction label before the stacked settings.
- Rebuilt the reference examples with source brackets and leader lines directly on the original spectrograms. Road noise is labeled Preserve; dog barking is labeled Increase pitch. The bracket spans follow the original PDF annotations and do not assert ground-truth segmentation.
- Added separate, explicitly labeled Source layout / schematic tracks to explain overlapping versus separated sources. These blocks are illustrative; they do not claim measured durations. The original PDF JPEG assets and embedded SVG JPEG payloads remain byte-for-byte unchanged.
- Verified both enlarged reference diagrams, desktop comparison, SVG XML, exact dataset counts, and mobile layout. At a 400-pixel viewport, document and scroll width are both 385 pixels; both cards have equal client/scroll width and all reference images load. Screenshots saved for author review. Changes remain local, uncommitted and unpushed.

## Review revision: explain the overview and its motivation

- Added one concise paragraph immediately below the research overview figure, before the optional evidence details. It connects source-selection failures, the causal early-trajectory interventions, and the two complementary STEER training stages. Wording was checked against the manuscript introduction and early-trajectory intervention section.
- Key findings and stage names are emphasized within a single paragraph. Desktop uses 16-pixel type with 1.9 line height; mobile uses 13-pixel type with 1.85 line height. Versioned the base stylesheet URL so cached previews load the new spacing and typography.
- Verified desktop rendering and a 400-pixel mobile viewport: document/scroll widths are both 385 pixels, and the paragraph fits its 349-pixel content area. Reset the viewport and left the local page on Research. No commit, push, or deployment was performed.

## Approved public GitHub Pages deployment — 26 October 2026

- After local review, the author explicitly approved deployment and changing GeWu-Lab/SteerSound to public. Repository visibility is now public; its default code branch remains `main`.
- Committed and pushed the reviewed website and restored Pages workflow as `fb1673aa9f384ddd426a9e958d76e1f9d6bd3747` on `gh-pages`. The static workflow uploads only `website/`, using checkout v6, configure-pages v5, upload-pages-artifact v4, and deploy-pages v4.
- Enabled the Pages workflow source. The first run was rejected because GitHub's new `github-pages` environment permitted only `main`; added the exact `gh-pages` branch to its allowed deployment branches and reran the failed job. Run `40322830398`, second attempt, completed successfully.
- Confirmed the public URL `https://gewu-lab.github.io/SteerSound/` loads in the browser. All six task tabs initialize, the default gallery renders three removal examples, all three effectiveness model cards render, and the latest research-motivation paragraph is present. No application warnings or errors were captured.
- Twelve production resources returned HTTP 200 and matched local bytes: the HTML, base/results/data stylesheets, application/results scripts, demo/settings JSON, overview raster and downloadable data SVG, and an original/edited audio pair. Saved a screenshot of the live homepage for delivery.

## Updated supplied demos — 26 October 2026

- Compared every supplied instruction and WAV against the published assets. The existing 20 WAVs are unchanged; `replace2` has a new replacement instruction, and `volume2` adds two WAVs. No supplied sample was removed.
- Regenerated the 12-example manifest and paired display waveforms; copied all 22 WAVs byte for byte. Verified source instructions, task grouping, 16 kHz mono 16-bit PCM, durations, 160 waveform values per audio, and bounded finite waveform amplitudes. Each of the six tasks now has two examples.
- Verified the corrected Piano → violin card under Replace and both Volume cards in the local browser. Both new 5.203-second recordings load to readyState 4 and play without errors. JavaScript syntax and diff whitespace checks passed.
- Checked the 400-pixel viewport: document and scroll widths are both 385 pixels, with no card overflow. Reset the viewport; no application warnings or errors were captured.
- Published commit `9898f6802712d2a596b38387e1d26d089a52fdde` on `gh-pages`; Pages run `40327264742` completed successfully. Verified 25 live resources (HTML, app script, manifest, and all 22 WAVs): HTTP 200 and byte-for-byte agreement with local files. The live browser shows all six counts as two, the total as 12, and the new Volume example with both audio sources. Saved a live screenshot for delivery.
