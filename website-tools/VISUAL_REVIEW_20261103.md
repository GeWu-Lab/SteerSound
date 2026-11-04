# SteerSound visual review — 3 November 2026

## References inspected

- [Gestalt](https://gewu-lab.github.io/Gestalt/): clear academic identity and a consistent restrained color palette.
- [ROMA](https://gewu-lab.github.io/ROMA/): persistent section navigation and prominent resource links.
- [Nerfies](https://nerfies.github.io/): a strong title hierarchy, a limited reading measure, and demonstrations close to the project identity.
- [SAO-Instruct](https://eth-disco.github.io/sao-instruct/): readable academic prose, followed by concrete instruction/audio examples.
- [FreeSonic](https://free-sonic.github.io/): task-specific audio examples and direct comparison labels.

The first four references were visually inspected in the browser; FreeSonic was read as an additional audio-project reference. The design below is specific to SteerSound and preserves its established purple/teal identity.

## Findings

The existing page has a clear scientific structure, but regular explanatory text falls to 12–13px, section introductions move between several layouts, and compact labels sometimes compete with the primary copy. Fixed line breaks and long single paragraphs add unnecessary reading effort. Scientific figures, audio examples, data-setting comparisons, and results already provide the needed visual material.

## Implementation plan

1. Establish consistent body, label, heading, and spacing scales, with comfortable desktop and mobile sizes.
2. Place section introductions beneath their titles; keep prose within a readable measure while figures and comparisons use the broader page width.
3. Split the unchanged abstract into problem, method, and evaluation paragraphs. Edit method explanations into short purpose/implementation paragraphs.
4. Refine audio, data, and results typography while retaining scientific values, author metadata, real audio, and the order of the three research disclosures.
5. Check wide desktop, intermediate, and phone layouts, text wrapping, keyboard disclosure controls, audio playback, result tabs, and figure enlargement. Publish through the existing gh-pages workflow with the single-root history requested by the author.

## Validation

- Inspected the hero, listening gallery, research disclosures, method stages, data comparison, result tables/cards, and resource/footer typography.
- Checked 1800, 1280, 1100, 900, 400, and 360px viewports. Page and content-card bounds were measured; the narrow task-tab overflow was corrected.
- Checked final-line word wrapping in main explanatory paragraphs; the identified one-word last lines were resolved.
- Confirmed that the abstract remains the exact 179-word current manuscript text, now in three paragraphs, and that authors/affiliations and disclosure order are preserved.
- Played the supplied 10-second violin generation output and checked task switching, Increase-before-Decrease pitch order, both effectiveness settings, the horizontally scrolling task table, keyboard disclosures, and figure enlargement.
- Audio files, figures, numeric result data, and application JavaScript are unchanged by this visual revision.

## ROMA-led research reader revision

The author selected ROMA as the primary visual reference. Its Overview was inspected directly: the title and prose share a stable column, body copy is 17.5px with 1.65 line height, and distinct content blocks establish a clear hierarchy. ROMA itself uses natural left alignment. This revision adopts its consistent framing and adds desktop justification specifically to address the requested abstract alignment.

The previous expanded research row spanned 1185px while its paragraphs used a 761px column anchored on the left. The revised disclosures share a centered frame up to 860px wide, with matching title and paragraph insets. The abstract paragraphs now share both boundaries; phone layouts retain natural left alignment. Paragraph spacing, line height, and disclosure controls are consistent across Abstract, Observation, and Evidence. STEER-Emphasis and STEER-Recovery stay together on a line, and the abstract wording is unchanged.

The expanded abstract was visually checked on desktop and phone. Bounds were measured at 1280, 1000, 800, 400, and 360px viewports: no page or panel overflow, identical paragraph left/right boundaries, desktop justification, and phone left alignment. Keyboard disclosures and figure enlargement were checked. The order of the three disclosures is unchanged.

## Whole-section grid correction

The author's full-section screenshot identified the remaining issue: the overview figure used the page width, while the motivation and disclosures used different centered widths. The preceding narrow-reader revision is superseded here.

The Research heading is now left aligned, and the heading block, overview figure, motivation paragraph, disclosure container, and all three disclosure cards share the page grid. The centered 860px limits and motivation insets were removed. Expanded disclosure titles and text retain their consistent internal padding, paragraph spacing, desktop justification, and phone left alignment.

Measured the seven outer blocks together. At a 1265px document width all extend from x=40 to x=1225. Measurements at 1785, 785, 385, and 345px document widths also show identical left and right boundaries, in both expanded and collapsed states, with no page or disclosure overflow. Reviewed the entire section, rather than just the expanded abstract. Scientific text, figures, and disclosure order are unchanged.

## Content-addressed stylesheet delivery

A subsequent screenshot still showed the preceding centered narrow-column layout. Read-only HTTP checks of the canonical public entry and a versioned entry both returned the current full-width stylesheet reference; the public stylesheet exactly matched the local source, with GitHub Pages advertising a 600-second cache lifetime. The existing client tab could not be inspected through the available browser handle, so its specific cache state was not confirmed.

The presentation stylesheet now also has a content-addressed public filename, generated by `version_styles.py`. The source and generated file are byte-identical. This avoids reusing an earlier stylesheet URL on future edits, and a versioned HTML entry is available for fresh reviews. Legacy source assets are retained for older documents. No layout, scientific content, audio, or results are changed by this delivery revision.

A fresh local review at a 1785px document width loaded the fingerprinted stylesheet. The figure, motivation, disclosure container, and all three disclosure cards each measured 1360px wide, with identical left and right boundaries and no page overflow.


## Concise method copy grounded in the current paper

Re-read the current manuscript's motivation, early-state observations, bidirectional prefix interventions, and both STEER training stages. The Method headline now states the goal directly: “Reliable source selection with STEER.” Its subtitle distinguishes Emphasis (reducing early source-selection errors) from Recovery (learning to correct remaining trajectory deviations).

The Emphasis heading follows the author's requested meaning: “Prioritize task-critical early states.” Its short body explains high-noise training coverage and intervention-derived ranges. Recovery explains corrective supervision on self-generated early rollout states. Sampling ranges, objectives, rollout progression, figures, and numeric results retain their manuscript meaning. Technical detail is carried by the existing range list, diagrams, and equation.

The editorial change is confined to the Method introduction and the two stage explanations. The rest of the HTML, styles, abstract, metadata, figures, audio, and result data remain unchanged. Desktop and phone layouts were checked for clear heading wraps and absence of horizontal overflow.


## Restrained typography within the retained page blocks

Reviewed the current page against ROMA, Gestalt, and Apple's AirPods Pro and MacBook Pro pages. The author chose Apple as the stronger visual direction, then explicitly requested retaining the original block structure. The final revision preserves the section backgrounds, boxed training flow, separate Research disclosure cards, paired data settings, and audio/results cards. It refines the content and internal composition within those blocks.

Section headings use a consistent, quieter type scale. Introductions and repeated result summaries are shorter. Method stage markers are compact inline labels; their headings use the requested “Emphasize” and “Recover” wording. Each stage has one short visible explanation and an optional Training details disclosure. The Emphasis figure and task ranges share the visual column; Recovery's objective and rollout schedule remain available in its disclosure. Research heading, figure, motivation and disclosure cards retain the same outer grid.

The full page was visually reviewed: hero, listening gallery, Research, both Method stages, data comparison, results and resources. The revised internal grids and typography were checked at 1800, 1280, 800, 400 and 360px, with no page or card overflow. After restoring the requested original block frames, the desktop render was checked again. The viewport capability stopped applying size changes during the final additional mobile recheck; the earlier responsive checks cover the revised content grids. Audio playback, Increase-before-Decrease ordering, keyboard training disclosures, result-setting tabs and figure enlargement were exercised.

Author/affiliation metadata, the exact abstract, the complete data comparison, scientific figure sources, resource destinations and numeric result arrays were compared to the preceding release and remain unchanged. Results JavaScript passes a syntax check. The stylesheet uses a new content-addressed filename; previously published CSS remains available for older documents.

Design references: https://www.apple.com/airpods-pro/ · https://www.apple.com/macbook-pro/ · https://gewu-lab.github.io/ROMA/ · https://gewu-lab.github.io/Gestalt/


## Always-visible training explanations

Following the author's final instruction, both Training details controls are now static content blocks with small headings. Their explanations and Recovery formula are visible on initial load; neither block can collapse. The existing page blocks and revised Method composition are retained.


## Restored explanatory copy

The author requested restoring the explanations shortened in the visual pass. Restored the original full Research motivation and the preceding listening, data, results, resources and disclosure wording. Method body copy also follows the preceding release; the explicitly requested Emphasize / Recover headings and always-visible Training details are retained. Block boundaries and the internal visual refinements remain in place.


## Original method figure proportions and expanded overview explanation

Restored the preceding Method column proportions: 0.9/1.1 for Emphasis, equal columns for Recovery, and a 52px desktop gap. At a 1265px document width, the figure frames measure 623px and 566.5px respectively. The original responsive single-column rules, retained page blocks and always-visible training explanations are unchanged.

Expanded the Research overview caption into two paragraphs that follow the figure from left to right: common selection failures; corrective/corruptive evidence that early states shape the final edit; and the complementary roles of Emphasis and Recovery in producing SteerSound. Rechecked the manuscript's observations, interventions and STEER descriptions. Visually reviewed the full Method area and overview with its explanation. The figure and caption share the same outer grid, with no horizontal overflow.


## Data-setting heading alignment

Centered each setting's heading and subtitle beneath its construction label and arrow. Both columns now share their own visual axis. Browser text-range measurements at a 1265px document width put the Complex label/title/subtitle centers at x=336.75px, and the Selective centers within 0.5px (its existing 1px column divider accounts for the difference). Visually inspected the source-pool, fork, labels, headings and subtitles together. Data values, reference examples and block structure are unchanged.


## Natural research narrative below the overview

Removed explicit left/middle/right-panel narration. The two paragraphs now follow the paper's reasoning: selective-editing failures; early emergence and persistence; bidirectional intervention evidence; complementary Emphasis and Recovery stages; and SteerSound's unified capability with more reliable selection. No layout, figure, metric, data or author changes are included.


## Sampling figure with matched task captions

Corrected the caption order to match the source image: Generate/Add; Remove; Replace/Pitch/Volume. The unchanged scientific image and its three descriptions now share one figure frame. Captions follow the source image's actual subplot centers (525, 1361, 2320 of 2800px), rather than an unrelated unequal-width text grid. Task names and primary training-range values share common vertical positions.

Measured the three caption centers against the corresponding rendered subplot centers at a 1265px document width: all differ by less than 0.01px. Checked a native 360px-wide iframe (345px document content) because the browser viewport override had been unreliable: the three captions remain paired with the plots, all primary values stay on one line, there is no horizontal page overflow, and the figure-enlargement control works. The temporary preview document was removed before release. The source image, scientific values, method copy, and always-visible training explanations are unchanged.


## Sunset at the closing horizon

Added an original native SVG sunset between the resource cards and footer: a warm peach/coral disk, lavender horizon, soft transparent glow and quiet reflected ripples. It serves as the author's requested visual metaphor of looking ahead, without adding copy or changing resource links. The image is decorative, excluded from the accessibility tree, static and non-interactive.

Reviewed the entire closing section on desktop and in a native 360px iframe. The artwork is 168px high on desktop and 142px on phone; its clipped wide scene keeps the sun legible without horizontal page overflow. All four resources remain accessible. Existing footer metadata, scientific content, audio and figures are unchanged. The SVG parses successfully and has no external dependencies. The temporary phone-review file was removed before release.


## One integrated architecture card

Unified the editor-architecture disclosure into a single white card containing its title/parameter header, diagram and explanation. Removed the diagram's separate inner frame, reduced the image/text gap, and gave both a shared inset and background. The card remains within the existing Method section. Scientific wording, parameters and the source image are unchanged.

Reviewed the expanded desktop card at 1265px document width. A native 360px iframe (345px document content) shows one stacked card with a 210px diagram and visible parameter metadata, without horizontal overflow. Checked keyboard collapse/expand and figure enlargement. The temporary phone-review document was removed before release.


## Task coverage without numeric adjustment limits

Removed “up to ±4 semitones” and “up to ±5 dB” from the architecture explanation as requested. The paragraph retains all six supported task types. No layout, source asset, metric or implementation changes are included.


## 4 November: full stage names throughout the page

Applied the author's two sentence edits: added “However,” before the existing-editors sentence and removed “can” from the early-error finding. Normalized stage references to STEER-Emphasis / STEER-Recovery in prose, flow labels, metadata, accessibility descriptions, both result settings, and task-wise tables. The formal STEER acronym expansion remains unchanged.

Created named-label figure variants from the manuscript's vector PDFs for the overview, Recovery supervision, robustness and human evaluation. Full STEER remains the cumulative two-stage result. The manuscript sources and old published images remain untouched. Checked protected numeric/plot regions pixel-for-pixel against source renders before WebP encoding and inspected all four outputs. Figure URLs and the results script use fresh content hashes.

Executed both result-setting renderers without a browser: all three model cards and detailed tables use the full names, and every result array matches the preceding release. Checked the two requested sentences, formal acronym, local resources, author metadata and dataset content. Browser preview is disabled in the user's Work settings, so no new whole-page visual check is claimed.


## 4 November: current manuscript originals supersede renamed figure variants

At the author's explicit direction, checked the live manuscript's includegraphics references and figure labels, then re-exported its original figures 1–9 directly. Mapping: overview/figure200; observation/figure103; interventions/figure102; sampling/figure53; Recovery supervision/figure71; base architecture/figure72; data/figure54; robustness/figure503; human evaluation/figure501. Recovery's current original heading is “Self-Generated Rollout Supervision.”

Removed the preceding custom label-overlay helper and its four image variants. The active webpage now uses original manuscript image text, while the previously requested full stage names remain in webpage prose, flow and result tables. Regenerated the data references from the PDF actually cited as Figure 7, retaining the approved data-card layout and facts. Only the Recovery original differs in bytes from the previous original exports; the other original figure and data exports already matched the current PDFs.

The new refresh helper resolves files by manuscript labels, uses no overlay or scientific-image edits, verifies direct-render WebP exports and source hashes, and records a source/asset manifest. Inspected a contact sheet of all nine current originals and checked the current original Recovery diagram. Browser preview remains disabled in Work settings; no new browser layout check is claimed. Audio, authors, body text, experimental result arrays and presentation styles are preserved.
