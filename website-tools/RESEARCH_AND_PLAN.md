# SteerSound project page: research, plan, and content map

Prepared 26 October 2026. The implementation lives on `gh-pages`; the scientific code remains available on `main`.

## Reading and design research

Read the complete current manuscript, including appendices A–I, its main result tables, experimental protocols, and original figure assets. The current manuscript is a preprint, with no arXiv identifier supplied. The page neither invents an identifier nor advertises conference acceptance.

The project page is an original, buildless implementation. The following pages informed the design research; their source code and scientific claims were not copied:

| Reference | Observation | Application here |
| --- | --- | --- |
| [Gestalt](https://gewu-lab.github.io/Gestalt/) | Clear lab identity, restrained purple palette, immediate resource links | Lab logo, local Manrope font, concise author/resource header |
| [ROMA](https://gewu-lab.github.io/ROMA/) | Persistent section navigation; project-specific diagrams and empirical detail | Compact sticky navigation; substantive method, data, and result sections |
| [SAO-Instruct](https://eth-disco.github.io/sao-instruct/) | Instructions sit directly beside before/after examples | Display exact supplied instructions with original/output audio |
| [FreeSonic](https://free-sonic.github.io/) | Task-specific audio and visual comparisons | Six task filters, actual waveform previews, paired listening |
| [SmartDJ](https://zitonglan.github.io/project/smartdj/smartdj.html) | Clear progression from problem to method, data, and demonstrations | Distinct scientific narrative beneath the immediate listening experience |
| [UNISON](https://lizhaoqing.github.io/UNISON-demo/) | Broad capability coverage and task-wise result presentation | Six-operation gallery, detailed task metrics, explicit supported-task caveats |
| [Audio-Omni](https://zeyuet.github.io/Audio-Omni/) | Unified resource entry points and navigation through audio tasks | Code/model/data/inference resources; focused task browsing |
| [Nerfies](https://nerfies.github.io/) | Concise academic identity followed by concrete demonstrations | Demonstrations precede detailed technical exposition |
| [Lyria](https://deepmind.google/models/lyria/) | Audio is the central material visitors explore | Prominent interactive audio rather than unrelated decorative imagery |

Design conclusion: a clean white and off-white page with plum/teal accents from the paper. Real audio and original paper figures carry the page. No stock images, artificial demo data, autoplay, analytics, or runtime CDN dependencies.

## Agreed execution plan

1. Paper identity, full author/affiliation list, actual resource links, and a clearly unavailable paper link.
2. A featured source/output comparison that preserves the playback position when switching versions.
3. Eleven supplied examples across six operations, with seekable waveform controls and downloads.
4. Scientific motivation and early-trajectory intervention evidence.
5. Separate explanations and figures for STEER-Emphasis and STEER-Recovery; optional architecture detail.
6. Results follow two questions: overall SteerSound capability, then STEER effectiveness. Dataset switching is nested inside the effectiveness block, with all three backbones shown together and explicit metric definitions.
7. Responsive layout, accessible keyboard controls, figure enlargement, static asset verification, and browser QA. Prepare revisions locally for author review, push only after approval, and do not deploy.

## Scientific content map

| Page section | Manuscript source |
| --- | --- |
| Headline and author list | Title block |
| Problem and source selection | Abstract; Sections 1 and 3; Figure 1 (`figure200.pdf`) |
| Corrective/corruptive prefixes | Section 3.2; Figure 3 (`figure102.pdf`); 100 validation examples per task |
| Emphasis | Section 4.2; Figure 4 (`figure53.pdf`); Appendix A |
| Recovery | Section 4.3; Figure 5 (`figure71.pdf`); Algorithm 1 |
| Model architecture | Section 5.1; Figure 6 (`figure72.pdf`); Appendix B |
| Data settings | Figure 7 (`figure54.pdf`); Appendices C–D |
| Selective results | Table 3; Appendix D criteria |
| Complex results | Primary metrics from Table 1; Table 2 TBG |
| Cross-model results | Tables 2–3; different supported task sets explicitly noted |
| Early-corruption robustness | Figure 8 (`figure503.pdf`); Appendix F.1 |
| Human evaluation | Figure 9 (`figure501.pdf`); 100 sampled examples per task; Appendix F.2 |
| Speaker-task extension | Section 5.6; Figure 11; Appendix H; separately adapted models |

Derived headline: `(28.48 - 8.45) / 28.48 = 70.33%` relative selective error reduction for our editor. It is not a 70 percentage-point decrease. Overall success and selective error use different task sets. Overall benchmark success pools valid sample counts; pitch-unmeasurable examples are excluded.

## Audio provenance

All published WAVs are byte-for-byte copies of supplied files. There is no resampling, compression, loudness normalization, model inference, or synthetic baseline generation. The waveform previews use windowed RMS; each pair shares a display scale. All files are 16 kHz, mono, 16-bit PCM.

| Supplied folder | Published task | Original | SteerSound output |
| --- | --- | --- | --- |
| remove1 | Remove | source.wav | edited.wav |
| remove2 | Remove | source.wav | edited.wav |
| replace2 | **Remove**, matching its actual instruction | source.wav | edited.wav |
| replace1 | Replace | source.wav | edited.wav |
| add1 | Add | reference_segment.wav | steersound_result 2.wav |
| add2 | Add | source.wav | edited.wav |
| pitch1 | Pitch | source.wav | edited.wav |
| pitch2 | Pitch | source.wav | target.wav — confirmed as model output by the author |
| volume1 | Volume | source.wav | edited.wav |
| generate1 | Generate | No input | edited.wav |
| generate2 | Generate | No input | edited.wav |

## Resource state checked before implementation

- GitHub repository exists; default branch is `main`; a new `gh-pages` branch was created from it.
- The repository was private and had no GitHub Pages configuration. Repository visibility is preserved.
- The supplied HTTP inference endpoint returned the SteerSound interface. It opens as an external top-level page to avoid mixed-content embedding.
- The Hugging Face dataset returned HTTP 200.
- The supplied weights URL returned HTTP 401 without a logged-in Hugging Face session. Its supplied address is preserved; no release availability is invented.
- Paper and BibTeX remain unavailable until a real public release exists.

## Assets and implementation

- Paper PDF figures are rendered to high-resolution WebP for the page; the original manuscript and TeX sources are not published.
- The GeWu-Lab logo is sourced from the lab's supplied ROMA project page.
- Manrope is self-hosted from the lab's Gestalt page; the SIL Open Font License is retained in `website/assets/fonts/OFL-Manrope.txt`.
- All page code and the SteerSound graphic mark are original to this implementation.
- Only `website/` is needed for static hosting; training code and project notes are not part of the site.
- Following author review, updates are first prepared locally and pushed only after approval. Pushing does not deploy the website; the initial automatic deployment workflow has been removed locally for the next approved push.

## Review revision: present STEER as percentage improvements

- Complex cards show Emphasis and Recovery/full STEER as two cumulative percentage gains over each original pretrained model. All values carry an explicit plus sign and percentage unit; the original-model zero row and explanatory banner are omitted.
- These percentages are the paper's Task-Balanced Gain, defined as 100 times the geometric-mean task metric ratio minus one. Full-STEER percentages are cumulative relative to the original model, rather than increments over Emphasis.
- Detailed results retain the original task scores, with the original model marked as the reference in the gain columns. The extra source-removal score strip is omitted from the main cards.
- Selective cards retain actual success and selective error rates for the original model, Emphasis, and full STEER. The overall-capability comparison is unchanged.
- Both expanded training-stage tables use grouped headers for Generate, Add, Replace, Remove, Pitch, and Volume. Selective reports task-wise success and, for the four selective operations, selective error. Aggregate columns follow the six task groups in both settings.

## Review revision: concise presentation

- The research question is presented through a centered title and the overview figure, without the three surrounding introductory paragraphs.
- Supplementary waveform notes, visible figure captions, and long table footnotes are omitted from the page at the author's request. Audio provenance and evaluation details remain in these project notes; metric units are explicit in table headers, with full accessible table descriptions retained.

## Review revision: keep each dataset's scale, composition, and example together

- The data section uses two complete dataset columns, Complex and Selective. Each column presents scale, source composition, and its corresponding reference example in that order. This replaces the earlier split sections, which separated dataset examples from scale.
- Complex uses approximately one million raw audio entries for online training-pair construction, with a nominal training length of approximately 600,000 samples (manuscript appendix, Online Data Constructor). Selective has exactly 61,440 fixed training samples (10,240 per task). The scale comparison now reports training samples for both settings, explicitly distinguishing nominal online length from a fixed sample collection. Each setting has 6,000 validation and 6,000 test examples.
- Composition text distinguishes source counts from temporal overlap: Complex has one to four sources that may overlap; Selective has exactly two temporally separated sources for the four selective editing tasks. The common pitch-editing instruction connects the two spectrogram examples.
- Scale counts and the exact source-count percentages from the paper are rendered as HTML. Separate Complex and Selective spectrograms keep the examples attached to their respective datasets. The latest redesign extracts the original JPEGs intact from the PDF and redraws annotations around them, replacing the earlier raster crops. No measured image pixels, results, source percentages, or timestamps were synthesized.
- Both columns are labeled as data settings: Complex setting / Selective setting, with parallel audio-editing-data subtitles. Navigation is Data; effectiveness tabs and surrounding page copy use the same setting names. The page no longer contrasts an entire dataset with a benchmark-only collection.

## Research-led data redraw

- A new reference review and implementation plan was written before source changes in `DATA_REDESIGN_PLAN.md`. The review covered Gestalt, ROMA, SAO-Instruct, SmartDJ and UNISON, with visual checks of the first three.
- The two source pools and their shared use are now shown upstream of both settings: 635,019 class-labeled entries plus 380,165 fine-captioned entries. The manuscript confirms both settings reuse the same pools and source-level data partition.
- The comparison uses one coherent frame, with fixed columns for the two settings. Matching rows contain three scale values, source-count distributions, and the real example with its exact editing instruction. Target and non-target roles use consistent colors and readable native labels.
- `settings.json` is the single data source for the static HTML and standalone vector figure. `build_data_figures.py` embeds the unmodified PDF JPEGs and the local font in the exported SVGs; the main page uses the same JPEGs with responsive HTML labels. The normal asset-preparation script calls this generator too.

## Updated supplied demos — 26 October 2026

- The author updated `demo/replace2/instruction.txt` to “Replace the sound of piano with the sound of violin.” This supersedes the original removal instruction and earlier classification notes. The sample now belongs to Replace and is titled Piano → violin. Its supplied WAVs are unchanged.
- Added `volume2`: “Increase the volume of the sound of background music by 5 dB.” Both supplied WAVs are copied without audio processing; display waveforms use the existing shared scale for each pair.
- The gallery now contains 12 samples and 22 WAVs, with two examples for each of the six tasks. The source generator supports `--audio-only`; page counts initialize from the manifest, and the manifest request revalidates cached data.
