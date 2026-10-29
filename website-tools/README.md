# Project page maintenance

`website/` is a standalone static site with relative resource paths, so it works at the GitHub Pages project subpath `/SteerSound/`. No package installation or build step is needed.

Preview from the repository root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory website
```

Open `http://127.0.0.1:8765/`. Serving through HTTP is required for the JSON gallery; opening `index.html` directly as a `file:` URL is not supported.

Review workflow: prepare and preview changes locally, wait for the author's review, then push the approved files to `gh-pages`. On 26 October 2026, after reviewing the page, the author authorized making the repository public and deploying with GitHub Pages. This supersedes the earlier private branch-only delivery instruction.

Publication uses `.github/workflows/project-page.yml`: pushes affecting `website/` on `gh-pages` upload only that static directory and deploy it to the `github-pages` environment. The repository is public and Pages uses GitHub Actions. The live URL is `https://gewu-lab.github.io/SteerSound/`; the main code branch stays `main`. The `github-pages` environment must allow the `gh-pages` branch in its deployment branch rules.

First public deployment succeeded on 26 October 2026 from `fb1673a`, workflow run `40322830398` (second attempt, after adding `gh-pages` to the environment's permitted branches). HTML, scripts, data, figures, and sample audio were verified against the reviewed local files.

To refresh paper figures or audio assets, install Pillow and Poppler, then run:

```sh
python3 website-tools/prepare_assets.py --paper-dir /path/to/manuscript --demo-dir /path/to/demo
```

The script maps the actual supplied filenames, preserves WAV bytes, and precomputes the waveform data. See `RESEARCH_AND_PLAN.md` for the source mapping and resolved filename ambiguities.

To refresh only the audio gallery, without rebuilding paper figures:

```sh
python3 website-tools/prepare_assets.py --audio-only --demo-dir /path/to/demo
```

The current gallery has 12 examples, two for each task. The updated `replace2` instruction replaces piano with violin; `volume2` increases background music by 5 dB. Pitch and Volume examples both show Increase before Decrease. Keep the source mapping in `prepare_assets.py` consistent with the supplied instructions. Counts initialize from `demos.json`, whose requests revalidate cached data.

The data section presents shared source pools above Complex and Selective as parallel data settings. Each column keeps its scale, source composition, and reference example together. Facts are stored in `website/data/settings.json`; the generated static HTML sits between the `DATA-STUDY` markers, with styles in `website/data-section.css`.

To regenerate just this comparison and its vector figures, run:

```sh
python3 website-tools/build_data_figures.py --paper-dir /path/to/manuscript
```

This extracts the original JPEG spectrograms intact from Figure 7's PDF and creates two annotated reference SVGs plus `data-settings-overview.svg`, a standalone downloadable figure. It also rewrites the static HTML comparison from the fact manifest. Change facts or rendering in their source files before regenerating. The shared source pools belong to both settings; preserve the distinction between nominal online training length and fixed samples. See `DATA_REDESIGN_PLAN.md` for the reference review and scientific relationships.

Visible counts use a consistent compact k/M notation (for example 61k and 6k). Preserve exact integer values in the manifest; the generator retains exact counts in native tooltips while keeping the page and exported figure concise.

When the paper is public, replace the unavailable paper badge in `website/index.html` and add the actual paper link and BibTeX. Do not invent a conference, DOI, or arXiv number.

Results are organized around two research questions: overall SteerSound capability (Table 1), then STEER effectiveness across model backbones and evaluation settings (Tables 2–3). The data and rendering are in `website/results.js`; the result-specific styles are in `website/results.css`. Keep the metric explanations and different task-coverage caveats when updating results. Resource URLs are in `website/index.html`.
