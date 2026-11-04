"""Refresh website figures directly from the current manuscript's references.

PDFs are rendered without text overlays, redrawing or scientific-content edits.
The figure labels select the source files, including multiline includegraphics.
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

SITE = Path(__file__).resolve().parents[1] / 'website'
FIGURE_LABELS = {
    'overview': 'figure1',
    'early-selection': 'figure2',
    'interventions': 'figure3',
    'emphasis': 'figure4',
    'recovery': 'fig:ter_supervision',
    'architecture': 'figure6',
    'data': 'figure7',
    'robustness': 'figure8',
    'human-evaluation': 'figure9',
}


def manuscript_figure_sources(paper_dir):
    tex = paper_dir / 'iclr2027_conference.tex'
    source = re.sub(r'(?<!\\)%[^\n]*', '', tex.read_text())
    includes = list(re.finditer(r'\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}', source))
    figures = {}
    for i, match in enumerate(includes):
        stop = includes[i + 1].start() if i + 1 < len(includes) else len(source)
        label = re.search(r'\\label\s*\{([^}]+)\}', source[match.end():stop])
        if label:
            figures[label.group(1)] = match.group(1)
    for label in FIGURE_LABELS.values():
        assert label in figures, 'Figure label missing from current manuscript: ' + label
        assert (paper_dir / figures[label]).is_file(), figures[label]
    return figures


def update_asset_reference(html, filename, width=None, height=None):
    path = SITE / 'assets' / 'figures' / filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    url = 'assets/figures/' + filename + '?v=' + digest
    stem, extension = filename.rsplit('.', 1)
    pattern = r'assets/figures/' + re.escape(stem) + r'(?:-stages)?\.' + re.escape(extension) + r'(?:\?[^"\s]+)?'
    html = re.sub(pattern, url, html)
    if width is not None:
        pattern = r'(<img src="' + re.escape(url) + r'"[^>]*width=")\d+(" height=")\d+'
        html, count = re.subn(pattern, lambda m: m[1] + str(width) + m[2] + str(height), html)
        # data.webp is archived for Figure 7; its source images appear in the data cards.
        assert count == (0 if stem == 'data' else 1), filename
    return html


def refresh_paper_figures(paper_dir):
    sources = manuscript_figure_sources(paper_dir)
    manifest = []
    for name, label in FIGURE_LABELS.items():
        source = paper_dir / sources[label]
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        output = SITE / 'assets' / 'figures' / (name + '.webp')
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / name
            subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '2800', '-png', str(source), str(prefix)], check=True, capture_output=True)
            with Image.open(prefix.with_suffix('.png')) as image:
                image = image.convert('RGB')
                width, height = image.size
                image.save(output, 'WEBP', quality=94, method=6)
                # Confirm the published pixels are exactly the direct-render export.
                check = Path(directory) / 'check.webp'
                image.save(check, 'WEBP', quality=94, method=6)
                assert check.read_bytes() == output.read_bytes()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        manifest.append({'name': name, 'label': label, 'source': sources[label],
                         'sourceSha256': source_hash, 'asset': 'assets/figures/' + output.name,
                         'assetSha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                         'width': width, 'height': height})
    from build_data_figures import build as build_data_figures
    build_data_figures(paper_dir, source_pdf=paper_dir / sources['figure7'])
    page = SITE / 'index.html'
    html = page.read_text()
    for item in manifest:
        html = update_asset_reference(html, Path(item['asset']).name, item['width'], item['height'])
    for filename in ['data-complex-spectrogram.jpg', 'data-selective-spectrogram.jpg',
                     'data-complex-reference.svg', 'data-selective-reference.svg', 'data-settings-overview.svg']:
        html = update_asset_reference(html, filename)
    page.write_text(html)
    (SITE / 'data' / 'paper-figures.json').write_text(json.dumps(
        {'manuscript': 'iclr2027_conference.tex', 'figures': manifest}, ensure_ascii=False, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper-dir', type=Path, required=True)
    for item in refresh_paper_figures(parser.parse_args().paper_dir):
        print(item['name'] + ' <- ' + item['source'] + ' (' + item['label'] + ')')
