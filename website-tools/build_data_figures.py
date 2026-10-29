"""Rebuild the data comparison and editable SVGs from manuscript facts/pixels.

The JPEGs are extracted intact from Figure 7's PDF. No measured spectrogram
pixels, source timings, model results, or audio examples are synthesized.
Separate, explicitly labeled schematic blocks explain temporal overlap.
"""
import argparse
import base64
import html
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'website'
FIGURES = SITE / 'assets/figures'
COLORS = ['#ded8ec', '#8973b1', '#5d4b82', '#342842']
PURPLE, TEAL, INK, MUTED = '#7450a3', '#328c87', '#22222c', '#656572'
# Callout spans follow the original PDF annotations, in normalized image width.
# They identify sources; they are not a ground-truth temporal segmentation.
REFERENCE_SPANS = {
    'complex': {'preserve': (0.0, 1.0), 'target': (0.1306, 0.9464)},
    'selective': {'preserve': (0.0488, 0.4511), 'target': (0.6972, 0.9493)},
}
# Illustrative blocks explain overlap, without asserting measured durations.
SCHEMATIC_LAYOUTS = {
    'complex': {'preserve': [(0, 1)], 'target': [(0.15, 0.30), (0.47, 0.64), (0.79, 0.94)]},
    'selective': {'preserve': [(0.05, 0.45)], 'target': [(0.70, 0.95)]},
}


def esc(value):
    return html.escape(str(value), quote=True)


def compact_count(value):
    if value >= 1_000_000:
        return f'{value / 1_000_000:.0f}M'
    if value >= 1_000:
        return f'{value / 1_000:.0f}k'
    return str(value)


def text(x, y, value, size=20, fill=INK, weight=500, anchor='start'):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{esc(value)}</text>')


def svg_document(body, width, height, font, title, description):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-labelledby="title desc">'
            f'<title id="title">{esc(title)}</title><desc id="desc">{esc(description)}</desc>'
            '<defs><style>@font-face{font-family:Manrope;src:url(data:font/woff2;base64,'
            f'{font}) format("woff2");font-weight:200 800}}'
            'text{font-family:Manrope,Arial,sans-serif}</style></defs>' + body + '</svg>')


def reference_annotations(setting):
    spans = REFERENCE_SPANS[setting['id']]
    road_start, road_end = [max(8, min(992, v * 1000)) for v in spans['preserve']]
    dog_start, dog_end = [v * 1000 for v in spans['target']]
    road_center, dog_center = (road_start + road_end) / 2, (dog_start + dog_end) / 2
    inner_pins = 'M301.5 235V270M504.2 235V270' if setting['id'] == 'complex' else ''
    return (f'<path d="M{road_start:.1f} 45V25H{road_end:.1f}V45M{road_center:.1f} 25L250 0" '
            'fill="none" stroke="#536679" stroke-width="4" stroke-linejoin="round"/>'
            f'<path d="M{dog_start:.1f} 235V270H{dog_end:.1f}V235M{dog_center:.1f} 270L750 300{inner_pins}" '
            f'fill="none" stroke="{PURPLE}" stroke-width="4" stroke-linejoin="round"/>')


def schematic_svg(setting, x=230, y=435, width=742):
    body = ''
    for i, (role, label, color) in enumerate([
            ('preserve', 'Road noise', '#697386'), ('target', 'Dog barking', PURPLE)]):
        row_y = y + i * 51
        body += text(20, row_y + 19, label, 23, color, 700)
        body += f'<rect x="{x}" y="{row_y}" width="{width}" height="23" rx="5" fill="#f1eef5"/>'
        for start, end in SCHEMATIC_LAYOUTS[setting['id']][role]:
            body += f'<rect x="{x+width*start:.2f}" y="{row_y}" width="{width*(end-start):.2f}" height="23" rx="5" fill="{color}"/>'
    return body


def reference_body(setting, image_uri, example):
    body = '<rect x="0" y="0" width="1000" height="540" rx="12" fill="#fff"/>'
    body += text(250, 31, example['nonTarget'] + ' · preserve', 25, '#536679', 750, 'middle')
    body += '<g transform="translate(18,40) scale(.964)">'
    body += f'<image href="{image_uri}" x="0" y="45" width="1000" height="205.6"/>'
    body += reference_annotations(setting) + '</g>'
    body += text(750, 363, example['target'] + ' · increase pitch', 25, PURPLE, 750, 'middle')
    body += '<line x1="18" y1="385" x2="982" y2="385" stroke="#e5e1e9"/>'
    body += text(20, 415, 'SOURCE LAYOUT · SCHEMATIC', 18, MUTED, 750)
    body += schematic_svg(setting)
    return body


def construction_flow_svg():
    return '''<svg class="pool-fork" viewBox="0 0 1000 84" preserveAspectRatio="none" aria-hidden="true"><defs><marker id="construct-complex-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#7450a3" stroke="none"/></marker><marker id="construct-selective-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#328c87" stroke="none"/></marker></defs><path d="M500 0V18" fill="none" stroke="#737080" stroke-width="2.2"/><path d="M500 18H250V76" fill="none" stroke="#7450a3" stroke-width="2.2" marker-end="url(#construct-complex-arrow)"/><path d="M500 18H750V76" fill="none" stroke="#328c87" stroke-width="2.2" marker-end="url(#construct-selective-arrow)"/></svg>'''


def distribution_svg(setting, x, y, width=516):
    values = setting['distribution']
    total = sum(values)
    current = x
    body = ''
    for i, value in enumerate(values):
        if not value:
            continue
        size = width * value / total
        body += f'<rect x="{current:.3f}" y="{y}" width="{size:.3f}" height="18" fill="{COLORS[i]}"/>'
        current += size
    for i, value in enumerate(values):
        if not value:
            continue
        legend_x = x + i * width / 4 if len([v for v in values if v]) > 1 else x
        body += f'<rect x="{legend_x}" y="{y+37}" width="9" height="9" rx="2" fill="{COLORS[i]}"/>'
        body += text(legend_x + 17, y + 47, f'{i+1} source' + ('s' if i else ''), 16, MUTED)
        body += text(legend_x, y + 74, f'{value:g}%', 21, INK, 750)
    return body


def overview_svg(data, image_uris, font):
    body = '<rect width="1240" height="1240" fill="#fff"/>'
    body += text(32, 49, 'Shared audio sources. Two data settings.', 33, INK, 800)
    body += '<rect x="32" y="78" width="1176" height="106" rx="12" fill="#f7f5fa" stroke="#e5e1e9"/>'
    body += text(57, 115, '~1M shared audio entries', 27, PURPLE, 800)
    pool_line = '  +  '.join(f'{compact_count(pool["count"])} {pool["name"].lower()}' for pool in data['pools'])
    body += text(57, 151, pool_line, 20, MUTED)
    body += '<path d="M620 184V208M319 276V208H921V276" fill="none" stroke="#737080" stroke-width="2.2"/>'
    for cx, color, label in [(319, PURPLE, 'Construct Complex data'), (921, TEAL, 'Construct Selective data')]:
        body += f'<rect x="{cx-126}" y="218" width="252" height="30" rx="5" fill="#fff" stroke="#e5e1e9"/>'
        body += text(cx, 239, label, 17, color, 750, 'middle')
        body += f'<path d="M{cx-5} 269L{cx} 277L{cx+5} 269" fill="none" stroke="{color}" stroke-width="2.5"/>'
    for i, setting in enumerate(data['settings']):
        x = 32 + i * 602
        left = x + 25
        accent = PURPLE if i == 0 else TEAL
        body += '<g transform="translate(0,52)">'
        body += f'<rect x="{x}" y="230" width="574" height="920" rx="12" fill="#fff" stroke="#e5e1e9"/>'
        body += text(left, 276, setting['name'], 29, accent, 800)
        body += text(left, 308, setting['purpose'], 18, MUTED)
        body += f'<line x1="{left}" y1="332" x2="{x+549}" y2="332" stroke="#e5e1e9"/>'
        body += text(left, 365, 'DATASET SCALE', 14, MUTED, 800)
        for j, (label, count, kind) in enumerate([
                ('Training', setting['trainingDisplay'], setting['construction']),
                ('Validation', compact_count(setting['validation']), ''), ('Test', compact_count(setting['test']), '')]):
            column = left + j * 175
            body += text(column, 402, label, 17, MUTED)
            body += text(column, 446, count, 32, accent if j == 0 else INK, 800)
            if kind:
                body += text(column, 477, kind, 16, MUTED)
        body += f'<line x1="{left}" y1="502" x2="{x+549}" y2="502" stroke="#e5e1e9"/>'
        body += text(left, 535, 'SOURCE COMPOSITION', 14, MUTED, 800)
        body += text(left, 563, 'Replace · Remove · Pitch · Volume', 16, MUTED)
        body += text(left, 607, setting['layout'], 27, accent, 750)
        body += text(x + 549, 607, setting['sourceCount'], 18, MUTED, 650, 'end')
        body += distribution_svg(setting, left, 633, 524)
        body += f'<line x1="{left}" y1="733" x2="{x+549}" y2="733" stroke="#e5e1e9"/>'
        body += text(left, 768, 'REFERENCE AUDIO', 14, MUTED, 800)
        body += text(left, 801, '“' + data['example']['instruction'] + '”', 19, INK, 650)
        body += f'<g transform="translate({left},827) scale(.524)">' + reference_body(setting, image_uris[setting['id']], data['example']) + '</g></g>'
    return svg_document(body, 1240, 1240, font, 'Two audio editing data settings',
                        'Shared source pools feed Complex and Selective construction. Each setting has its own training, validation, and test data, source composition, and reference example. Spectrograms are original measured images; the separately labeled source-layout strips are schematic, not measured durations.')


def distribution_html(setting):
    bars = ''.join(f'<span style="flex-grow:{value};background:{COLORS[i]}"></span>'
                   for i, value in enumerate(setting['distribution']) if value)
    legend = ''.join(
        f'<div><dt><i style="background:{COLORS[i]}" aria-hidden="true"></i>{i+1} source{"s" if i else ""}</dt><dd>{value:g}%</dd></div>'
        for i, value in enumerate(setting['distribution']) if value)
    return f'<div class="source-distribution" aria-hidden="true">{bars}</div><dl class="source-distribution-legend">{legend}</dl>'


def schematic_html(setting):
    rows = ''
    for role, label in [('preserve', 'Road noise'), ('target', 'Dog barking')]:
        blocks = ''.join(f'<i style="left:{start*100:g}%;width:{(end-start)*100:g}%"></i>'
                         for start, end in SCHEMATIC_LAYOUTS[setting['id']][role])
        rows += f'<span class="layout-track layout-{role}"><strong>{label}</strong><span class="layout-rail" aria-hidden="true">{blocks}</span></span>'
    return f'<span class="source-layout"><span class="layout-title">Source layout <small>schematic</small></span>{rows}</span>'


def reference_html(setting, example, alt):
    ident = setting['id']
    return f'''<button class="figure-zoom annotated-reference" data-figure="assets/figures/data-{ident}-reference.svg" data-title="{esc(setting['name'])}: reference audio" aria-label="Enlarge the {ident.capitalize()} reference example"><span class="reference-source-label reference-top-label"><span class="preserve-source"><strong>{esc(example['nonTarget'])}</strong><small>Preserve</small></span></span><span class="reference-plot"><img src="assets/figures/data-{ident}-spectrogram.jpg" alt="{esc(alt)}" loading="lazy" width="1357" height="279"><svg class="reference-callouts" viewBox="0 0 1000 300" aria-hidden="true">{reference_annotations(setting)}</svg></span><span class="reference-source-label reference-bottom-label"><span class="target-source"><strong>{esc(example['target'])}</strong><small>Increase pitch</small></span></span>{schematic_html(setting)}</button>'''


def render_html(data):
    cards = []
    for setting in data['settings']:
        ident = setting['id']
        cells = ''.join(f'<div><dt>{label}</dt><dd title="{exact:,} samples">{esc(count)}'+(f'<small>{esc(kind)}</small>' if kind else '')+'</dd></div>' for label, count, kind, exact in [
            ('Training', setting['trainingDisplay'], setting['construction'], setting['training']),
            ('Validation', compact_count(setting['validation']), '', setting['validation']),
            ('Test', compact_count(setting['test']), '', setting['test'])])
        alt = ('Dog barking and road noise overlap in the Complex reference. Edit the pitch of dog barking while preserving road noise.' if ident == 'complex' else
               'Road noise and dog barking occur in separate intervals in the Selective reference. Edit the pitch of dog barking while preserving road noise.')
        cards.append(f'''        <article class="dataset-card dataset-{ident}" aria-labelledby="{ident}-data-title">
          <header class="dataset-card-heading"><h3 id="{ident}-data-title">{esc(setting['name'])}</h3><p>{esc(setting['purpose'])}</p></header>
          <section class="dataset-card-row dataset-scale" aria-labelledby="{ident}-scale-title"><h4 class="dataset-row-label" id="{ident}-scale-title">Dataset scale</h4><dl class="scale-metrics">{cells}</dl></section>
          <section class="dataset-card-row dataset-composition" aria-labelledby="{ident}-composition-title"><h4 class="dataset-row-label" id="{ident}-composition-title">Source composition</h4><p class="dataset-task-scope">{' · '.join(data['selectiveTasks'])}</p><h5>{esc(setting['layout'])}<span>{esc(setting['sourceCount'])}</span></h5>{distribution_html(setting)}</section>
          <section class="dataset-card-row dataset-example" aria-labelledby="{ident}-example-title"><h4 class="dataset-row-label" id="{ident}-example-title">Reference audio</h4><p>“Increase the pitch of <strong>dog barking.</strong>”</p>{reference_html(setting, data['example'], alt)}</section>
        </article>''')
    pools = ''.join(f'<div><strong title="{pool["count"]:,} audio entries">{compact_count(pool["count"])}</strong><span>{esc(pool["name"])}</span></div>' for pool in data['pools'])
    return '''      <div class="data-study" id="data-scale-title" aria-label="Shared source pools and two audio editing data settings">
        <div class="source-pools"><div class="pool-total"><p class="small-label">SHARED AUDIO SOURCE POOLS</p><h3>~1M <span>audio entries</span></h3></div><div class="pool-breakdown">''' + pools + '''</div></div>
        <div class="pool-construction-flow">''' + construction_flow_svg() + '''<div class="construction-labels"><div><span>Construct Complex data</span></div><div><span>Construct Selective data</span></div></div><p class="mobile-construction-flow">↓ Construct two data settings</p></div>
        <div class="dataset-comparison" aria-label="Dataset scale, composition, and corresponding examples">
''' + '\n'.join(cards) + '''
        </div>
      </div>
'''


def build(paper_dir):
    data = json.loads((SITE / 'data/settings.json').read_text())
    font = base64.b64encode((SITE / 'assets/fonts/manrope.woff2').read_bytes()).decode()
    images = {}
    with tempfile.TemporaryDirectory() as directory:
        prefix = Path(directory) / 'reference'
        subprocess.run(['pdfimages', '-j', str(paper_dir / 'figures/figure54.pdf'), str(prefix)], check=True, capture_output=True)
        for setting in data['settings']:
            path = prefix.with_name(f'reference-{setting["referenceImageIndex"]:03d}.jpg')
            with Image.open(path) as image:
                assert image.size == (1357, 279), 'Source figure changed; recheck image mapping.'
            (FIGURES / f'data-{setting["id"]}-spectrogram.jpg').write_bytes(path.read_bytes())
            images[setting['id']] = 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode()
            svg = svg_document(reference_body(setting, images[setting['id']], data['example']), 1000, 540, font,
                               setting['name'] + ' reference audio', setting['composition'] + ' ' + data['example']['instruction'] + ' Original spectrogram with source callouts. The separate source-layout bars are schematic, not measured timings.')
            (FIGURES / f'data-{setting["id"]}-reference.svg').write_text(svg)
    (FIGURES / 'data-settings-overview.svg').write_text(overview_svg(data, images, font))
    index = SITE / 'index.html'
    source = index.read_text()
    marker = '      <!-- DATA-STUDY:START -->'
    if marker in source:
        start = source.index(marker)
        end = source.index('      <!-- DATA-STUDY:END -->', start) + len('      <!-- DATA-STUDY:END -->\n')
    else:
        start = source.index('      <div class="dataset-comparison" id="data-scale-title"')
        end = source.index('      <div class="outcomes">', start)
    index.write_text(source[:start] + marker + '\n' + render_html(data) + '      <!-- DATA-STUDY:END -->\n' + source[end:])
    print('Generated two reference SVGs, two original JPEG assets, one standalone overview, and the static data comparison.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper-dir', type=Path, required=True)
    build(parser.parse_args().paper_dir)
