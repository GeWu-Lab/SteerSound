"""Give the presentation stylesheet a content-addressed public URL."""
import hashlib
import re
from pathlib import Path

site = Path(__file__).resolve().parents[1] / "website"
styles = (site / "presentation.css").read_bytes()
version = hashlib.sha256(styles).hexdigest()[:12]
filename = f"presentation.{version}.css"
(site / filename).write_bytes(styles)

index = site / "index.html"
html, count = re.subn(
    r'href="presentation(?:\.[0-9a-f]{12})?\.css(?:\?[^\"]*)?"',
    f'href="{filename}"',
    index.read_text(),
)
if count != 1:
    raise ValueError(f"Expected one presentation stylesheet, found {count}")
index.write_text(html)
print(f"Presentation stylesheet: {filename}")
