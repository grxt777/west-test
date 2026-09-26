"""Собирает index.html: вставляет data.json в template.html (один самодостаточный файл)."""
from pathlib import Path
here = Path(__file__).parent
t = (here / "template.html").read_text(encoding="utf-8")
d = (here / "data.json").read_text(encoding="utf-8").replace("</", "<\\/")
(here / "index.html").write_text(t.replace("__DATA__", d), encoding="utf-8")
print("index.html written")
