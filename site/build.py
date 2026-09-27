"""Собирает index.html: вставляет data.json и фото команды в template.html (один самодостаточный файл)."""
import base64
import re
from pathlib import Path
here = Path(__file__).parent
t = (here / "template.html").read_text(encoding="utf-8")
d = (here / "data.json").read_text(encoding="utf-8").replace("</", "<\\/")


def photo(m):
    f = here / "team" / f"{m.group(1)}.jpg"
    return "data:image/jpeg;base64," + base64.b64encode(f.read_bytes()).decode() if f.exists() else ""


t = re.sub(r"__PHOTO_(\w+)__", photo, t)
(here / "index.html").write_text(t.replace("__DATA__", d), encoding="utf-8")
print("index.html written")
