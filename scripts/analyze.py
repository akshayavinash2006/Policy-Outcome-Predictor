"""Statistical profile of the evidence base -> artifacts/analytics.md"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pome import config as C  # noqa: E402
from pome.analytics import run_all, to_markdown  # noqa: E402
from pome.data import load_records  # noqa: E402

df = load_records()
md = to_markdown(run_all(df), df)
C.ARTIFACTS.mkdir(exist_ok=True)
(C.ARTIFACTS / "analytics.md").write_text(md, encoding="utf-8")
print(md)
