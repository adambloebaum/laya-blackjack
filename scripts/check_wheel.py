"""Check that distributable wheels include the dashboard and pinned release contract."""

from pathlib import Path
from zipfile import ZipFile

for path in Path("dist").glob("*.whl"):
    with ZipFile(path) as wheel:
        names = set(wheel.namelist())
        expected = {
            "blackjack/static/index.html", "blackjack/static/app.js", "blackjack/static/style.css",
            "blackjack/model_release.json", "blackjack/releases.py", "blackjack/evaluation.py",
        }
        assert expected <= names, f"Missing wheel assets: {expected - names}"
        assert not any(name.startswith("artifacts/") for name in names)
        print(f"Verified dashboard and release assets in {path.name}")
        break
else:
    raise SystemExit("No wheel found. Run uv build --wheel first.")
