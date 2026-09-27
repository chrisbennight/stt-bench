"""Download the pinned pyannote snapshot using a token read only from a terminal."""

import getpass
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DownloadRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urllib.parse.urlparse(req.full_url).netloc != urllib.parse.urlparse(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def main():
    repo = "pyannote/speaker-diarization-community-1"
    revision = json.loads((ROOT / "model-revisions.json").read_text())[repo]["revision"]
    token = getpass.getpass("Hugging Face read-only token: ")
    destination = ROOT / "models" / "pyannote"
    destination.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(DownloadRedirect())

    def request(url):
        return opener.open(
            urllib.request.Request(url, headers={"Authorization": "Bearer " + token}), timeout=120
        )

    with request(f"https://huggingface.co/api/models/{repo}/revision/{revision}") as response:
        metadata = json.load(response)
    if metadata["sha"] != revision:
        raise ValueError("The model revision does not match the pinned revision")
    for entry in metadata["siblings"]:
        name = entry["rfilename"]
        target = (destination / name).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError("Snapshot path escapes the model directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        url = f"https://huggingface.co/{repo}/resolve/{revision}/" + urllib.parse.quote(name)
        with request(url) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(target)
    (destination / "snapshot-revision.json").write_text(
        json.dumps({"repo": repo, "revision": revision}) + "\n"
    )
    print("Pinned pyannote snapshot downloaded. The token was not saved.")


if __name__ == "__main__":
    main()
