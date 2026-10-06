"""Add Markdown to collected answers with the team's formatting service.

Reads a JSONL file written by scripts/collect_answers.py, sends each plain
answer to the service (POST /format with {"text": ...}; the instruction for
its model lives inside the service) and writes the formatted answers to a new
JSONL file and an HTML table next to it. The service runs on the server and
needs a token; this client does not use a GPU.
"""

import argparse
import html
import json
import urllib.request
from pathlib import Path

from collect_answers import HTML_HEAD

OUTPUTS = Path.home() / "data" / "dpo-qwen" / "outputs"


def format_text(url: str, token: str, text: str) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def write_html(rows: list[dict], path: Path) -> None:
    header = ["uid", "question", "answer", "markdown", "action", "reason"]
    lines = [HTML_HEAD, '<table><colgroup><col class="uid"></colgroup><thead><tr>']
    lines += [f"<th>{name}</th>" for name in header]
    lines.append("</tr></thead><tbody>")
    for row in rows:
        lines.append("<tr>" + "".join(f"<td>{html.escape(str(row[name]))}</td>" for name in header) + "</tr>")
    lines.append("</tbody></table></html>")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUTPUTS / "sberquad_sample.jsonl")
    parser.add_argument("--out", type=Path, default=OUTPUTS / "sberquad_markdown.jsonl")
    parser.add_argument("--limit", type=int, default=15, help="number of answers")
    parser.add_argument("--url", default="http://127.0.0.1:30100/format")
    parser.add_argument("--token-file", type=Path, default=Path("/var/tmp/agkov-markdown-api/token"))
    args = parser.parse_args()

    token = args.token_file.read_text().strip()
    source = [json.loads(line) for line in args.input.read_text().splitlines()[: args.limit]]
    rows = []
    # One request at a time: the service and its API key are shared with the team.
    for i, row in enumerate(source, start=1):
        answer = row["instant"]["answer"]
        result = format_text(args.url, token, answer)
        rows.append(
            {
                "uid": row["uid"],
                # The question is the last paragraph of the prompt; the rest is its context.
                "question": row["question"].split("\n\n")[-1],
                "answer": answer,
                "markdown": result["text"],
                "changed": result["changed"],
                "action": result["action"],
                "reason": result["reason"],
            }
        )
        print(f"{i}/{len(source)}")

    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_html(rows, args.out.with_suffix(".html"))
    print(f"Saved {len(rows)} answers to {args.out} and .html")


if __name__ == "__main__":
    main()
