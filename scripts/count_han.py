"""Count collected answers that contain Chinese characters.

Reads a JSONL file written by scripts/collect_answers.py and finds the answers
with Chinese characters that are not in the question itself. Prints their
share overall and per sampling temperature and saves the answers found to a
JSONL file and an HTML table next to the input.
"""

import argparse
import collections
import html
import json
import re
from pathlib import Path

from collect_answers import DATA_DIR, HTML_HEAD

HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "outputs" / "wikiomnia_plain.jsonl")
    parser.add_argument("--variant", default="plain", help="which answer of a row to check")
    args = parser.parse_args()

    total, found = collections.Counter(), collections.Counter()
    hits = []
    with args.input.open() as file:
        for line in file:
            row = json.loads(line)
            answer = row[args.variant]["answer"]
            # Rows collected before --temperatures existed have no temperature; they used 0.
            temperature = row.get("temperature", 0)
            total[temperature] += 1
            if set("".join(HAN.findall(answer))) - set(row["question"]):
                found[temperature] += 1
                hits.append({"uid": row["uid"], "temperature": temperature, "question": row["question"], "answer": answer})

    out = args.input.with_name(args.input.stem + "_han.jsonl")
    out.write_text("".join(json.dumps(hit, ensure_ascii=False) + "\n" for hit in hits))
    lines = [HTML_HEAD, '<table><colgroup><col class="uid"><col class="uid"><col style="width:22%"></colgroup><thead><tr>']
    lines += [f"<th>{name}</th>" for name in ("uid", "temperature", "question", "answer")]
    lines.append("</tr></thead><tbody>")
    for hit in hits:
        answer = HAN.sub(lambda match: f"<mark>{match.group()}</mark>", html.escape(hit["answer"]))
        lines.append(
            f"<tr><td>{hit['uid']}</td><td>{hit['temperature']}</td>"
            f"<td>{html.escape(hit['question'])}</td><td>{answer}</td></tr>"
        )
    lines.append("</tbody></table></html>")
    out.with_suffix(".html").write_text("\n".join(lines))

    for temperature in sorted(total):
        share = found[temperature] / total[temperature]
        print(f"temperature {temperature}: {found[temperature]} of {total[temperature]} answers ({share:.2%})")
    print(f"all: {len(hits)} of {sum(total.values())} answers ({len(hits) / sum(total.values()):.2%})")
    print(f"Saved the answers found to {out} and .html")


if __name__ == "__main__":
    main()
