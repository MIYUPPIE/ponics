"""Write main.bbl (natbib numeric, unsrt order) from refs.bib, because arXiv does not run
BibTeX and this machine has no TeX. Entries in citation order of main.tex."""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).parent


def parse_bib(text: str) -> dict:
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        kind, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        fields = {k.lower(): v.strip() for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body)}
        out[key] = (kind, fields)
    return out


def authors(s: str) -> str:
    names = []
    for a in s.split(" and "):
        last, _, first = a.partition(",")
        names.append(f"{first.strip()} {last.strip()}".strip())
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + ", and " + names[-1]


def item(key: str, kind: str, f: dict) -> str:
    a, t, y = authors(f.get("author", "")), f.get("title", ""), f.get("year", "")
    doi = f"\n\\newblock \\doi{{{f['doi']}}}." if "doi" in f else ""
    if kind == "article":
        vol = f.get("volume", "")
        num = f"({f['number']})" if "number" in f else ""
        pages = f":{f['pages']}" if "pages" in f else ""
        body = f"\\newblock {t}.\n\\newblock \\emph{{{f['journal']}}}, {vol}{num}{pages}, {y}."
    elif kind == "book":
        ed = f", {f['edition']} edition" if "edition" in f else ""
        body = f"\\newblock \\emph{{{t}}}{ed}.\n\\newblock {f.get('publisher', '')}, {f.get('address', '')}, {y}."
    elif kind == "phdthesis":
        body = f"\\newblock \\emph{{{t}}}.\n\\newblock PhD thesis, {f.get('school', '')}, {y}."
    else:  # misc / arXiv
        eprint = f"arXiv:{f['eprint']}" if "eprint" in f else ""
        body = f"\\newblock {t}.\n\\newblock {eprint}, {y}."
    return f"\\bibitem{{{key}}}\n{a}.\n{body}{doi}\n"


def main() -> None:
    bib = parse_bib((HERE / "refs.bib").read_text())
    tex = (HERE / "main.tex").read_text()
    order = []
    for grp in re.findall(r"\\cite[pt]?\{([^}]*)\}", tex):
        for k in grp.split(","):
            k = k.strip()
            if k not in order:
                order.append(k)
    missing = [k for k in order if k not in bib]
    if missing:
        raise SystemExit(f"cited but not in refs.bib: {missing}")
    unused = [k for k in bib if k not in order]
    lines = ["\\providecommand{\\doi}[1]{\\href{https://doi.org/#1}{doi:#1}}",
             f"\\begin{{thebibliography}}{{{len(order)}}}", ""]
    lines += [item(k, *bib[k]) for k in order]
    lines.append("\\end{thebibliography}")
    (HERE / "main.bbl").write_text("\n".join(lines) + "\n")
    print(f"main.bbl: {len(order)} references" + (f"; not cited: {unused}" if unused else ""))


if __name__ == "__main__":
    main()
