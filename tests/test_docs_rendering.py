"""Validate GitHub Markdown tables and mathematical notation in README catalogs."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
FILES=[ROOT/"README.md",ROOT/"catalog"/"README_TOY_MODELS.md",
       ROOT/"catalog"/"README_METRICS.md"]

def check_markdown(path):
    text=path.read_text(encoding="utf8")
    if "\\|" in text:
        raise AssertionError(f"{path}: LaTeX \\| can break Markdown table parsing; use \\lVert / \\rVert")
    # The README renderer rejects this macro and falls back to raw LaTeX.
    if r"\operatorname" in text:
        raise AssertionError(f"{path}: unsupported \\operatorname; use simple math operators")
    if "§" in text:
        raise AssertionError(f"{path}: unresolved placeholder")
    if len(re.findall(r"(?m)^\$\$$",text))%2:
        raise AssertionError(f"{path}: unmatched display-math $$ fence")
    if len(re.findall(r"(?m)^\x60\x60\x60",text))%2:
        raise AssertionError(f"{path}: unmatched code fence")
    block=[]
    for i,line in enumerate(text.splitlines(),start=1):
        if line.startswith("|"):
            cols=line.count("|")
            block.append((i,cols,line))
        elif block:
            n=block[0][1]
            for idx,count,content in block:
                if count!=n:
                    raise AssertionError(f"{path}:{idx}: Markdown table broken: {count} pipes, expected {n}\n{content}")
            block=[]
    if block:
        n=block[0][1]
        for idx,count,content in block:
            if count!=n:
                raise AssertionError(f"{path}:{idx}: Markdown table broken: {count} pipes, expected {n}")
    # Math inside tables uses $...$ on one line.
    for i,line in enumerate(text.splitlines(),start=1):
        if line.startswith("|") and line.count("$")%2:
            raise AssertionError(f"{path}:{i}: unmatched inline-math delimiter in table row")
    return True

if __name__=="__main__":
    for filename in FILES:check_markdown(filename)
    print("PASS: GitHub Markdown table columns, math delimiters and code fences")
