"""Scan the repository before every commit for content that must not be published.

Checks (all text files tracked by git, or all files if git is unavailable):
  1. terms listed in the git-ignored file .confidential_terms (one per line), e.g.
     company and customer names, plant and tool codes, internal hold reason codes;
  2. data files (xls, xlsx, pkl, parquet, csv outside tests) and anything under data/ or results*/;
  3. absolute local paths, e-mail addresses, IP addresses, and compact timestamps.
Exit code 1 if anything is found. Use as a pre-commit hook:
    python tools/check_confidentiality.py
"""
import os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED_CSV_DIRS = ("tests/", "paper_results/", "paper_output/")  # aggregated, anonymised results only
PATTERNS = {
    "absolute path": re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/][^\s'\"]+|/Users/[^\s]+|/home/[A-Za-z0-9_]+/"),
    "e-mail": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "IP address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "compact timestamp": re.compile(r"\b20\d{6}[ _]?\d{6}\b"),
}
ALLOW_EMAIL = ("noreply@github.com",)


def files():
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
        return out
    except Exception:
        return [os.path.relpath(os.path.join(d, f), ROOT) for d, _, fs in os.walk(ROOT) if ".git" not in d for f in fs]


def main():
    terms = []
    tf = os.path.join(ROOT, ".confidential_terms")
    if os.path.exists(tf):
        terms = [t.strip() for t in open(tf, encoding="utf-8") if t.strip() and not t.startswith("#")]
    problems = []
    for f in files():
        if f.startswith(("data/", "results")) or f == ".confidential_terms":
            problems.append((f, "data or result file must not be committed")); continue
        if f.endswith((".xls", ".xlsx", ".pkl", ".parquet")) or (f.endswith(".csv") and not f.startswith(ALLOWED_CSV_DIRS)):
            problems.append((f, "data file")); continue
        if f.endswith((".png", ".jpg", ".pdf", ".pptx", ".docx", ".zip")):
            continue
        try:
            text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for t in terms:
            if re.search(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])", text, re.IGNORECASE):
                problems.append((f, f"confidential term: {t[:2]}***"))
        if f == "tools/check_confidentiality.py":
            continue
        for name, p in PATTERNS.items():
            for m in p.findall(text):
                if name == "e-mail" and m in ALLOW_EMAIL:
                    continue
                problems.append((f, f"{name}: {m[:40]}"))
    for f, why in problems:
        print(f"[BLOCK] {f}: {why}")
    print("confidentiality check:", "FAILED" if problems else "passed", f"({len(terms)} local terms checked)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
