"""Replay a documentation page the way a reader would, and report what no longer works.

Every page under docs/ is a transcript: it prints commands and the output they produce.
That output rots silently — a message is reworded, a flag is added, a count changes — and
a doc that lies is worse than a doc that is missing, because a reader trusts it.

So the pages are executable. This saves the files a page tells you to save, runs the setup
it gives, then runs every `$`-prefixed command and compares what came back against what the
page claims came back.

    python3 tests/replay_docs.py docs/graph.md

It honours the conventions the pages already use, rather than asking the pages to change
for the harness: a file named in prose ("Save as `probe.py`:") or in the block's first line,
`/.../` standing in for a volatile absolute path, `[ ... elided ]` for a long block, and the
`(exit status 1)` and `exit=N` annotations. Timestamps, durations and run ids are normalised
because none of them can reproduce.

What it cannot check, and why each page still needs a human:

  * `echo $?` — every command runs in its own shell, so the exit code of the previous one
    is already gone. The pages that print it are checked by hand.
  * A page whose setup is described in prose rather than shown. docs/expressions.md says
    outright that each variant is `cp -r /tmp/jig-expr-demo <path>` plus one named edit;
    that reads better than four lines of shell, and it is not replayable.
  * `jig` on PATH, as docs/building.md uses. Correct for an installed reader, invisible here.

Exit status is the number of commands that did not reproduce, so this is usable as a gate.
"""

import subprocess, sys, pathlib, os, re, tempfile, shutil

doc = pathlib.Path(sys.argv[1]).resolve()
root = pathlib.Path(__file__).resolve().parent.parent
lines = doc.read_text().split("\n")

# Run in a scratch directory. A page writes packs and probe scripts as it goes, and those
# belong nowhere near the tree being checked — a harness that dirties the repo it guards
# gets switched off.
work = pathlib.Path(tempfile.mkdtemp(prefix="jig-doc-"))
for name in ("examples", "tests", "jig"):
    try:
        (work / name).symlink_to(root / name)
    except OSError:
        pass
os.chdir(work)

# A page also writes to absolute paths — /tmp/hello, /tmp/notify.db, an outbox. Those
# survive between runs, and leftovers make a page pass or fail depending on what ran
# before it. Clear every /tmp path the page names, so a replay always starts from the
# state the page assumes: nothing.
for target in sorted(set(re.findall(r"/tmp/[\w.-]+", doc.read_text()))):
    path = pathlib.Path(target)
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        try:
            path.unlink()
        except OSError:
            pass

regions, inside, buf, lang, pending = [], False, [], "", None
prev = ""
for line in lines:
    if line.startswith("```"):
        if inside:
            regions.append((lang, buf, pending)); buf = []; inside = False; pending = None
        else:
            inside = True; lang = line[3:].strip(); buf = []
            m = re.match(r".*[Ss]ave (?:this )?as `([\w./-]+\.py)`", prev)
            pending = m.group(1) if m else None
        continue
    if inside:
        buf.append(line)
    elif line.strip():
        prev = line

env = dict(os.environ, PYTHONPATH=str(root))
IGNORE_TS = re.compile(r"^\d\d:\d\d:\d\d\.\d+ ")

def norm(text):
    """Strip log timestamps and duration_ms — neither can reproduce across runs."""
    out = []
    for line in text.split("\n"):
        line = IGNORE_TS.sub("", line)
        line = re.sub(r"duration_ms=[\d.]+", "duration_ms=X", line)
        line = re.sub(r"elapsed_ms=[\d.]+", "elapsed_ms=X", line)
        line = re.sub(r"run_id=[0-9a-f]{32}", "run_id=UUID", line)
        if line.strip() in ("(exit status 1)",): continue
        if re.match(r"^(validate )?exit=\d+$", line.strip()): continue
        out.append(line.rstrip())
    return "\n".join(out).strip()

ran = checked = fails = 0
for ri, (lang, block, named) in enumerate(regions):
    text = "\n".join(block)
    has_prompt = any(l.startswith("$ ") for l in block)

    if not has_prompt:
        first = block[0] if block else ""
        m_name = re.match(r'\s*(?:#|"""|\'\'\')\s*([\w./-]+\.py)\b', first)
        if lang == "python" and (named or m_name):
            name = named or m_name.group(1)
            pathlib.Path(name).write_text(text + "\n")
        elif lang in ("bash", "sh", "console", ""):
            if text.strip() and not text.strip().startswith(("{", "[")):
                subprocess.run(["bash","-c",text], capture_output=True, text=True,
                               env=env, timeout=180)
        continue

    i = 0
    while i < len(block):
        line = block[i]
        if not line.startswith("$ "):
            i += 1; continue
        cmd = line[2:]; i += 1
        while i < len(block) and block[i].startswith("> "):
            cmd += "\n" + block[i][2:]; i += 1
        m = re.search(r"<<'([A-Za-z]+)'|<<\"([A-Za-z]+)\"", cmd)
        if m:
            tag = m.group(1) or m.group(2)
            if ("\n" + tag) not in cmd:
                while i < len(block) and block[i].strip() != tag:
                    cmd += "\n" + block[i]; i += 1
                if i < len(block):
                    cmd += "\n" + block[i]; i += 1
        expected = []
        rest_has_cmd = any(l.startswith("$ ") for l in block[i:])
        while i < len(block) and not block[i].startswith("$ ") and (
                block[i].strip() != "" or not rest_has_cmd):
            expected.append(block[i]); i += 1
        ran += 1
        try:
            proc = subprocess.run(["bash","-c",cmd], capture_output=True, text=True,
                                  env=env, timeout=180)
            actual = (proc.stderr + proc.stdout).rstrip("\n")
        except Exception as exc:
            actual = "HARNESS: %s" % exc
        exp = "\n".join(expected)
        if not exp.strip():
            continue
        checked += 1
        def loose(a, b):
            """A doc may elide a volatile path as /.../ or a long block as [ ... ]."""
            if a == b: return True
            if "/.../" not in b and "elided" not in b: return False
            parts = [x for x in re.split(r"/\.\.\.|\[[^\]]*elided[^\]]*\]", b) if x.strip()]
            pos = 0
            for part in parts:
                k = a.find(part.strip(), pos)
                if k < 0: return False
                pos = k + len(part.strip())
            return True
        if not loose(norm(actual), norm(exp)):
            fails += 1
            print("MISMATCH  region %d: %s" % (ri, cmd.split("\n")[0][:88]))
            print("  expected| %s" % norm(exp)[:340].replace("\n", "\n          | "))
            print("  actual  | %s" % norm(actual)[:340].replace("\n", "\n          | "))
            print()
print("%s: ran %d, %d had printed output, %d did not reproduce"
      % (doc.name, ran, checked, fails))
shutil.rmtree(work, ignore_errors=True)
raise SystemExit(1 if fails else 0)
