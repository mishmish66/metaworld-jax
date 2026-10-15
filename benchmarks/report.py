"""
Write the result tables of `benchmarks/README.md` from `benchmarks/results/`, below its `<!-- results -->` line.

    uv run python benchmarks/report.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
REFERENCE = json.loads((HERE.parent / "tests" / "assets" / "reference" / "success.json").read_text())
MARKER = "<!-- results -->"


def _load(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def _rate(x):
    return "—" if x is None else f"{x / 1000:.0f}k" if x >= 1000 else f"{x:.0f}"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def _tasks(setting):
    return {k: v for k, v in setting.items() if k != "options"}


def physics_settings(paired, variants):
    """Success and throughput of each physics setting on the tasks they were all run on."""
    settings = {**variants, **paired}
    tasks = sorted(set.intersection(*(set(_tasks(s)) for s in settings.values())))
    rows = []
    base = _tasks(paired["upstream"])
    for name, setting in settings.items():
        results = _tasks(setting)
        success = sum(results[t]["success_rate"] for t in tasks) / len(tasks)
        worst = min(results[t]["success_rate"] - REFERENCE[t]["success_rate"] for t in tasks)
        speed = [results[t]["steps_per_second"] / base[t]["steps_per_second"] for t in tasks] if name in paired else None
        rows.append([
            f"`{name}`", f"`{json.dumps(setting['options'])}`", f"{success:.2f}", f"{worst:+.2f}",
            "—" if speed is None else f"{sum(speed) / len(speed):.2f}×",
        ])
    upstream = sum(REFERENCE[t]["success_rate"] for t in tasks) / len(tasks)
    return (
        f"Tasks: {', '.join(tasks)}. Upstream Meta-World's mean success is {upstream:.2f}. Speed is relative to "
        "`upstream` on the same GPU in the same run, so it is given only for the settings run alongside it.\n\n"
        + _table(["setting", "MuJoCo options", "mean success", "worst task vs upstream", "speed"], rows)
    )


def per_task(physics, upstream, timed="upstream"):
    """Success and steps to success per task, next to upstream Meta-World's, and the throughput of setting `timed`."""
    names = list(physics)
    tasks = sorted(set.union(*(set(_tasks(physics[n])) for n in names)))
    header = ["task", "success (upstream)"] + [f"success, `{n}`" for n in names] + ["steps (upstream)"] + [
        f"steps, `{n}`" for n in names
    ] + [f"steps/s, `{timed}`", "upstream steps/s, 1 process", "upstream steps/s, all processes"]
    rows = []
    for t in tasks:
        results = [_tasks(physics[n]).get(t) for n in names]
        cpu = upstream.get(t, {})

        def steps(x):
            return "—" if x is None or x.get("mean_steps") is None else f"{x['mean_steps']:.0f}"

        rows.append(
            [t, f"{REFERENCE[t]['success_rate']:.2f}"]
            + ["—" if r is None else f"{r['success_rate']:.2f}" for r in results]
            + [steps(REFERENCE[t])] + [steps(r) for r in results]
            + [_rate(_tasks(physics[timed]).get(t, {}).get("steps_per_second"))]
            + [_rate(cpu.get("one_process")), _rate(cpu.get("all_processes"))]
        )
    return _table(header, rows)


def main():
    physics = _load("physics.json")
    paired, variants = _load("physics-paired.json"), _load("physics-variants.json")
    upstream = _load("upstream.json")
    sections = []
    if paired:
        sections.append("### Physics settings\n\n" + physics_settings(paired, variants))
    if physics:
        sections.append(
            "### Per task\n\nThe settings ran at different times on GPUs with different loads, so their speeds are "
            "compared in the table above, where they ran together.\n\n" + per_task(physics, upstream)
        )
    readme = HERE / "README.md"
    text = readme.read_text().split(MARKER)[0] + MARKER + "\n\n" + "\n\n".join(sections) + "\n"
    readme.write_text(text)
    print(f"wrote {readme}")


if __name__ == "__main__":
    main()
