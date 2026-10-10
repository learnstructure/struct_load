"""
Master Runner for all struct_load examples.
Executes each example script sequentially and reports status.
"""

from pathlib import Path
import subprocess
import sys
import time


def main():
    examples_dir = Path(__file__).resolve().parent
    example_scripts = sorted([
        f for f in examples_dir.glob("*.py")
        if f.name != "run_all.py" and f.name.startswith("0")
    ])

    print("=" * 80)
    print(f"RUNNING ALL STRUCT_LOAD EXAMPLES ({len(example_scripts)} total)")
    print("=" * 80)

    results = []
    overall_start = time.time()

    for script in example_scripts:
        print(f"\n>>> Running: {script.name} ...")
        t0 = time.time()
        res = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(examples_dir.parent.parent),
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0

        if res.returncode == 0:
            print(f"    [PASSED] in {elapsed:.2f}s")
            results.append((script.name, "PASSED", elapsed, None))
        else:
            print(f"    [FAILED] with exit code {res.returncode} in {elapsed:.2f}s")
            print("--- STDERR ---")
            print(res.stderr[:1000])
            results.append((script.name, "FAILED", elapsed, res.stderr))

    total_elapsed = time.time() - overall_start

    print("\n" + "=" * 80)
    print("EXAMPLES EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{'Example Script':<40} | {'Status':<10} | {'Time (s)':<10}")
    print("-" * 66)

    all_passed = True
    for name, status, elapsed, _ in results:
        if status != "PASSED":
            all_passed = False
        print(f"{name:<40} | {status:<10} | {elapsed:<10.2f}")

    print("-" * 66)
    print(f"Total Time: {total_elapsed:.2f}s | Result: {'ALL EXAMPLES PASSED' if all_passed else 'SOME EXAMPLES FAILED'}")
    print("=" * 80)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
