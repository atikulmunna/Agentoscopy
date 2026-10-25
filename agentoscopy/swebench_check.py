"""Grades a SWE-bench eval log the way SWE-bench does: `python check.py LOG TESTS_JSON`.

An imported SWE-bench task copies this file into its hidden/ folder, and its grader runs it in
the grading sandbox with the image's own Python, after the row's eval script has written LOG.
It exits 0 only when every FAIL_TO_PASS test passes and every PASS_TO_PASS test still passes
(or is skipped), and prints which ones did not otherwise.

The log parsers and grading rules are ported from the SWE-bench harness (MIT License,
github.com/SWE-bench/SWE-bench, v5.0.1: swebench/harness/log_parsers/python.py and
swebench/harness/grading.py). The file runs on its own with only the standard library, on
Python 3.8 or later, since it never sees the rest of agentoscopy.
"""

from __future__ import annotations

import json
import re
import sys

PASSED, FAILED, SKIPPED, ERROR, XFAIL = "PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL"
STATUSES = (FAILED, PASSED, SKIPPED, ERROR, XFAIL)
PASSING = (PASSED, XFAIL)

APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
RESET_FAILED = ">>>>> Reset Failed"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"
TEST_EXIT_CODE = ">>>>> Test Exit Code"
TEST_EXIT_CODE_RE = re.compile(re.escape(TEST_EXIT_CODE) + r":\s*(-?\d+)")
# Evidence that a runner executed when nothing was parsed; SWE-bench keeps it for its
# JavaScript runners, and it is kept here so the rule reads the same.
SUITE_RAN = re.compile(
    r"Executed [1-9]\d* of \d+"
    r"|TOTAL: [1-9]\d* (?:SUCCESS|FAILED)"
    r"|[1-9]\d* passing"
    r"|Tests:\s+[1-9]\d*"
    r"|Test Suites:\s+(?:\d+ \w+, )*[1-9]\d* total"
    r"|^# tests [1-9]\d*"
    r"|[1-9]\d* specs?, \d+ failures?"
    r"|': ok$",
    re.M,
)
# git apply's own complaints, which no line of a diff can start with. SWE-bench does not look
# for them, but when the hidden tests do not apply the log grades whatever tests are there.
TEST_PATCH_FAILED = re.compile(r"^error: (patch failed: |.*: patch does not apply$)", re.M)
SKIP_SUMMARY_COUNT = re.compile(r"^\[\d+\]$")
SHOWN_FAILURES = 10


def parse_pytest(log: str) -> dict[str, str]:
    status_map = {}
    for line in log.split("\n"):
        if line.startswith(STATUSES):
            _record_pytest_line(line, status_map)
    return status_map


def parse_matplotlib(log: str) -> dict[str, str]:
    status_map = {}
    for line in log.split("\n"):
        line = line.replace("MouseButton.LEFT", "1").replace("MouseButton.RIGHT", "3")
        if line.startswith(STATUSES):
            _record_pytest_line(line, status_map)
    return status_map


def _record_pytest_line(line: str, status_map: dict[str, str]) -> None:
    if line.startswith(FAILED):
        line = line.replace(" - ", " ")
    words = line.split()
    if len(words) > 1 and not _is_skip_summary(words[0], words[1]):
        status_map[words[1]] = words[0]


def parse_pytest_options(log: str) -> dict[str, str]:
    option_pattern = re.compile(r"(.*?)\[(.*)\]")
    status_map = {}
    for line in log.split("\n"):
        if not line.startswith(STATUSES):
            continue
        if line.startswith(FAILED):
            line = line.replace(" - ", " ")
        words = line.split()
        if len(words) <= 1 or _is_skip_summary(words[0], words[1]):
            continue
        has_option = option_pattern.search(words[1])
        if has_option:
            main, option = has_option.groups()
            if option.startswith("/") and not option.startswith("//") and "*" not in option:
                option = "/" + option.split("/")[-1]
            status_map[f"{main}[{option}]"] = words[0]
        else:
            status_map[words[1]] = words[0]
    return status_map


def parse_pytest_v2(log: str) -> dict[str, str]:
    escapes = str.maketrans("", "", "".join(chr(code) for code in range(1, 32)))
    status_map = {}
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line).translate(escapes)
        if line.startswith(STATUSES):
            if line.startswith(FAILED):
                line = line.split(" - ", 1)[0]  # keeps the assertion message out of the id
            words = line.split()
            if len(words) >= 2 and not _is_skip_summary(words[0], words[1]):
                status_map[" ".join(words[1:])] = words[0]
        elif line.endswith(STATUSES):  # older pytest prints the status last
            words = line.split()
            if len(words) >= 2:
                status_map[" ".join(words[:-1])] = words[-1]
    return status_map


def _is_skip_summary(status: str, name: str) -> bool:
    """pytest's `SKIPPED [N] path:line: reason` summary, where [N] is not a test."""
    return status == SKIPPED and bool(SKIP_SUMMARY_COUNT.match(name))


DJANGO_PASS_SUFFIXES = (" ... ok", " ... OK", " ...  OK")
# Django's logger sometimes interrupts a test's line with a multi-line message before "ok".
DJANGO_INTERRUPTED = (
    r"^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",
    r"^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$",
    r"^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$",
)


def parse_django(log: str) -> dict[str, str]:
    status_map: dict[str, str] = {}
    previous_test = None
    for raw_line in log.split("\n"):
        line = raw_line.strip()
        if "--version is equivalent to version" in line:
            status_map["--version is equivalent to version"] = PASSED
        if " ... " in line:
            previous_test = line.split(" ... ")[0]
        _record_django_pass(line, status_map)
        _record_django_other(line, status_map)
        if line.startswith("ok") and previous_test is not None:
            status_map[previous_test] = PASSED  # other output came between "..." and "ok"
    for pattern in DJANGO_INTERRUPTED:
        for match in re.finditer(pattern, log, re.MULTILINE):
            status_map[match.group(1)] = PASSED
    return status_map


def _record_django_pass(line: str, status_map: dict[str, str]) -> None:
    for suffix in DJANGO_PASS_SUFFIXES:
        if line.endswith(suffix):
            # SWE-bench's own special case for django__django-7188
            if line.startswith("Applying sites.0002_alter_domain_unique...test_no_migrations"):
                line = line.split("...", 1)[-1].strip()
            status_map[line.rsplit(suffix, 1)[0]] = PASSED
            return


def _record_django_other(line: str, status_map: dict[str, str]) -> None:
    if " ... skipped" in line:
        status_map[line.split(" ... skipped")[0]] = SKIPPED
    if line.endswith(" ... FAIL"):
        status_map[line.split(" ... FAIL")[0]] = FAILED
    if line.startswith("FAIL:"):
        status_map[line.split()[1].strip()] = FAILED
    if line.endswith(" ... ERROR"):
        status_map[line.split(" ... ERROR")[0]] = ERROR
    if line.startswith("ERROR:"):
        status_map[line.split()[1].strip()] = ERROR


def parse_seaborn(log: str) -> dict[str, str]:
    status_map = {}
    for line in log.split("\n"):
        words = line.split()
        if len(words) < 2:
            continue
        if line.startswith(FAILED):
            status_map[words[1]] = FAILED
        elif f" {PASSED} " in line:
            if words[1] == PASSED:
                status_map[words[0]] = PASSED
        elif line.startswith(PASSED):
            status_map[words[1]] = PASSED
    return status_map


SYMPY_LINE_STATUSES = ((" E", ERROR), (" F", FAILED), (" ok", PASSED))


def parse_sympy(log: str) -> dict[str, str]:
    status_map = {}
    for match in re.findall(r"(_*) (.*)\.py:(.*) (_*)", log):
        status_map[f"{match[1]}.py:{match[2]}"] = FAILED
    for raw_line in log.split("\n"):
        line = raw_line.strip()
        if not line.startswith("test_"):
            continue
        for ending, status in SYMPY_LINE_STATUSES:
            if line.endswith(ending):
                status_map[line.split()[0]] = status
    return status_map


# SWE-bench's parser names for Python repositories, by the function each one is.
PARSERS = {
    "parse_log_pytest": parse_pytest,
    "parse_log_astroid": parse_pytest,
    "parse_log_flask": parse_pytest,
    "parse_log_marshmallow": parse_pytest,
    "parse_log_pvlib": parse_pytest,
    "parse_log_pyvista": parse_pytest,
    "parse_log_sqlfluff": parse_pytest,
    "parse_log_xarray": parse_pytest,
    "parse_log_pytest_options": parse_pytest_options,
    "parse_log_pydicom": parse_pytest_options,
    "parse_log_requests": parse_pytest_options,
    "parse_log_pylint": parse_pytest_options,
    "parse_log_pytest_v2": parse_pytest_v2,
    "parse_log_astropy": parse_pytest_v2,
    "parse_log_scikit": parse_pytest_v2,
    "parse_log_sphinx": parse_pytest_v2,
    "parse_log_django": parse_django,
    "parse_log_seaborn": parse_seaborn,
    "parse_log_sympy": parse_sympy,
    "parse_log_matplotlib": parse_matplotlib,
}


class LogUnusable(Exception):
    """The log does not describe a test run that can be graded."""


def status_map_from_log(content: str, parser_name: str) -> dict[str, str]:
    """Each test's status, from the part of the log between the test output markers."""
    parse = PARSERS[parser_name]
    for marker in (APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT):
        if marker in content:
            raise LogUnusable(f"the eval script reported {marker!r}")
    if START_TEST_OUTPUT not in content or END_TEST_OUTPUT not in content:
        raise LogUnusable("the test output markers are missing, so the tests never ran")
    if TEST_PATCH_FAILED.search(content.split(START_TEST_OUTPUT, 1)[0]):
        raise LogUnusable("the hidden tests did not apply, so other tests would be graded")
    sliced = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    status_map = parse(sliced) or parse(content)
    if not status_map and not SUITE_RAN.search(content):
        raise LogUnusable("no test results in the log")
    # A patch can print its own "PASSED" lines, so a run whose tests exited non-zero must
    # show at least one failure.
    exit_code = _test_exit_code(content)
    reported_failure = any(status in (FAILED, ERROR) for status in status_map.values())
    if exit_code not in (None, 0) and status_map and not reported_failure:
        raise LogUnusable(f"the tests exited {exit_code} but the log shows no failure")
    return status_map


def _test_exit_code(content: str) -> int | None:
    match = TEST_EXIT_CODE_RE.search(content)
    return int(match.group(1)) if match else None


def _resolve_case(case: str, status_map: dict[str, str]) -> str | None:
    """The status map's key for `case`. SWE-bench Verified truncates some parametrized ids
    mid-parameter, so those, and only those, match by prefix when every match agrees."""
    if case in status_map:
        return case
    if case.count("[") > case.count("]"):
        matches = [key for key in status_map if key.startswith(case)]
        if matches and len({status_map[key] in PASSING for key in matches}) == 1:
            return matches[0]
    return None


def fails_to_pass(case: str, status_map: dict[str, str]) -> bool:
    """A FAIL_TO_PASS test still failing: missing, failed, errored, or skipped. Like SWE-bench,
    a status that is neither passing nor one of those counts against nothing."""
    key = _resolve_case(case, status_map)
    return key is None or status_map[key] in (FAILED, ERROR, SKIPPED)


def stopped_passing(case: str, status_map: dict[str, str]) -> bool:
    """A PASS_TO_PASS test that regressed: missing, failed, or errored. Skipped is not."""
    key = _resolve_case(case, status_map)
    return key is None or status_map[key] in (FAILED, ERROR)


def grade(content: str, tests: dict) -> list[str]:
    """Problems that keep the instance from being resolved; empty means resolved."""
    try:
        status_map = status_map_from_log(content, tests["log_parser"])
    except LogUnusable as exc:
        return [str(exc)]
    unfixed = [case for case in tests["FAIL_TO_PASS"] if fails_to_pass(case, status_map)]
    broken = [case for case in tests["PASS_TO_PASS"] if stopped_passing(case, status_map)]
    problems = []
    if unfixed:
        problems.append(_failures("FAIL_TO_PASS", unfixed, len(tests["FAIL_TO_PASS"])))
    if broken:
        problems.append(_failures("PASS_TO_PASS", broken, len(tests["PASS_TO_PASS"])))
    return problems


def _failures(kind: str, cases: list[str], total: int) -> str:
    shown = "\n  ".join(cases[:SHOWN_FAILURES])
    more = f"\n  and {len(cases) - SHOWN_FAILURES} more" if len(cases) > SHOWN_FAILURES else ""
    return f"{kind}: {len(cases)} of {total} did not pass:\n  {shown}{more}"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: check.py LOG TESTS_JSON", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8", errors="replace") as log_file:
        content = log_file.read()
    with open(argv[2], encoding="utf-8") as tests_file:
        tests = json.load(tests_file)
    problems = grade(content, tests)
    if problems:
        print("not resolved\n" + "\n".join(problems))
        return 1
    fixed, kept = len(tests["FAIL_TO_PASS"]), len(tests["PASS_TO_PASS"])
    print(f"resolved: {fixed} FAIL_TO_PASS tests pass, {kept} PASS_TO_PASS tests still pass")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
