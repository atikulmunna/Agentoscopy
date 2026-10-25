"""Grading a SWE-bench eval log the way SWE-bench does (agentoscopy.swebench_check)."""

import ast
import json
from pathlib import Path

import pytest

from agentoscopy import swebench_check
from agentoscopy.swebench_check import (
    fails_to_pass,
    grade,
    main,
    parse_django,
    parse_matplotlib,
    parse_pytest,
    parse_pytest_options,
    parse_pytest_v2,
    parse_seaborn,
    parse_sympy,
    stopped_passing,
)


def log(body, exit_code=0):
    """An eval log as `bash -x eval.sh` writes it, around the test output."""
    return (
        "+ git apply -v -\n+ : '>>>>> Start Test Output'\n"
        f"{body}\n+ SWEBENCH_TEST_EXIT_CODE={exit_code}\n+ : '>>>>> End Test Output'\n"
        f">>>>> Test Exit Code: {exit_code}\n"
    )


def test_pytest_reads_the_short_summary_and_skips_skip_counts():
    summary = (
        "PASSED tests/test_a.py::test_one\n"
        "FAILED tests/test_a.py::test_two - AssertionError: boom\n"
        "ERROR tests/test_a.py::test_three\n"
        "XFAIL tests/test_a.py::test_four - flaky\n"
        "SKIPPED [2] tests/test_a.py:10: needs a network\n"
    )

    assert parse_pytest(summary) == {
        "tests/test_a.py::test_one": "PASSED",
        "tests/test_a.py::test_two": "FAILED",
        "tests/test_a.py::test_three": "ERROR",
        "tests/test_a.py::test_four": "XFAIL",
    }


def test_pytest_v2_drops_colours_and_reads_old_and_spaced_ids():
    summary = (
        "\x1b[32mPASSED\x1b[0m tests/x.py::test_a\n"
        "FAILED tests/x.py::test_c - assert 1 == 2\n"
        "PASSED tests/x.py::test_d[a b]\n"
        "tests/x.py::test_b PASSED\n"
    )

    assert parse_pytest_v2(summary) == {
        "tests/x.py::test_a": "PASSED",
        "tests/x.py::test_c": "FAILED",
        "tests/x.py::test_d[a b]": "PASSED",
        "tests/x.py::test_b": "PASSED",
    }


def test_pytest_options_shortens_path_parameters():
    summary = "PASSED tests/t.py::test_x[/tmp/dir/file]\nPASSED tests/t.py::test_y[//net]\n"

    assert parse_pytest_options(summary) == {
        "tests/t.py::test_x[/file]": "PASSED",
        "tests/t.py::test_y[//net]": "PASSED",
    }


def test_matplotlib_names_mouse_buttons_by_number():
    assert parse_matplotlib("PASSED lib/t.py::test_click[MouseButton.LEFT]\n") == {
        "lib/t.py::test_click[1]": "PASSED"
    }


def test_seaborn_reads_status_after_the_name():
    summary = "tests/t.py::test_a PASSED [ 50%]\nFAILED tests/t.py::test_b - boom\n"

    assert parse_seaborn(summary) == {
        "tests/t.py::test_a": "PASSED",
        "tests/t.py::test_b": "FAILED",
    }


def test_django_reads_verbose_results_and_interrupted_lines():
    output = (
        "test_ascii (auth_tests.test_validators.UsernameValidatorsTests) ... ok\n"
        "test_unicode (auth_tests.test_validators.UsernameValidatorsTests) ... FAIL\n"
        "test_help (auth_tests.Other) ... skipped 'no reason'\n"
        "test_broken (auth_tests.Other) ... ERROR\n"
        "test_noisy (auth_tests.Other) ... some output first\n"
        "ok\n"
        "FAIL: test_listed (auth_tests.Other)\n"
    )

    assert parse_django(output) == {
        "test_ascii (auth_tests.test_validators.UsernameValidatorsTests)": "PASSED",
        "test_unicode (auth_tests.test_validators.UsernameValidatorsTests)": "FAILED",
        "test_help (auth_tests.Other)": "SKIPPED",
        "test_broken (auth_tests.Other)": "ERROR",
        "test_noisy (auth_tests.Other)": "PASSED",
        "test_listed": "FAILED",
    }


def test_sympy_reads_results_and_failure_banners():
    output = (
        "test_immutable F\ntest_structure ok\ntest_has E\n"
        "______ sympy/core/tests/test_basic.py:test_immutable ______\n"
    )

    assert parse_sympy(output) == {
        "test_immutable": "FAILED",
        "test_structure": "PASSED",
        "test_has": "ERROR",
        "sympy/core/tests/test_basic.py:test_immutable": "FAILED",
    }


TESTS = {"log_parser": "parse_log_pytest", "FAIL_TO_PASS": ["t.py::fixed"],
         "PASS_TO_PASS": ["t.py::kept", "t.py::skipped"]}  # fmt: skip


def test_an_instance_is_resolved_when_its_tests_pass_and_skips_are_kept():
    body = "PASSED t.py::fixed\nPASSED t.py::kept\nSKIPPED t.py::skipped"

    assert grade(log(body), TESTS) == []


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        ("FAILED t.py::fixed\nPASSED t.py::kept", "FAIL_TO_PASS: 1 of 1 did not pass"),
        ("SKIPPED t.py::fixed\nPASSED t.py::kept", "FAIL_TO_PASS: 1 of 1"),  # skipped is no fix
        ("PASSED t.py::kept", "FAIL_TO_PASS: 1 of 1"),  # missing
        ("PASSED t.py::fixed\nFAILED t.py::kept", "PASS_TO_PASS: 2 of 2"),  # and one missing
    ],
)
def test_unfixed_or_broken_tests_keep_an_instance_unresolved(body, problem):
    (found,) = [line for line in grade(log(body), TESTS) if line.startswith(problem[:12])]

    assert found.startswith(problem)


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (log("PASSED t.py::fixed") + ">>>>> Patch Apply Failed\n", "Patch Apply Failed"),
        ("PASSED t.py::fixed\n", "markers are missing"),
        (log("collected 0 items"), "no test results"),
        (log("PASSED t.py::fixed\nPASSED t.py::kept", exit_code=1), "exited 1"),
        (
            "error: patch failed: tests/t.py:237\nerror: tests/t.py: patch does not apply\n"
            + log("PASSED t.py::fixed\nPASSED t.py::kept"),
            "the hidden tests did not apply",
        ),
    ],
)
def test_a_log_that_does_not_describe_a_run_is_not_graded(content, reason):
    (problem,) = grade(content, TESTS)

    assert reason in problem


def test_an_agent_diff_that_mentions_a_failed_patch_is_not_mistaken_for_one():
    diff = "+    message = 'error: patch failed: x'\n-error: y: patch does not apply\n"
    body = "PASSED t.py::fixed\nPASSED t.py::kept\nPASSED t.py::skipped"

    assert grade(diff + log(body), TESTS) == []


def test_truncated_parametrized_ids_match_by_prefix_only_when_unambiguous():
    agree = {"t.py::test_a[log(photon)]": "PASSED", "t.py::test_a[log(photon2)]": "XFAIL"}
    split = {"t.py::test_a[log(photon)]": "PASSED", "t.py::test_a[log(photon2)]": "FAILED"}

    assert not fails_to_pass("t.py::test_a[log(photon", agree)
    assert fails_to_pass("t.py::test_a[log(photon", split)
    assert fails_to_pass("t.py::test_a", agree)  # only ids cut inside brackets


def test_a_status_neither_passing_nor_failing_counts_against_nothing():
    """As in SWE-bench, where such a test lands in neither its success nor failure list."""
    assert not fails_to_pass("t", {"t": "XPASSED"})
    assert not stopped_passing("t", {"t": "XPASSED"})
    assert not stopped_passing("t", {"t": "SKIPPED"})


def test_every_python_parser_name_in_swe_bench_is_known():
    names = {"astroid", "astropy", "django", "flask", "marshmallow", "matplotlib", "pydicom",
             "pylint", "pytest", "pytest_options", "pytest_v2", "pvlib", "pyvista", "requests",
             "scikit", "seaborn", "sphinx", "sqlfluff", "sympy", "xarray"}  # fmt: skip

    assert set(swebench_check.PARSERS) == {f"parse_log_{name}" for name in names}


def test_main_exits_zero_only_when_resolved(tmp_path, capsys):
    tests_file = tmp_path / "tests.json"
    tests_file.write_text(json.dumps(TESTS))
    resolved, unresolved = tmp_path / "resolved.log", tmp_path / "unresolved.log"
    resolved.write_text(log("PASSED t.py::fixed\nPASSED t.py::kept\nPASSED t.py::skipped"))
    unresolved.write_text(log("FAILED t.py::fixed\nPASSED t.py::kept"))

    assert main(["check.py", str(resolved), str(tests_file)]) == 0
    assert "resolved: 1 FAIL_TO_PASS tests pass" in capsys.readouterr().out
    assert main(["check.py", str(unresolved), str(tests_file)]) == 1
    assert "not resolved\nFAIL_TO_PASS" in capsys.readouterr().out
    assert main(["check.py"]) == 2


def test_the_check_runs_on_the_old_pythons_in_swe_bench_images():
    source = Path(swebench_check.__file__).read_text(encoding="utf-8")

    ast.parse(source, feature_version=(3, 8))
    assert "from agentoscopy" not in source and "import agentoscopy" not in source
