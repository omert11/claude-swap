"""Per-account switch priority: the value, the store, the CLI verb and display.

`cswap priority` mirrors `cswap threshold` (see `test_account_policy_cli.py`):
validated at the write boundary before any file is touched, omitted from the
record when it is the default, degraded rather than raised on a bad stored
value, and shown wherever the threshold override is shown (`--list --json`,
the TUI badges).

Synthetic fixtures only. The real store at the user home is never read or
written; `tests/test_real_store_guard.py` enforces that boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_swap import cli
from claude_swap.exceptions import AccountNotFoundError
from claude_swap.json_output import account_row
from claude_swap.models import (
    ACCOUNT_PRIORITY_MAX,
    ACCOUNT_PRIORITY_MIN,
    AccountPolicy,
    normalize_account_priority,
)
from claude_swap.switcher import ClaudeAccountSwitcher

from tests.test_account_policy_cli import PolicyCliBase
from tests.test_tui import make_account
from tests.test_tui_policy_display import card, mini, with_policy


class TestNormalizeAccountPriority:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(10, 10), (0, 0), (-5, -5), ("7", 7), (" -3 ", -3), ("+4", 4), (2.0, 2),
         (ACCOUNT_PRIORITY_MIN, ACCOUNT_PRIORITY_MIN),
         (ACCOUNT_PRIORITY_MAX, ACCOUNT_PRIORITY_MAX)],
    )
    def test_accepts_integers_in_range(self, value, expected):
        result = normalize_account_priority(value)
        assert result == expected
        assert isinstance(result, int)

    @pytest.mark.parametrize(
        "value",
        [True, False, None, [], 1.5, "1.5", "ten", "", float("nan"), float("inf"),
         ACCOUNT_PRIORITY_MAX + 1, ACCOUNT_PRIORITY_MIN - 1, "1000"],
    )
    def test_rejects_everything_else_naming_the_range(self, value):
        with pytest.raises(ValueError, match="integer between"):
            normalize_account_priority(value)

    def test_the_default_policy_has_priority_zero(self):
        assert AccountPolicy().priority == 0


class TestPriorityStore(PolicyCliBase):
    def test_set_writes_an_int_and_reads_back(self, temp_home: Path, capsys):
        s = self._fleet(temp_home)
        s.set_account_priority("2", 10)
        record = self._record(ClaudeAccountSwitcher(), "2")
        assert record["priority"] == 10 and isinstance(record["priority"], int)
        assert s.account_policies()["2"] == AccountPolicy(priority=10)
        assert s.account_policies()["1"] == AccountPolicy()
        assert "priority to 10" in capsys.readouterr().out

    def test_it_coexists_with_a_threshold(self, temp_home: Path):
        s = self._fleet(temp_home)
        s.set_account_threshold("2", 85.0)
        s.set_account_priority("2", -1)
        assert s.account_policies()["2"] == AccountPolicy(threshold=85.0, priority=-1)

    def test_unset_removes_the_key_rather_than_writing_null(self, temp_home: Path):
        s = self._fleet(temp_home)
        s.set_account_priority("2", 10)
        s.set_account_priority("2", None)
        assert "priority" not in self._record(s, "2")

    def test_zero_is_the_default_and_clears(self, temp_home: Path, capsys):
        s = self._fleet(temp_home)
        s.set_account_priority("2", 5)
        s.set_account_priority("2", 0)
        assert "priority" not in self._record(s, "2")
        assert "Cleared" in capsys.readouterr().out

    def test_unset_without_a_priority_does_not_write(self, temp_home: Path, capsys):
        s = self._fleet(temp_home)
        before = self._bytes(s)
        s.set_account_priority("2", None)
        assert self._bytes(s) == before
        assert "already at the default" in capsys.readouterr().out

    def test_setting_the_same_value_does_not_write(self, temp_home: Path, capsys):
        s = self._fleet(temp_home)
        s.set_account_priority("2", 3)
        before = self._bytes(s)
        s.set_account_priority("2", "3")
        assert self._bytes(s) == before
        assert "already at priority 3" in capsys.readouterr().out

    def test_a_rejected_value_leaves_the_store_byte_identical(self, temp_home: Path):
        s = self._fleet(temp_home)
        before = self._bytes(s)
        with pytest.raises(ValueError):
            s.set_account_priority("2", "1.5")
        assert self._bytes(s) == before

    def test_an_unknown_account_raises(self, temp_home: Path):
        s = self._fleet(temp_home)
        with pytest.raises(AccountNotFoundError):
            s.set_account_priority("9", 1)

    @pytest.mark.parametrize("garbage", ["high", 1.5, True, [], 1000, None])
    def test_an_unreadable_stored_priority_degrades_to_zero(
        self, temp_home: Path, garbage
    ):
        s = self._fleet(temp_home)
        data = s._get_sequence_data() or {}
        data["accounts"]["1"]["priority"] = garbage
        s.sequence_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        reloaded = s._get_sequence_data() or {}
        assert s._policy_from_data(reloaded, "1").priority == 0


class TestPriorityCli(PolicyCliBase):
    @staticmethod
    def _priority(argv: list[str]) -> None:
        from unittest.mock import patch

        with patch("os.geteuid", return_value=1000, create=True):
            cli._priority_command(argv)

    def test_set_through_main_dispatch(self, temp_home: Path, capsys):
        self._fleet(temp_home)
        self._main(["priority", "2", "10"])
        assert self._record(ClaudeAccountSwitcher(), "2")["priority"] == 10

    def test_a_negative_value_parses_as_the_positional(self, temp_home: Path):
        self._fleet(temp_home)
        self._priority(["3", "-2"])
        assert self._record(ClaudeAccountSwitcher(), "3")["priority"] == -2

    def test_email_and_alias_resolve(self, temp_home: Path):
        s = self._fleet(temp_home)
        self._alias(s, "3", "team")
        self._priority(["account-2@example.test", "4"])
        self._priority(["team", "5"])
        policies = ClaudeAccountSwitcher().account_policies()
        assert (policies["2"].priority, policies["3"].priority) == (4, 5)

    def test_unset(self, temp_home: Path):
        self._fleet(temp_home)
        self._priority(["2", "10"])
        self._priority(["2", "--unset"])
        assert "priority" not in self._record(ClaudeAccountSwitcher(), "2")

    def test_list_shows_every_account_highest_first(self, temp_home: Path, capsys):
        self._fleet(temp_home)
        self._priority(["3", "10"])
        self._priority(["1", "-1"])
        capsys.readouterr()
        self._priority([])
        out = capsys.readouterr().out
        rows = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("Account-")]
        assert [r.split()[0] for r in rows] == ["Account-3", "Account-2", "Account-1"]
        assert rows[0].endswith("10") and rows[2].endswith("-1")

    def test_a_bad_value_exits_1_without_writing(self, temp_home: Path, capsys):
        s = self._fleet(temp_home)
        before = self._bytes(s)
        with pytest.raises(SystemExit) as exc:
            self._priority(["2", "high"])
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "integer between" in captured.err + captured.out
        assert self._bytes(s) == before

    @pytest.mark.parametrize(
        "argv", [["2"], ["--unset"], ["2", "5", "--unset"]]
    )
    def test_argument_errors(self, temp_home: Path, argv):
        self._fleet(temp_home)
        with pytest.raises(SystemExit) as exc:
            self._priority(argv)
        assert exc.value.code == 2


class TestPriorityDisplay(PolicyCliBase):
    def test_json_row_omits_priority_at_the_default(self):
        row = account_row(1, "a@example.test", "", "", True, None)
        assert "priority" not in row

    def test_json_row_emits_a_set_priority(self):
        row = account_row(
            1, "a@example.test", "", "", True, None, policy=AccountPolicy(priority=10)
        )
        assert row["priority"] == 10

    def test_list_json_carries_it_end_to_end(self, temp_home: Path):
        s = self._fleet(temp_home)
        s.set_account_priority("2", 10)
        payload = s.list_accounts(json_output=True, fetch=set())
        rows = {r["number"]: r for r in payload["accounts"]}
        assert rows[2]["priority"] == 10
        assert "priority" not in rows[1] and "priority" not in rows[3]

    def test_tui_badges_show_a_signed_priority(self):
        acc = with_policy(make_account(2, active=True, alias="dev"), priority=10)
        assert "prio +10" in card(acc).splitlines()[0]
        assert "prio +10" in mini(acc)
        low = with_policy(make_account(2, active=True, alias="dev"), priority=-1)
        assert "prio -1" in mini(low)

    def test_no_priority_no_badge(self):
        acc = with_policy(make_account(2, active=True, alias="dev"))
        assert "prio" not in card(acc) and "prio" not in mini(acc)
