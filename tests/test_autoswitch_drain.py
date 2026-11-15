"""Drain-before-reset (``autoswitch.drainHours``).

An account whose weekly window resets within ``drain_hours`` and that is
still under its hard lines outranks every stored priority (soonest reset
first) and runs without a soft line, so its perishable weekly quota is spent
before the reset. Over a hard line it falls back to its stored priority.

Fleet: account 1 a personal account (priority 0), accounts 2 and 3 Team
accounts (priority 10), ``threshold=90``, ``softThreshold=60``,
``weeklyThreshold=97``, ``drainHours=12``. Synthetic fixtures only.
"""

from __future__ import annotations

from claude_swap.autoswitch import NoSwitchEvent, SwitchEvent, TickOutcome
from claude_swap.settings import AutoSwitchSettings, SETTING_SPECS

from tests.test_autoswitch import EngineHarness, _iso_at

SETTINGS = {
    "threshold": 90.0,
    "soft_threshold": 60.0,
    "weekly_threshold": 97.0,
    "drain_hours": 12.0,
}
EMAILS = {1: "personal@example.com", 2: "team-b@example.com", 3: "team-c@example.com"}


def _fleet(temp_home, *, active: int, **settings) -> EngineHarness:
    h = EngineHarness(temp_home, **{**SETTINGS, **settings})
    for num, email in EMAILS.items():
        h.seed(num, email)
    data = h.switcher._get_sequence_data()
    data["activeAccountNumber"] = active
    h.switcher._write_json(h.switcher.sequence_file, data)
    h.make_live(EMAILS[active], active)
    h.switcher.set_account_priority("2", 10)
    h.switcher.set_account_priority("3", 10)
    return h


def _u(h: EngineHarness, pct5: float, pct7: float, reset_in_h: float) -> dict:
    return {
        "five_hour": {"pct": pct5},
        "seven_day": {"pct": pct7, "resets_at": _iso_at(h.clock.now + reset_in_h * 3600)},
    }


def _reasons(h: EngineHarness) -> list[str]:
    return [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]


def _switch(h: EngineHarness) -> SwitchEvent:
    return next(e for e in h.events if isinstance(e, SwitchEvent))


class TestDrainTakesOver:
    def test_switches_to_the_account_resetting_soon(self, temp_home):
        h = _fleet(temp_home, active=2)
        outcome = h.tick_with_usage({
            "1": _u(h, 0, 30, 90),
            "2": _u(h, 20, 40, 100),
            "3": _u(h, 10, 50, 6),
        })
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 3
        assert _switch(h).trigger == "priority"

    def test_outranks_a_higher_stored_priority(self, temp_home):
        h = _fleet(temp_home, active=2)
        outcome = h.tick_with_usage({
            "1": _u(h, 0, 30, 5),
            "2": _u(h, 20, 40, 100),
            "3": _u(h, 10, 50, 100),
        })
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 1

    def test_soonest_reset_wins_among_several(self, temp_home):
        h = _fleet(temp_home, active=2)
        outcome = h.tick_with_usage({
            "1": _u(h, 0, 30, 10),
            "2": _u(h, 20, 40, 100),
            "3": _u(h, 10, 50, 3),
        })
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 3


class TestDrainHolds:
    def test_ignores_the_soft_line(self, temp_home):
        h = _fleet(temp_home, active=3)
        outcome = h.tick_with_usage({
            "1": _u(h, 0, 30, 90),
            "2": _u(h, 0, 40, 100),
            "3": _u(h, 75, 80, 6),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert _reasons(h) == ["below-threshold"]
        assert h.active_number() == 3

    def test_leaves_at_the_hard_line(self, temp_home):
        h = _fleet(temp_home, active=3)
        outcome = h.tick_with_usage({
            "1": _u(h, 0, 30, 90),
            "2": _u(h, 0, 40, 100),
            "3": _u(h, 95, 80, 6),
        })
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 2

    def test_leaves_at_the_weekly_line(self, temp_home):
        h = _fleet(temp_home, active=3)
        outcome = h.tick_with_usage({
            "1": _u(h, 0, 30, 90),
            "2": _u(h, 0, 40, 100),
            "3": _u(h, 20, 98, 6),
        })
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 2


class TestDrainOff:
    def test_beyond_the_horizon_holds(self, temp_home):
        h = _fleet(temp_home, active=2)
        outcome = h.tick_with_usage({
            "1": _u(h, 0, 30, 90),
            "2": _u(h, 20, 40, 100),
            "3": _u(h, 10, 50, 13),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 2

    def test_zero_hours_disables_it(self, temp_home):
        h = _fleet(temp_home, active=2, drain_hours=0.0)
        outcome = h.tick_with_usage({
            "1": _u(h, 0, 30, 2),
            "2": _u(h, 20, 40, 100),
            "3": _u(h, 10, 50, 2),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 2


class TestSetting:
    def test_default_is_off(self):
        assert AutoSwitchSettings().drain_hours == 0.0

    def test_registered_as_config_key(self):
        spec = SETTING_SPECS["autoswitch.drainHours"]
        assert spec.field == "drain_hours"
