"""Per-account switch priority across every trigger the engine has.

Priority orders the candidates the existing gates already accept — higher
first — on the proactive, consume-first, at-limit and failover paths, and
never admits one they reject. While the active account is under its own line,
a healthy candidate of strictly higher priority is switched back to (the
``priority`` trigger), under the hysteresis margin, on fresh data, and inside
the cooldown floor. The all-above regime keeps its soonest-recovery order.

Every test drives a real tick through `EngineHarness`. The fleet is the one
the feature was asked for: account 1 a personal account with a weekly
window, accounts 2 and 3 Team accounts that report only a 5h window, with
`threshold=60`, `weeklyThreshold=97`.

Synthetic fixtures only: `EngineHarness` seeds a throwaway store under
`temp_home`.
"""

from __future__ import annotations

import json

from claude_swap.autoswitch import (
    NoSwitchEvent,
    PriorityMap,
    SwitchEvent,
    TickOutcome,
)
from claude_swap.usage_store import UsageEntry

from tests.test_autoswitch import (
    _R_LATER,
    _R_LATEST,
    _R_SOON,
    EngineHarness,
    _entry_for,
    _iso_at,
    _usage,
    _usage7,
)

SETTINGS = {"threshold": 60.0, "weekly_threshold": 97.0}


def _team(pct5: float, resets_at: str | None = None) -> dict:
    """A Team account's usage: a 5h window only."""
    window: dict = {"pct": pct5}
    if resets_at:
        window["resets_at"] = resets_at
    return {"five_hour": window}


def _personal(pct5: float, pct7: float = 30.0) -> dict:
    return _usage7(pct5, pct7)


def _fleet(temp_home, *, active: int = 1, **settings) -> EngineHarness:
    h = EngineHarness(temp_home, **{**SETTINGS, **settings})
    h.seed(1, "personal@example.com")
    h.seed(2, "team-b@example.com")
    h.seed(3, "team-c@example.com")
    emails = {1: "personal@example.com", 2: "team-b@example.com", 3: "team-c@example.com"}
    data = h.switcher._get_sequence_data()
    data["activeAccountNumber"] = active
    h.switcher._write_json(h.switcher.sequence_file, data)
    h.make_live(emails[active], active)
    return h


def _prioritise(h: EngineHarness, **by_num: int) -> None:
    for num, value in by_num.items():
        h.switcher.set_account_priority(num.lstrip("n"), value)


def _reasons(h: EngineHarness) -> list[str]:
    return [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]


def _switch(h: EngineHarness) -> SwitchEvent:
    return next(e for e in h.events if isinstance(e, SwitchEvent))


class TestPriorityMap:
    def test_absent_accounts_read_zero(self):
        assert PriorityMap({"2": 10})["9"] == 0
        assert PriorityMap()["1"] == 0

    def test_resolution_carries_only_set_priorities(self, temp_home):
        h = _fleet(temp_home)
        _prioritise(h, n2=10, n3=-1)
        priorities = h.engine._resolve_priorities()
        assert dict(priorities) == {"2": 10, "3": -1}
        assert priorities["1"] == 0


class TestNoPrioritiesIsUnchanged:
    """With nothing set, every path below decides exactly as before."""

    def test_below_the_line_holds(self, temp_home):
        h = _fleet(temp_home)
        outcome = h.tick_with_usage(
            {"1": _personal(20), "2": _team(0), "3": _team(0)}
        )
        assert outcome is TickOutcome.NO_ACTION
        assert _reasons(h) == ["below-threshold"]

    def test_proactive_takes_the_most_headroom(self, temp_home):
        h = _fleet(temp_home)
        outcome = h.tick_with_usage(
            {"1": _personal(65), "2": _team(30), "3": _team(5)}
        )
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3
        assert _switch(h).trigger == "proactive"

    def test_all_zero_priorities_store_nothing(self, temp_home):
        h = _fleet(temp_home)
        _prioritise(h, n2=0)
        record = h.switcher._get_sequence_data()["accounts"]["2"]
        assert "priority" not in record


class TestPriorityReturn:
    """Active under its line; a healthier, higher-priority account exists."""

    def test_it_switches_to_the_higher_priority_account(self, temp_home):
        h = _fleet(temp_home)
        _prioritise(h, n2=10)
        outcome = h.tick_with_usage(
            {"1": _personal(20), "2": _team(10), "3": _team(0)}
        )
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 2
        sw = _switch(h)
        assert sw.trigger == "priority"
        assert sw.to_json()["trigger"] == "priority"
        assert h.state()["leftTrigger"] == "priority"

    def test_the_highest_priority_wins_over_more_headroom(self, temp_home):
        h = _fleet(temp_home)
        _prioritise(h, n2=5, n3=10)
        outcome = h.tick_with_usage(
            {"1": _personal(20), "2": _team(0), "3": _team(40)}
        )
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 3

    def test_a_negative_priority_active_returns_to_the_default(self, temp_home):
        h = _fleet(temp_home)
        _prioritise(h, n1=-1)
        outcome = h.tick_with_usage(
            {"1": _personal(20), "2": _team(30), "3": _team(10)}
        )
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 3

    def test_not_when_the_target_is_over_its_line(self, temp_home):
        h = _fleet(temp_home)
        _prioritise(h, n2=10)
        outcome = h.tick_with_usage(
            {"1": _personal(20), "2": _team(60), "3": _team(0)}
        )
        assert outcome is TickOutcome.NO_ACTION
        assert _reasons(h) == ["below-threshold"]
        assert h.active_number() == 1

    def test_not_inside_the_hysteresis_margin(self, temp_home):
        """55% sits 5 points under a 60 line; hysteresisPct is 10."""
        h = _fleet(temp_home)
        _prioritise(h, n2=10)
        outcome = h.tick_with_usage(
            {"1": _personal(20), "2": _team(55), "3": _team(0)}
        )
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1

    def test_exactly_the_margin_is_enough(self, temp_home):
        h = _fleet(temp_home)
        _prioritise(h, n2=10)
        outcome = h.tick_with_usage(
            {"1": _personal(20), "2": _team(50), "3": _team(0)}
        )
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 2

    def test_not_onto_a_weekly_window_over_its_line(self, temp_home):
        h = _fleet(temp_home, active=2)
        _prioritise(h, n1=10)
        outcome = h.tick_with_usage(
            {"1": _personal(0, 97), "2": _team(20), "3": _team(0)}
        )
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 2

    def test_not_on_stale_data(self, temp_home):
        """Decision-trusted but past the serve TTL: provisional pick, then the
        per-target freshness gate holds instead of switching."""
        h = _fleet(temp_home)
        _prioritise(h, n2=10)
        now = h.clock.now
        entries = {
            "1": _entry_for(_personal(20), now),
            "2": UsageEntry(last_good=_team(10), fetched_at=now - 240, age_s=240.0),
            "3": _entry_for(_team(55), now),
        }
        outcome = h.tick_with_entries(entries)
        assert outcome is TickOutcome.NO_ACTION
        assert "stale-usage" in _reasons(h)
        assert h.active_number() == 1

    def test_not_during_the_cooldown(self, temp_home):
        h = _fleet(temp_home)
        _prioritise(h, n2=10)
        (h.switcher.backup_dir / "autoswitch_state.json").write_text(
            json.dumps({"lastSwitchAt": h.clock.now - 10})
        )
        outcome = h.tick_with_usage(
            {"1": _personal(20), "2": _team(10), "3": _team(0)}
        )
        assert outcome is TickOutcome.NO_ACTION
        assert _reasons(h) == ["cooldown"]
        h.clock.advance(h.settings.cooldown_seconds)
        h.events.clear()
        outcome = h.tick_with_usage(
            {"1": _personal(20), "2": _team(10), "3": _team(0)}
        )
        assert outcome is TickOutcome.SWITCHED

    def test_not_to_an_equal_priority(self, temp_home):
        h = _fleet(temp_home, active=2)
        _prioritise(h, n2=10, n3=10)
        outcome = h.tick_with_usage(
            {"1": _personal(0), "2": _team(30), "3": _team(0)}
        )
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 2

    def test_not_to_a_lower_priority(self, temp_home):
        h = _fleet(temp_home, active=2)
        _prioritise(h, n2=10)
        outcome = h.tick_with_usage(
            {"1": _personal(0), "2": _team(30), "3": _team(0)}
        )
        assert outcome is TickOutcome.NO_ACTION
        assert _reasons(h) == ["below-threshold"]

    def test_no_flapping_back(self, temp_home):
        h = _fleet(temp_home)
        _prioritise(h, n2=10)
        assert h.tick_with_usage(
            {"1": _personal(20), "2": _team(10), "3": _team(0)}
        ) is TickOutcome.SWITCHED
        for _ in range(5):
            h.clock.advance(h.settings.cooldown_seconds + 1)
            h.events.clear()
            outcome = h.tick_with_usage(
                {"1": _personal(5), "2": _team(40), "3": _team(0)}
            )
            assert outcome is TickOutcome.NO_ACTION
            assert h.active_number() == 2

    def test_dry_run_reports_and_does_not_switch(self, temp_home):
        h = _fleet(temp_home)
        _prioritise(h, n2=10)
        h.engine = h._make_engine(dry_run=True)
        outcome = h.tick_with_usage(
            {"1": _personal(20), "2": _team(10), "3": _team(0)}
        )
        assert outcome is TickOutcome.SWITCHED
        assert _switch(h).dry_run and _switch(h).trigger == "priority"
        assert h.active_number() == 1


class TestPriorityOrdersTheForcedPaths:
    def test_proactive_prefers_priority_over_headroom(self, temp_home):
        h = _fleet(temp_home)
        _prioritise(h, n2=10)
        outcome = h.tick_with_usage(
            {"1": _personal(65), "2": _team(30), "3": _team(5)}
        )
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2
        assert _switch(h).trigger == "proactive"

    def test_proactive_never_lands_over_a_line_for_priority(self, temp_home):
        h = _fleet(temp_home)
        _prioritise(h, n2=10)
        outcome = h.tick_with_usage(
            {"1": _personal(65), "2": _team(61), "3": _team(5)}
        )
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_at_limit_prefers_priority_among_healthy(self, temp_home):
        h = _fleet(temp_home)
        _prioritise(h, n2=10)
        outcome = h.tick_with_usage(
            {"1": _personal(100), "2": _team(30), "3": _team(5)}
        )
        assert outcome is TickOutcome.SWITCHED
        assert _switch(h).trigger == "at-limit"
        assert h.active_number() == 2

    def test_at_limit_puts_a_healthy_account_before_an_over_line_priority(
        self, temp_home
    ):
        h = _fleet(temp_home)
        _prioritise(h, n2=10, n3=-5)
        outcome = h.tick_with_usage(
            {"1": _personal(100), "2": _team(80), "3": _team(5)}
        )
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_failover_prefers_priority(self, temp_home):
        h = _fleet(temp_home, unhealthy_ticks=1)
        _prioritise(h, n2=10)
        outcome = h.tick_with_usage({"1": None, "2": _team(30), "3": _team(5)})
        assert outcome is TickOutcome.SWITCHED
        assert _switch(h).trigger == "failover"
        assert h.active_number() == 2

    def test_all_above_keeps_the_soonest_recovery(self, temp_home):
        """Every account over its line: the account back first wins even
        against a higher priority."""
        h = _fleet(temp_home)
        _prioritise(h, n2=10)
        now = h.clock.now
        outcome = h.tick_with_usage({
            "1": _usage(70, _iso_at(now + 3 * 3600)),
            "2": _team(70, _iso_at(now + 2 * 3600)),
            "3": _team(70, _iso_at(now + 600)),
        })
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 3

    def test_the_last_priority_account_falls_back_unbarred(self, temp_home):
        """Left by a `priority` move, account 1 is not held by the no-return
        bar once the account we moved to is spent: going back is the fallback
        the user configured, not a burn-driven flip."""
        h = _fleet(temp_home)
        _prioritise(h, n2=10, n3=10)
        assert h.tick_with_usage(
            {"1": _personal(30), "2": _team(10), "3": _team(61)}
        ) is TickOutcome.SWITCHED
        assert h.active_number() == 2
        h.clock.advance(h.settings.cooldown_seconds + 1)
        h.events.clear()
        outcome = h.tick_with_usage(
            {"1": _personal(30), "2": _team(65), "3": _team(70)}
        )
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 1
        assert _switch(h).trigger == "proactive"


class TestConsumeFirstWithPriority:
    def test_the_nudge_never_goes_to_a_lower_priority(self, temp_home):
        h = _fleet(temp_home, strategy="consume-first")
        _prioritise(h, n1=10)
        outcome = h.tick_with_usage({
            "1": _usage7(20, 20, _R_LATER),
            "2": _usage7(10, 10, _R_SOON),
            "3": _usage7(10, 10, _R_LATEST),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1

    def test_without_priority_the_same_fleet_nudges(self, temp_home):
        h = _fleet(temp_home, strategy="consume-first")
        outcome = h.tick_with_usage({
            "1": _usage7(20, 20, _R_LATER),
            "2": _usage7(10, 10, _R_SOON),
            "3": _usage7(10, 10, _R_LATEST),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_priority_return_ranks_by_priority_then_reset(self, temp_home):
        h = _fleet(temp_home, strategy="consume-first")
        _prioritise(h, n2=10, n3=10)
        outcome = h.tick_with_usage({
            "1": _usage7(20, 20, _R_SOON),
            "2": _usage7(10, 10, _R_LATEST),
            "3": _usage7(10, 10, _R_LATER),
        })
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert _switch(h).trigger == "priority"
        assert h.active_number() == 3
