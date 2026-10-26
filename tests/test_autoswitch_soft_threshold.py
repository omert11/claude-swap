"""The soft line (`autoswitch.softThreshold`) and its interplay with priority.

Two lines per account on the windows `threshold` governs: at/over the soft
line the active account moves voluntarily to an account of equal or higher
priority still under ITS soft line by the hysteresis margin (the ``soft``
trigger), else stays until the hard line; the hard-switch ranking is
priority, then under-the-soft-line, then the strategy key; and the priority
return asks its target to be under its soft line.

The fleet is the one the feature was asked for: account 1 a personal account
with a weekly window (priority 0), accounts 2-4 Team accounts with only a 5h
window (priority 10), `threshold=90`, `softThreshold=60`,
`weeklyThreshold=97`. Every engine test drives a real tick.

Synthetic fixtures only: `EngineHarness` seeds a throwaway store under
`temp_home`.
"""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from claude_swap.autoswitch import NoSwitchEvent, SwitchEvent, TickOutcome
from claude_swap.exceptions import ConfigError
from claude_swap.settings import (
    AutoSwitchSettings,
    load_settings,
    merged_with_cli,
    set_setting,
)

from tests.test_autoswitch import _R_LATER, _R_SOON, EngineHarness, _usage7
from tests.test_tui import FakeSwitcher, make_account, make_app, make_entry, settle
from tests.test_tui import fake_engine  # noqa: F401 - fixture used by name
from tests.test_tui_policy_display import with_policy

SETTINGS = {"threshold": 90.0, "soft_threshold": 60.0, "weekly_threshold": 97.0}
EMAILS = {
    1: "personal@example.com",
    2: "team-b@example.com",
    3: "team-c@example.com",
    4: "team-d@example.com",
}


def _team(pct5: float) -> dict:
    return {"five_hour": {"pct": pct5}}


def _personal(pct5: float, pct7: float = 30.0) -> dict:
    return _usage7(pct5, pct7)


def _fleet(
    temp_home, *, active: int = 2, teams: tuple[int, ...] = (2, 3, 4), **settings
) -> EngineHarness:
    h = EngineHarness(temp_home, **{**SETTINGS, **settings})
    h.seed(1, EMAILS[1])
    for num in teams:
        h.seed(num, EMAILS[num])
        h.switcher.set_account_priority(str(num), 10)
    data = h.switcher._get_sequence_data()
    data["activeAccountNumber"] = active
    h.switcher._write_json(h.switcher.sequence_file, data)
    h.make_live(EMAILS[active], active)
    return h


def _reasons(h: EngineHarness) -> list[str]:
    return [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]


def _switch(h: EngineHarness) -> SwitchEvent:
    return next(e for e in h.events if isinstance(e, SwitchEvent))


def _next_tick(h: EngineHarness, usage: dict) -> TickOutcome:
    h.clock.advance(h.settings.cooldown_seconds + 1)
    h.events.clear()
    return h.tick_with_usage(usage)


class TestSoftUnsetIsUnchanged:
    def test_the_default_is_off(self):
        assert AutoSwitchSettings().soft_threshold is None

    def test_between_the_lines_it_holds(self, temp_home):
        h = EngineHarness(temp_home, threshold=90.0)
        h.seed(1, EMAILS[1])
        h.seed(2, EMAILS[2])
        h.make_live(EMAILS[1], 1)
        outcome = h.tick_with_usage({"1": _personal(70), "2": _team(0)})
        assert outcome is TickOutcome.NO_ACTION
        assert _reasons(h) == ["below-threshold"]

    def test_cadence_is_pinned_to_the_hard_line(self, temp_home):
        h = EngineHarness(temp_home, threshold=90.0)
        h.seed(1, EMAILS[1])
        assert h.engine._min_effective_threshold() == 90.0


class TestSoftTrigger:
    def test_over_the_soft_line_it_moves_to_a_peer_under_its_own(self, temp_home):
        h = _fleet(temp_home)
        outcome = h.tick_with_usage(
            {"1": _personal(0), "2": _team(65), "3": _team(20), "4": _team(58)}
        )
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 3
        assert _switch(h).trigger == "soft"

    def test_under_the_soft_line_it_holds(self, temp_home):
        h = _fleet(temp_home)
        outcome = h.tick_with_usage(
            {"1": _personal(0), "2": _team(55), "3": _team(0), "4": _team(0)}
        )
        assert outcome is TickOutcome.NO_ACTION
        assert _reasons(h) == ["below-threshold"]

    def test_never_to_a_lower_priority(self, temp_home):
        """Only the personal account is under 60: stay until the hard line."""
        h = _fleet(temp_home)
        outcome = h.tick_with_usage(
            {"1": _personal(0), "2": _team(65), "3": _team(70), "4": _team(62)}
        )
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 2

    def test_the_target_needs_the_hysteresis_margin(self, temp_home):
        h = _fleet(temp_home)
        outcome = h.tick_with_usage(
            {"1": _personal(0), "2": _team(65), "3": _team(55), "4": _team(70)}
        )
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 2

    def test_it_respects_the_cooldown(self, temp_home):
        h = _fleet(temp_home)
        (h.switcher.backup_dir / "autoswitch_state.json").write_text(
            json.dumps({"lastSwitchAt": h.clock.now - 10})
        )
        outcome = h.tick_with_usage(
            {"1": _personal(0), "2": _team(65), "3": _team(20), "4": _team(70)}
        )
        assert outcome is TickOutcome.NO_ACTION
        assert _reasons(h) == ["cooldown"]

    def test_two_peers_around_the_soft_line_do_not_ping_pong(self, temp_home):
        h = _fleet(temp_home, teams=(2, 3))
        assert h.tick_with_usage(
            {"1": _personal(0), "2": _team(60), "3": _team(50)}
        ) is TickOutcome.SWITCHED
        assert h.active_number() == 3
        for pct3 in (55, 60, 61, 70):
            outcome = _next_tick(h, {"1": _personal(0), "2": _team(60), "3": _team(pct3)})
            assert outcome is TickOutcome.NO_ACTION, _reasons(h)
            assert h.active_number() == 3

    def test_a_per_account_hard_line_under_the_soft_one_disables_it(self, temp_home):
        h = _fleet(temp_home)
        h.switcher.set_account_threshold("2", 55.0)
        outcome = h.tick_with_usage(
            {"1": _personal(0), "2": _team(50), "3": _team(0), "4": _team(0)}
        )
        assert outcome is TickOutcome.NO_ACTION
        assert _reasons(h) == ["below-threshold"]

    def test_a_soft_departure_is_not_barred_at_the_hard_switch(self, temp_home):
        h = _fleet(temp_home, teams=(2, 3))
        assert h.tick_with_usage(
            {"1": _personal(0), "2": _team(62), "3": _team(10)}
        ) is TickOutcome.SWITCHED
        outcome = _next_tick(h, {"1": _personal(0), "2": _team(62), "3": _team(91)})
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 2


class TestHardSwitchRanking:
    def test_a_peer_under_its_soft_line_comes_first(self, temp_home):
        h = _fleet(temp_home)
        outcome = h.tick_with_usage(
            {"1": _personal(0), "2": _team(91), "3": _team(20), "4": _team(70)}
        )
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_soft_tier_beats_headroom_within_a_priority(self, temp_home):
        """Account 1 joins the Team level here. Under both its soft lines it
        holds less hard-line headroom (its 7d window at 80 binds) than Team 3
        at 65, which is over the soft line: the soft tier puts 1 first."""
        h = _fleet(temp_home)
        h.switcher.set_account_priority("1", 10)
        outcome = h.tick_with_usage(
            {"1": _personal(50, 80), "2": _team(91), "3": _team(65), "4": _team(95)}
        )
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 1

    def test_without_a_soft_line_headroom_decides_that_tie(self, temp_home):
        h = _fleet(temp_home, soft_threshold=None)
        h.switcher.set_account_priority("1", 10)
        outcome = h.tick_with_usage(
            {"1": _personal(50, 80), "2": _team(91), "3": _team(65), "4": _team(95)}
        )
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_then_a_peer_still_under_its_hard_line(self, temp_home):
        h = _fleet(temp_home)
        outcome = h.tick_with_usage(
            {"1": _personal(0), "2": _team(91), "3": _team(75), "4": _team(95)}
        )
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_only_then_the_lower_priority_account(self, temp_home):
        h = _fleet(temp_home)
        outcome = h.tick_with_usage(
            {"1": _personal(0), "2": _team(91), "3": _team(90), "4": _team(95)}
        )
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 1
        assert _switch(h).trigger == "proactive"


class TestPriorityReturnUsesTheSoftLine:
    def test_a_team_over_its_soft_line_is_not_returned_to(self, temp_home):
        h = _fleet(temp_home, active=1)
        outcome = h.tick_with_usage(
            {"1": _personal(10), "2": _team(55), "3": _team(70), "4": _team(91)}
        )
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1

    def test_a_reset_team_is_returned_to(self, temp_home):
        h = _fleet(temp_home, active=1)
        outcome = h.tick_with_usage(
            {"1": _personal(10), "2": _team(91), "3": _team(0), "4": _team(91)}
        )
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3
        assert _switch(h).trigger == "priority"


class TestConsumeFirstWithSoft:
    def test_the_nudge_never_lands_over_a_soft_line(self, temp_home):
        h = _fleet(temp_home, strategy="consume-first", teams=(2, 3))
        outcome = h.tick_with_usage({
            "1": _usage7(0, 0, _R_LATER),
            "2": _usage7(10, 10, _R_LATER),
            "3": _usage7(65, 10, _R_SOON),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 2


class TestTheRequestedFlow:
    """Teams rotate under 60, then run to 90, then the personal account; a
    Team that resets is taken straight back."""

    def test_full_flow(self, temp_home):
        h = _fleet(temp_home, active=1, teams=(2, 3))
        steps = [
            # (usage 1, 2, 3), expected active after, expected trigger or None
            ((10, 0, 0), 2, "priority"),
            ((10, 45, 0), 2, None),           # under soft: stay
            ((10, 62, 30), 3, "soft"),         # over soft, a Team under it
            ((10, 62, 61), 3, None),           # every Team over soft: stay
            ((10, 62, 80), 3, None),           # ... up to the hard line
            ((10, 62, 90), 2, "proactive"),    # hard line: a Team under 90
            ((10, 90, 90), 1, "proactive"),    # every Team at 90: personal
            ((20, 90, 90), 1, None),
            ((20, 90, 0), 3, "priority"),      # a Team reset: straight back
            ((20, 0, 65), 2, "soft"),          # on a Team over 60, other reset
        ]
        for (p1, p2, p3), want_active, want_trigger in steps:
            outcome = _next_tick(
                h, {"1": _personal(p1), "2": _team(p2), "3": _team(p3)}
            )
            assert h.active_number() == want_active, (p1, p2, p3, _reasons(h))
            if want_trigger is None:
                assert outcome is TickOutcome.NO_ACTION, (p1, p2, p3, _reasons(h))
            else:
                assert outcome is TickOutcome.SWITCHED
                assert _switch(h).trigger == want_trigger

    def test_cadence_tightens_at_the_soft_line(self, temp_home):
        h = _fleet(temp_home)
        assert h.engine._min_effective_threshold() == 60.0


class TestSoftSettings:
    def test_config_set_accepts_a_soft_line_below_the_hard_one(self, tmp_path):
        assert set_setting(tmp_path, "autoswitch.softThreshold", "60") == 60.0
        assert load_settings(tmp_path).soft_threshold == 60.0

    def test_config_set_refuses_a_soft_line_at_or_above_the_hard_one(self, tmp_path):
        with pytest.raises(ConfigError, match="below autoswitch.threshold"):
            set_setting(tmp_path, "autoswitch.softThreshold", "90")
        assert not (tmp_path / "settings.json").exists()

    def test_config_set_refuses_a_hard_line_at_or_below_the_soft_one(self, tmp_path):
        set_setting(tmp_path, "autoswitch.softThreshold", "60")
        with pytest.raises(ConfigError, match="above autoswitch.softThreshold"):
            set_setting(tmp_path, "autoswitch.threshold", "60")
        assert set_setting(tmp_path, "autoswitch.threshold", "61") == 61.0

    def test_out_of_range_is_refused(self, tmp_path):
        with pytest.raises(ConfigError):
            set_setting(tmp_path, "autoswitch.softThreshold", "20")

    def test_a_bad_stored_value_loads_as_off(self, tmp_path):
        (tmp_path / "settings.json").write_text(
            json.dumps({"autoswitch": {"softThreshold": "high"}})
        )
        assert load_settings(tmp_path).soft_threshold is None

    def test_the_cli_flag_overrides(self):
        args = Namespace(soft_threshold=70.0)
        assert merged_with_cli(AutoSwitchSettings(), args).soft_threshold == 70.0


@pytest.mark.asyncio
class TestNextBestShowsPriorityOrder:
    async def test_priority_orders_the_list(self, tmp_path, fake_engine):
        from textual.widgets import Static

        (tmp_path / "settings.json").write_text(json.dumps({
            "schemaVersion": 1,
            "autoswitch": {"threshold": 90},
        }))
        switcher = FakeSwitcher(
            [
                make_account(1, active=True, entry=make_entry(91.0, 20.0)),
                make_account(2, entry=make_entry(10.0, 10.0)),
                with_policy(make_account(3, entry=make_entry(50.0, 10.0)), priority=10),
            ],
            tmp_path,
        )
        app = make_app(switcher)
        async with app.run_test(size=(100, 40)) as pilot:
            await settle(pilot)
            await pilot.press("g")
            await pilot.pause()
            await settle(pilot)
            plain = app.screen.query_one("#candidates", Static).render().plain
            assert plain.index("user3@example.com") < plain.index("user2@example.com")
