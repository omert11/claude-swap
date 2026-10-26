"""Cooldown scope and the live usage probe (`autoswitch.liveProbe`).

Cooldown: it spaces out VOLUNTARY moves only (consume-first, priority, soft).
A switch forced by a hard line never waits on it — the incident it fixes held
an account from 91% to 100% for three minutes on a 300s cooldown after an
at-limit switch had just landed on it.

Live probe: one minimal Messages request with the active account's token,
whose `anthropic-ratelimit-unified-*` headers carry the 5h/7d windows live.
It runs only when enabled and only when the stored reading is near the line
the next decision turns on, or stale; it never raises, and it backs off after
failures. HTTP is mocked throughout — nothing here touches the network.

Synthetic fixtures only: `EngineHarness` seeds a throwaway store under
`temp_home`.
"""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from claude_swap import oauth
from claude_swap.autoswitch import (
    LIVE_PROBE_BACKOFF_S,
    NoSwitchEvent,
    PollEvent,
    SwitchEvent,
    TickOutcome,
)
from claude_swap.settings import AutoSwitchSettings, merged_with_cli
from claude_swap.usage_store import UsageEntry

from tests.test_autoswitch import EngineHarness, _entry_for

PROD = {
    "threshold": 90.0,
    "soft_threshold": 60.0,
    "weekly_threshold": 97.0,
    "hysteresis_pct": 5.0,
    "cooldown_seconds": 60.0,
    "interval_seconds": 60.0,
}
RESET_5H = 1793023200
RESET_7D = 1793178000


def _team(pct5: float) -> dict:
    return {"five_hour": {"pct": pct5}}


def _personal(pct5: float, pct7: float = 30.0) -> dict:
    return {"five_hour": {"pct": pct5}, "seven_day": {"pct": pct7}}


def _headers(
    u5: float | None = 0.26,
    u7: float | None = 0.93,
    *,
    s5: str = "allowed",
    s7: str = "allowed_warning",
    overall: str = "allowed",
    claim: str = "seven_day",
) -> dict:
    h = {
        "anthropic-ratelimit-unified-status": overall,
        "anthropic-ratelimit-unified-representative-claim": claim,
    }
    if u5 is not None:
        h["anthropic-ratelimit-unified-5h-utilization"] = str(u5)
        h["anthropic-ratelimit-unified-5h-reset"] = str(RESET_5H)
        h["anthropic-ratelimit-unified-5h-status"] = s5
    if u7 is not None:
        h["anthropic-ratelimit-unified-7d-utilization"] = str(u7)
        h["anthropic-ratelimit-unified-7d-reset"] = str(RESET_7D)
        h["anthropic-ratelimit-unified-7d-status"] = s7
    return h


def _fleet(temp_home, *, live_probe: bool = True, **overrides) -> EngineHarness:
    """Team accounts 1-3 at priority 10, account 1 live and active."""
    h = EngineHarness(temp_home, **{**PROD, "live_probe": live_probe, **overrides})
    for num in (1, 2, 3):
        h.seed(num, f"team-{num}@example.com")
        h.switcher.set_account_priority(str(num), 10)
    h.make_live("team-1@example.com", 1)
    return h


def _reasons(h: EngineHarness) -> list[str]:
    return [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]


def _switch(h: EngineHarness) -> SwitchEvent:
    return next(e for e in h.events if isinstance(e, SwitchEvent))


def _poll(h: EngineHarness) -> PollEvent:
    return next(e for e in h.events if isinstance(e, PollEvent))


def _probe_with(headers: dict | None = None, *, status: int = 200, error=None):
    """Patch the HTTP call under `oauth.probe_live_usage`; returns the mock."""
    if error is not None:
        return patch.object(oauth, "request_live_ratelimit", side_effect=error)
    return patch.object(
        oauth, "request_live_ratelimit", return_value=(status, headers)
    )


# -- A: cooldown scope ---------------------------------------------------------


class TestCooldownGatesOnlyVoluntaryMoves:
    def _just_switched(self, h: EngineHarness, ago: float = 10.0) -> None:
        h.engine._mutate_state(lambda s: s.update(lastSwitchAt=h.clock() - ago))

    def test_the_incident_over_the_hard_line_switches_inside_the_cooldown(
        self, temp_home
    ):
        """15:33:01 at-limit onto #6, 15:33:57 #6 at 91% -> must not wait."""
        h = _fleet(temp_home, live_probe=False, cooldown_seconds=300.0)
        self._just_switched(h, ago=56.0)
        outcome = h.tick_with_usage({"1": _team(91), "2": _team(30), "3": _team(95)})
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert _switch(h).trigger == "proactive"
        assert h.active_number() == 2

    def test_a_weekly_line_switch_also_bypasses_it(self, temp_home):
        h = _fleet(temp_home, live_probe=False)
        self._just_switched(h)
        outcome = h.tick_with_usage(
            {"1": _personal(10, 98), "2": _team(30), "3": _team(95)}
        )
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert _switch(h).trigger == "proactive"

    def test_a_soft_move_still_waits(self, temp_home):
        h = _fleet(temp_home, live_probe=False)
        self._just_switched(h)
        outcome = h.tick_with_usage({"1": _team(65), "2": _team(10), "3": _team(95)})
        assert outcome is TickOutcome.NO_ACTION
        assert _reasons(h) == ["cooldown"]

    def test_a_priority_move_still_waits(self, temp_home):
        h = _fleet(temp_home, live_probe=False)
        h.switcher.set_account_priority("1", 0)
        self._just_switched(h)
        outcome = h.tick_with_usage({"1": _team(20), "2": _team(10), "3": _team(95)})
        assert outcome is TickOutcome.NO_ACTION
        assert _reasons(h) == ["cooldown"]

    def test_no_flap_without_the_cooldown(self, temp_home):
        """Back-to-back ticks after a forced move: the account left is over
        its line, so nothing returns to it."""
        h = _fleet(temp_home, live_probe=False)
        assert h.tick_with_usage(
            {"1": _team(91), "2": _team(30), "3": _team(95)}
        ) is TickOutcome.SWITCHED
        for pct2 in (35, 45, 58):
            h.events.clear()
            outcome = h.tick_with_usage(
                {"1": _team(91), "2": _team(pct2), "3": _team(95)}
            )
            assert outcome is TickOutcome.NO_ACTION, _reasons(h)
            assert h.active_number() == 2


# -- B: header parsing and the HTTP boundary ----------------------------------


class TestParseRatelimitHeaders:
    def test_both_windows_in_percent_with_resets(self):
        windows = oauth.parse_ratelimit_headers(_headers(0.26, 0.93))
        assert windows["five_hour"]["pct"] == pytest.approx(26.0)
        assert windows["seven_day"]["pct"] == pytest.approx(93.0)
        assert windows["five_hour"]["resets_at"] == "2026-10-26T14:00:00Z"
        assert windows["seven_day"]["resets_at"].endswith("Z")
        assert "countdown" in windows["five_hour"]

    def test_a_team_account_without_7d_headers_reports_only_5h(self):
        windows = oauth.parse_ratelimit_headers(_headers(0.91, None, claim="five_hour"))
        assert set(windows) == {"five_hour"}

    def test_a_rejected_window_reads_100(self):
        windows = oauth.parse_ratelimit_headers(
            _headers(0.97, 0.5, s5="rejected", overall="rejected", claim="five_hour")
        )
        assert windows["five_hour"]["pct"] == 100.0
        assert windows["seven_day"]["pct"] == pytest.approx(50.0)

    def test_an_overall_rejection_marks_the_claimed_window(self):
        windows = oauth.parse_ratelimit_headers(
            _headers(0.5, 0.99, overall="rejected", claim="seven_day")
        )
        assert windows["seven_day"]["pct"] == 100.0
        assert windows["five_hour"]["pct"] == pytest.approx(50.0)

    def test_garbage_values_are_skipped(self):
        h = _headers(None, None)
        h["anthropic-ratelimit-unified-5h-utilization"] = "n/a"
        assert oauth.parse_ratelimit_headers(h) == {}


class TestProbeLiveUsage:
    def test_the_request_looks_like_claude_code(self):
        seen = {}

        class _Resp:
            status = 200
            headers = _headers()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout):
            seen["req"] = req
            return _Resp()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            outcome = oauth.probe_live_usage("sk-test")
        req = seen["req"]
        body = json.loads(req.data)
        assert req.get_method() == "POST"
        assert req.full_url == oauth.LIVE_PROBE_URL
        assert body["model"] == oauth.LIVE_PROBE_MODEL
        assert body["max_tokens"] == 1
        assert "Claude Code" in body["system"][0]["text"]
        headers = {k.lower(): v for k, v in req.header_items()}
        assert headers["authorization"] == "Bearer sk-test"
        assert "claude-code-20250219" in headers["anthropic-beta"]
        assert "oauth-2025-04-20" in headers["anthropic-beta"]
        assert headers["x-app"] == "cli"
        assert outcome.error is None
        assert outcome.windows["five_hour"]["pct"] == pytest.approx(26.0)

    def test_a_rejected_429_with_headers_is_a_reading(self):
        err = urllib.error.HTTPError(
            oauth.LIVE_PROBE_URL, 429, "Too Many Requests",
            _headers(1.0, None, s5="rejected", overall="rejected", claim="five_hour"),
            io.BytesIO(b""),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            outcome = oauth.probe_live_usage("sk-test")
        assert outcome.error is None
        assert outcome.windows["five_hour"]["pct"] == 100.0

    def test_a_429_without_headers_is_an_error(self):
        err = urllib.error.HTTPError(
            oauth.LIVE_PROBE_URL, 429, "Too Many Requests", {}, io.BytesIO(b"")
        )
        with patch("urllib.request.urlopen", side_effect=err):
            outcome = oauth.probe_live_usage("sk-test")
        assert outcome.windows is None
        assert outcome.error == "http-429"

    @pytest.mark.parametrize(
        ("exc", "kind"),
        [
            (urllib.error.URLError("down"), "network"),
            (TimeoutError(), "timeout"),
            (urllib.error.HTTPError(oauth.LIVE_PROBE_URL, 500, "x", {}, io.BytesIO(b"")),
             "http-500"),
        ],
    )
    def test_transport_errors_never_raise(self, exc, kind):
        with patch("urllib.request.urlopen", side_effect=exc):
            outcome = oauth.probe_live_usage("sk-test")
        assert outcome.windows is None
        assert outcome.error == kind

    def test_the_token_is_never_logged(self, caplog):
        with caplog.at_level("DEBUG", logger="claude-swap"), patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("sk-secret")
        ):
            oauth.probe_live_usage("sk-secret")
        assert "sk-secret" not in caplog.text


# -- B: the engine -------------------------------------------------------------


class TestLiveProbeGating:
    def test_off_by_default(self, temp_home):
        assert AutoSwitchSettings().live_probe is False
        h = _fleet(temp_home, live_probe=False)
        with _probe_with(_headers(0.99)) as probe:
            h.tick_with_usage({"1": _team(72), "2": _team(10), "3": _team(10)})
        probe.assert_not_called()

    def test_not_far_from_the_line_with_fresh_data(self, temp_home):
        """Soft line 60, margin 15: a fresh 40% is not probed."""
        h = _fleet(temp_home)
        with _probe_with(_headers(0.99)) as probe:
            outcome = h.tick_with_usage({"1": _team(40), "2": _team(10), "3": _team(10)})
        probe.assert_not_called()
        assert outcome is TickOutcome.NO_ACTION

    def test_near_the_soft_line(self, temp_home):
        h = _fleet(temp_home)
        with _probe_with(_headers(0.46, None)) as probe:
            h.tick_with_usage({"1": _team(45), "2": _team(10), "3": _team(10)})
        probe.assert_called_once()

    def test_near_the_hard_line_without_a_soft_line(self, temp_home):
        h = _fleet(temp_home, soft_threshold=None)
        with _probe_with(_headers(0.5, None)) as probe:
            h.tick_with_usage({"1": _team(60), "2": _team(10), "3": _team(10)})
        probe.assert_not_called()
        with _probe_with(_headers(0.8, None)) as probe:
            h.tick_with_usage({"1": _team(75), "2": _team(10), "3": _team(10)})
        probe.assert_called_once()

    def test_a_weekly_window_near_its_own_line(self, temp_home):
        h = _fleet(temp_home)
        with _probe_with(_headers(0.1, 0.9)) as probe:
            h.tick_with_usage(
                {"1": _personal(10, 85), "2": _team(10), "3": _team(10)}
            )
        probe.assert_called_once()

    def test_a_stale_reading_far_from_the_line(self, temp_home):
        h = _fleet(temp_home)
        now = h.clock.now
        entries = {
            "1": UsageEntry(last_good=_team(20), fetched_at=now - 150, age_s=150.0),
            "2": _entry_for(_team(10), now),
            "3": _entry_for(_team(10), now),
        }
        with _probe_with(_headers(0.2, None)) as probe:
            h.tick_with_entries(entries)
        probe.assert_called_once()

    def test_a_sentinel_row_is_never_probed(self, temp_home):
        h = _fleet(temp_home)
        with _probe_with(_headers(0.9)) as probe:
            h.tick_with_usage(
                {"1": "token_expired", "2": _team(10), "3": _team(10)}
            )
        probe.assert_not_called()

    def test_an_expired_live_token_sends_nothing(self, temp_home):
        h = _fleet(temp_home)
        (temp_home / ".claude" / ".credentials.json").write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "sk-live", "refreshToken": "rt", "expiresAt": 1000,
            },
        }))
        with _probe_with(_headers(0.99)) as probe:
            h.tick_with_usage({"1": _team(72), "2": _team(10), "3": _team(10)})
        probe.assert_not_called()


class TestLiveProbeDecides:
    def test_frozen_stored_value_plus_live_value_over_the_hard_line_switches(
        self, temp_home
    ):
        """The incident: the usage endpoint keeps serving 72% while the
        account is really at 91%."""
        h = _fleet(temp_home)
        with _probe_with(_headers(0.91, None, claim="five_hour")):
            outcome = h.tick_with_usage(
                {"1": _team(72), "2": _team(58), "3": _team(95)}
            )
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert _switch(h).trigger == "proactive"
        assert h.active_number() == 2

    def test_the_same_frozen_value_without_the_probe_holds(self, temp_home):
        h = _fleet(temp_home, live_probe=False)
        outcome = h.tick_with_usage({"1": _team(72), "2": _team(58), "3": _team(95)})
        assert outcome is TickOutcome.NO_ACTION

    def test_a_rejected_429_triggers_the_at_limit_escape(self, temp_home):
        h = _fleet(temp_home)
        rejected = _headers(1.0, None, s5="rejected", overall="rejected", claim="five_hour")
        with _probe_with(rejected, status=429):
            outcome = h.tick_with_usage(
                {"1": _team(72), "2": _team(58), "3": _team(95)}
            )
        assert outcome is TickOutcome.SWITCHED
        assert _switch(h).trigger == "at-limit"

    def test_the_log_line_names_the_live_source(self, temp_home):
        h = _fleet(temp_home)
        with _probe_with(_headers(0.55, None)):
            h.tick_with_usage({"1": _team(50), "2": _team(20), "3": _team(20)})
        poll = _poll(h)
        assert poll.live is True
        assert poll.to_json()["liveProbe"] is True
        assert poll.windows["1"]["5h"] == pytest.approx(55.0)
        assert "5h 55% used (live)" in poll.human()

    def test_without_a_probe_the_log_line_is_unchanged(self, temp_home):
        h = _fleet(temp_home, live_probe=False)
        h.tick_with_usage({"1": _team(50), "2": _team(20), "3": _team(20)})
        poll = _poll(h)
        assert "liveProbe" not in poll.to_json()
        assert "(live)" not in poll.human()

    def test_a_missing_7d_keeps_the_stored_weekly_window(self, temp_home):
        h = _fleet(temp_home)
        with _probe_with(_headers(0.5, None)):
            h.tick_with_usage({"1": _personal(50, 40), "2": _team(20), "3": _team(20)})
        assert _poll(h).windows["1"] == {"5h": pytest.approx(50.0), "7d": 40.0}

    def test_the_reading_is_persisted_fresh(self, temp_home):
        h = _fleet(temp_home)
        with _probe_with(_headers(0.55, None)):
            h.tick_with_usage({"1": _team(50), "2": _team(20), "3": _team(20)})
        stored = h.switcher.usage_entries_by_account(fetch=set())["1"]
        assert stored.last_good["five_hour"]["pct"] == pytest.approx(55.0)
        assert stored.age_s == pytest.approx(0.0)

    def test_dry_run_probes_but_writes_nothing(self, temp_home):
        h = _fleet(temp_home)
        h.engine = h._make_engine(dry_run=True)
        with _probe_with(_headers(0.55, None)):
            h.tick_with_usage({"1": _team(50), "2": _team(20), "3": _team(20)})
        assert _poll(h).live is True
        stored = h.switcher.usage_entries_by_account(fetch=set())["1"]
        assert stored.last_good is None

    def test_the_escalation_refetch_does_not_overwrite_the_live_value(
        self, temp_home
    ):
        h = _fleet(temp_home)
        fetch_sets: list[set] = []
        now = h.clock.now
        entries = {n: _entry_for(_team(p), now) for n, p in
                   (("1", 72), ("2", 58), ("3", 95))}

        def collect(fetch=None, **_kw):
            fetch_sets.append(set(fetch or ()))
            return entries

        with _probe_with(_headers(0.91, None)), patch.object(
            h.switcher, "usage_entries_by_account", side_effect=collect
        ):
            outcome = h.engine.tick()
        assert outcome is TickOutcome.SWITCHED
        assert {"2", "3"} in fetch_sets          # candidates escalated
        assert all("1" not in s for s in fetch_sets[1:])


class TestLiveProbeFailures:
    def test_an_error_falls_back_to_the_stored_reading(self, temp_home):
        h = _fleet(temp_home)
        with _probe_with(error=urllib.error.URLError("down")):
            outcome = h.tick_with_usage(
                {"1": _team(72), "2": _team(58), "3": _team(95)}
            )
        assert outcome is TickOutcome.NO_ACTION
        assert _poll(h).live is False
        assert "error" not in h.kinds()

    def test_a_raising_switcher_never_crashes_the_tick(self, temp_home):
        h = _fleet(temp_home)
        with patch.object(
            h.switcher, "probe_active_usage", side_effect=RuntimeError("boom")
        ):
            outcome = h.tick_with_usage(
                {"1": _team(72), "2": _team(58), "3": _team(95)}
            )
        assert outcome is TickOutcome.NO_ACTION

    def test_failures_back_off_and_success_resets(self, temp_home):
        h = _fleet(temp_home)
        usage = {"1": _team(72), "2": _team(58), "3": _team(95)}
        with _probe_with(error=urllib.error.URLError("down")) as probe:
            h.tick_with_usage(usage)                     # fail 1 -> 60s
            h.clock.advance(30)
            h.tick_with_usage(usage)                     # inside backoff
            assert probe.call_count == 1
            h.clock.advance(LIVE_PROBE_BACKOFF_S)
            h.tick_with_usage(usage)                     # fail 2 -> 120s
            assert probe.call_count == 2
            h.clock.advance(LIVE_PROBE_BACKOFF_S + 1)
            h.tick_with_usage(usage)                     # still backing off
            assert probe.call_count == 2
        h.clock.advance(LIVE_PROBE_BACKOFF_S)
        with _probe_with(_headers(0.7, None)) as probe:
            h.tick_with_usage(usage)
            assert probe.call_count == 1
        assert h.engine._probe_failures == 0
        assert h.engine._probe_backoff_until is None

    def test_a_retry_after_longer_than_the_backoff_wins(self, temp_home):
        h = _fleet(temp_home)
        err = urllib.error.HTTPError(
            oauth.LIVE_PROBE_URL, 503, "x", {"Retry-After": "900"}, io.BytesIO(b"")
        )
        with _probe_with(error=err):
            h.tick_with_usage({"1": _team(72), "2": _team(58), "3": _team(95)})
        assert h.engine._probe_backoff_until == pytest.approx(h.clock.now + 900)


class TestLiveProbeSettings:
    def test_the_cli_flag_overrides(self):
        from argparse import Namespace

        merged = merged_with_cli(AutoSwitchSettings(), Namespace(live_probe=True))
        assert merged.live_probe is True

    def test_a_30s_interval_is_allowed(self, tmp_path):
        from claude_swap.settings import load_settings, set_setting

        set_setting(tmp_path, "autoswitch.intervalSeconds", "30")
        set_setting(tmp_path, "autoswitch.liveProbe", "true")
        loaded = load_settings(tmp_path)
        assert (loaded.interval_seconds, loaded.live_probe) == (30.0, True)
