"""The SessionStart context block.

This text is resident in every session the plugin touches, so it has two
competing obligations: be forceful enough to override a confident prior, and be
small enough that nobody minds paying for it.
"""

from __future__ import annotations

import datetime as dt

import pytest

from sota_anchor.catalog import build_catalog
from sota_anchor.seed import (
    BOOTSTRAP_BLOCK,
    SEED_FILENAME,
    is_stale,
    render_seed,
    seed_path,
    write_seed,
)

from .conftest import NOW, or_model, ts


@pytest.fixture
def catalog(raw_models):
    return build_catalog(raw_models, now=NOW)


class TestRenderSeed:
    def test_names_current_endpoints(self, catalog):
        assert "anthropic/claude-opus-5" in render_seed(catalog)

    def test_names_endpoints_from_every_focus_provider(self, catalog):
        rendered = render_seed(catalog)
        assert "google/" in rendered
        assert "openai/" in rendered

    def test_forbids_specific_superseded_endpoints(self, catalog):
        # A generic "don't use old models" does not dislodge a specific prior.
        rendered = render_seed(catalog)
        assert "anthropic/claude-opus-4.8" in rendered
        assert "openai/gpt-4o" in rendered

    def test_states_the_directive_not_merely_the_data(self, catalog):
        rendered = render_seed(catalog).lower()
        assert "do not" in rendered

    def test_tells_the_agent_its_priors_are_the_stale_part(self, catalog):
        rendered = render_seed(catalog).lower()
        assert "training" in rendered

    def test_points_at_the_verification_command(self, catalog):
        assert "/sota-check" in render_seed(catalog)

    def test_dates_itself_so_staleness_is_visible(self, catalog):
        assert "2026-09-19" in render_seed(catalog)

    def test_marks_a_cached_catalog_as_cached(self, raw_models):
        stale_catalog = build_catalog(raw_models, now=NOW, stale=True)
        assert "cached" in render_seed(stale_catalog).lower()

    def test_is_ascii_only(self, catalog):
        # It is JSON-escaped into a hook payload and printed to cp1252 consoles.
        assert render_seed(catalog).isascii()

    def test_stays_small_enough_to_live_in_every_session(self, catalog):
        # Resident cost: this is paid on every single session start.
        assert len(render_seed(catalog)) < 2600

    def test_omits_models_failing_the_capability_gate(self, catalog):
        assert "lyria" not in render_seed(catalog)

    def test_excludes_alias_namespaces(self, catalog):
        assert "~google" not in render_seed(catalog)

    def test_respects_the_provider_filter(self, catalog):
        rendered = render_seed(catalog, providers=["anthropic"])
        assert "anthropic/claude-opus-5" in rendered
        assert "google/" not in rendered


class TestBootstrapBlock:
    def test_is_used_when_no_catalog_has_ever_been_fetched(self):
        assert BOOTSTRAP_BLOCK.strip()

    def test_still_warns_that_model_knowledge_may_be_stale(self):
        assert "training" in BOOTSTRAP_BLOCK.lower()

    def test_names_no_specific_model(self):
        # With no catalog it must not guess; guessing is the failure mode.
        for guess in ("gpt-4o", "claude-3", "gemini-2.5", "claude-opus-5"):
            assert guess not in BOOTSTRAP_BLOCK

    def test_tells_the_user_how_to_populate_it(self):
        assert "/sota-sync" in BOOTSTRAP_BLOCK

    def test_is_ascii_only(self):
        assert BOOTSTRAP_BLOCK.isascii()


class TestSeedFile:
    def test_lives_beside_the_catalog_cache(self):
        from sota_anchor.catalog import default_cache_path

        assert seed_path().name == SEED_FILENAME
        assert seed_path().parent == default_cache_path().parent

    def test_writes_the_block(self, tmp_path, catalog):
        target = tmp_path / SEED_FILENAME
        write_seed(render_seed(catalog), path=target)
        assert "anthropic/claude-opus-5" in target.read_text(encoding="utf-8")

    def test_creates_missing_directories(self, tmp_path, catalog):
        target = tmp_path / "nested" / "deeper" / SEED_FILENAME
        write_seed(render_seed(catalog), path=target)
        assert target.exists()

    def test_writes_utf8_regardless_of_platform_default(self, tmp_path):
        target = tmp_path / SEED_FILENAME
        write_seed("café block", path=target)
        assert target.read_text(encoding="utf-8") == "café block"

    def test_replaces_previous_content_rather_than_appending(self, tmp_path):
        target = tmp_path / SEED_FILENAME
        write_seed("first", path=target)
        write_seed("second", path=target)
        assert target.read_text(encoding="utf-8") == "second"

    def test_leaves_no_temporary_files_behind(self, tmp_path):
        target = tmp_path / SEED_FILENAME
        write_seed("block", path=target)
        assert [p.name for p in tmp_path.iterdir()] == [SEED_FILENAME]


class TestStaleness:
    def test_a_missing_file_is_stale(self, tmp_path):
        assert is_stale(tmp_path / "absent.md", now=NOW) is True

    def test_a_fresh_file_is_not_stale(self, tmp_path):
        target = tmp_path / SEED_FILENAME
        write_seed("block", path=target)
        assert is_stale(target, now=dt.datetime.now(dt.UTC)) is False

    def test_a_file_older_than_the_ttl_is_stale(self, tmp_path):
        target = tmp_path / SEED_FILENAME
        write_seed("block", path=target)
        later = dt.datetime.now(dt.UTC) + dt.timedelta(hours=25)
        assert is_stale(target, now=later) is True

    def test_the_ttl_is_configurable(self, tmp_path):
        target = tmp_path / SEED_FILENAME
        write_seed("block", path=target)
        later = dt.datetime.now(dt.UTC) + dt.timedelta(hours=2)
        assert is_stale(target, now=later, ttl_hours=1) is True
        assert is_stale(target, now=later, ttl_hours=48) is False


class TestRefresh:
    async def test_writes_both_the_catalog_and_the_block(self, tmp_path, raw_models):
        from sota_anchor.seed import refresh

        async def fake_fetch(**kwargs):
            return build_catalog(raw_models, now=NOW)

        block = tmp_path / SEED_FILENAME
        await refresh(fetch=fake_fetch, seed_file=block)
        assert "anthropic/claude-opus-5" in block.read_text(encoding="utf-8")

    async def test_a_catalog_outage_leaves_any_existing_block_alone(self, tmp_path):
        from sota_anchor.catalog import CatalogUnavailable
        from sota_anchor.seed import refresh

        block = tmp_path / SEED_FILENAME
        write_seed("previous block", path=block)

        async def broken(**kwargs):
            raise CatalogUnavailable("offline")

        with pytest.raises(CatalogUnavailable):
            await refresh(fetch=broken, seed_file=block)
        assert block.read_text(encoding="utf-8") == "previous block"


class TestForbiddenSelection:
    """Which superseded identifiers get named is the whole value of the block.

    Sorting the legacy map by ID put all eight slots on one provider on live
    data, truncating away `openai/gpt-4o` and `google/gemini-2.5-flash` -- the
    exact identifiers a model with an older cutoff reaches for.
    """

    @pytest.fixture
    def catalog(self, raw_models):
        return build_catalog(raw_models, now=NOW)

    def test_forbidden_list_spans_providers(self, catalog):
        rendered = render_seed(catalog, max_forbidden=3)
        forbidden = [
            line.strip()
            for line in rendered.split("do not reach for:")[1].splitlines()
            if line.startswith("  ") and "/" in line
        ]
        assert len({entry.split("/")[0] for entry in forbidden}) == 3

    def test_names_the_direct_predecessor_of_a_current_model(self, catalog):
        """The highest-value entry per family is the model the current one replaced.

        Ranking by age instead surfaced `gpt-3.5-turbo-instruct` and
        `gemma-2-27b-it` on live data -- identifiers nobody reaches for -- while
        `gpt-4o` and `gemini-2.5-flash` were pushed out.
        """
        rendered = render_seed(catalog, max_forbidden=6)
        assert "openai/gpt-4o" in rendered
        assert "anthropic/claude-opus-4.8" in rendered
        assert "google/gemini-2.5-flash" in rendered

    def test_prefers_families_that_are_still_in_use(self, catalog):
        # A lineage with no current model is a dead end; spend slots elsewhere.
        assert "openai/gpt-3.5-turbo-instruct" not in render_seed(catalog, max_forbidden=6)

    def test_does_not_list_a_whole_version_run_of_one_family(self, catalog):
        """At most two per family: the direct predecessor and the training-era name.

        Listing opus-4.1 through opus-4.8 would spend the entire resident budget
        on a single lineage.
        """
        rendered = render_seed(catalog, max_forbidden=12)
        forbidden = [
            line.strip()
            for line in rendered.split("do not reach for:")[1].splitlines()
            if line.startswith("  ") and "/" in line
        ]
        assert len(forbidden) == len(set(forbidden))
        opus = [entry for entry in forbidden if "claude-opus" in entry]
        assert len(opus) <= 2

    def test_honours_the_cap(self, catalog):
        rendered = render_seed(catalog, max_forbidden=2)
        forbidden = [
            line.strip()
            for line in rendered.split("do not reach for:")[1].splitlines()
            if line.startswith("  ") and "/" in line
        ]
        assert len(forbidden) == 2

    def test_never_forbids_something_it_also_recommends(self, catalog):
        rendered = render_seed(catalog)
        _active, forbidden = rendered.split("do not reach for:")
        for model in catalog.featured():
            assert model.id not in forbidden

    def test_forbids_a_legacy_model_whose_successor_is_not_featured(self, catalog):
        """`legacy_map` only maps to featured successors, but the block renders no
        mapping -- just identifiers not to use. Restricting it to mapped entries
        dropped `openai/gpt-4o` on live data, which is the single most likely
        stale identifier for an older-cutoff model to emit.
        """
        rendered = render_seed(catalog, max_per_provider=1, max_forbidden=12)
        assert "openai/gpt-6-astra" in rendered.split("do not reach for:")[0]
        assert "openai/gpt-4o" in rendered.split("do not reach for:")[1]

    def test_includes_retired_models_too(self, catalog):
        rendered = render_seed(catalog, max_forbidden=12)
        assert "openai/gpt-5.4-mini" in rendered.split("do not reach for:")[1]

    def test_names_both_a_recent_and_a_training_era_entry_of_a_live_family(self, catalog):
        """One slot per lineage let the newest supersession shadow the famous one.

        On live data `openai/gpt-5.4` occupies the `(gpt,)` lineage and pushed out
        `gpt-4o`, but `gpt-4o` is exactly the identifier the brief asks to forbid,
        because it is what an older-cutoff model emits.
        """
        rendered = render_seed(catalog, max_forbidden=12)
        forbidden = rendered.split("do not reach for:")[1]
        assert "anthropic/claude-opus-4.8" in forbidden
        assert "anthropic/claude-opus-3" in forbidden

    def test_the_recent_entry_still_comes_first(self, catalog):
        forbidden = render_seed(catalog, max_forbidden=12).split("do not reach for:")[1]
        assert forbidden.index("claude-opus-4.8") < forbidden.index("claude-opus-3")


class TestDeclaredCapabilityLine:
    """The session baseline should say what current endpoints accept and emit.

    An agent that believes models only read and emit plain text will design
    around a limitation the registry already contradicts. The counts are derived
    from registry fields, never asserted, so the line cannot go stale.
    """

    @pytest.fixture
    def catalog(self, raw_models):
        return build_catalog(raw_models, now=NOW)

    def test_reports_how_many_accept_image_input(self, raw_models):
        models = [*raw_models, or_model("acme/vision-1", ts(2026, 9, 1), prompt_price="0.00002")]
        models[-1]["architecture"]["input_modalities"] = ["text", "image"]
        rendered = render_seed(build_catalog(models, now=NOW), providers=["acme"])
        assert "image" in rendered.lower()

    def test_reports_tool_support(self, catalog):
        assert "tool" in render_seed(catalog).lower()

    def test_makes_no_claim_the_registry_does_not_support(self, catalog):
        rendered = render_seed(catalog).lower()
        for invented in ("svg", "coordinate loop", "natively output"):
            assert invented not in rendered

    def test_tells_the_agent_the_text_only_assumption_is_checkable(self, catalog):
        assert "plain text" in render_seed(catalog).lower()

    def test_still_fits_the_resident_budget(self, catalog):
        assert len(render_seed(catalog)) < 2600

    def test_is_ascii_only(self, catalog):
        assert render_seed(catalog).isascii()
