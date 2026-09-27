"""One existing sale selection, once; no Body or projector changes."""
from one_shot_view_lens_harness_v0 import OneShotViewLensHarness
from short_plan_action_projector_v0 import project_short_plan


def propose_sale_once(name, pre, selectable, baseline_id):
    base = {
        "eligible": False,
        "canonical_observable": "existing_sellable_shed_stock",
        "observable_by_candidate": {},
        "metadata": {"tie_break": "item ascending, candidate_id ascending"},
    }
    by_id = {p.candidate_id: p for p in selectable}
    baseline = by_id[baseline_id]
    if len(selectable) < 2 or baseline.kind == "realize_shed_stock_sale":
        return base
    sales = sorted(
        (p for p in selectable if p.kind == "realize_shed_stock_sale"),
        key=lambda p: (str(p.target["item"]), p.candidate_id),
    )
    if not sales:
        return base
    replacement = sales[0]
    if project_short_plan(pre, baseline) == project_short_plan(pre, replacement):
        return base
    return {**base, "eligible": True,
            "selected_candidate_id": replacement.candidate_id}


def make_candidate():
    return OneShotViewLensHarness("sale_once", evaluator=propose_sale_once)
