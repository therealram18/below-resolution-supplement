"""Assemble boards/chronos2_noleak_report.json from boards/s7_parts and print a markdown table."""
import json, os
import audit_fev_family as af
H = os.path.dirname(os.path.abspath(__file__)); P = H + "/boards/s7_parts/"
first = json.load(open(H + "/boards/fev_family_report.json"))
RULES = [("wr", "C_wr", "win rate", 100), ("rank", "C_rank", "mean rank", 1), ("gm", "C_gm_mase", "geometric mean relative MASE", 100)]
out = dict(seed=20261111, leakage_reference_model="chronos_bolt_base", library_check=json.load(open(P + "check.json")), rows={}, replay=json.load(open(P + "replay.json")), reports={})
rows = []
for rk, fk, lab, sc in RULES:
    for var, rep in (("first round (flags ignored)", first["reports"][fk]), ("library replacement", json.load(open(P + f"report_lib_{rk}.json"))), ("tasks with flags dropped", json.load(open(P + f"report_drop_{rk}.json")))):
        s = af.summarise(rep, sc)
        s["tasks"] = rep.get("tasks") or "all 27"
        out["rows"][f"{rk}|{var}"] = s
        r = rep["resolution"]
        rows.append(f"| {lab} | {var} | {s['entries']} | {s['units']} | {s['sources']} | {', '.join(s['top10'])} | {s['resolution_paper']:.2f} / {s['resolution_corrected']:.2f} | {s['median_adjacent_gap']:.3f} | {s['ratio_paper']:.2f} / {s['ratio_corrected']:.2f} | {s['adjacent_of_9']['paper']} / {s['adjacent_of_9']['CR2-icc']} / {s['adjacent_of_9']['wild']} | {s['all_holm']['paper']} / {s['all_holm']['CR2-icc']} | {s['tier']['paper']} ({', '.join(s['tier_members']['paper'])}) / {s['tier']['CR2-icc']} |")
af.jdump(out, H + "/boards/chronos2_noleak_report.json")
print("\n".join(rows))
