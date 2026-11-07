"""Source-disjoint, time-ordered INTERNAL diagnostic on existing synthetic data.

This does NOT create an independent market test: the original generator used
global statistics and this existing dataset was already audited during development.
"""
from pathlib import Path
import hashlib
import json

import pandas as pd
import airpulse_v4 as m
from train import fit_models, tune_policy, pair_metrics, json_write

ROOT=Path(__file__).resolve().parent


def run():
    quotes=m.load_quotes(ROOT/"data/AirPulse_V2_SEMISYNTHETIC_BASE.csv")
    assigned=pd.read_csv(ROOT/"trained/source_partitions.csv")
    meta=quotes.groupby("itinerary_id").agg(first=("observed_at","min"),departure=("departure_at","first"))
    meta=meta.join(assigned.set_index("itinerary_id")[["partition","source_flight_id"]])
    t1=pd.Timestamp("2027-03-01T00:00:00+07:00")
    t2=pd.Timestamp("2027-06-01T00:00:00+07:00")
    t3=pd.Timestamp("2027-09-01T00:00:00+07:00")
    # Embargo is defined using the FIRST observation and complete departure time.
    # Thus previous-stage target labels end BEFORE the next stage begins.
    ids={"train":set(meta.index[meta.partition.eq("train")&(meta.departure<t1)]),
         "calibration":set(meta.index[meta.partition.eq("calibration")&(meta['first']>=t1)&(meta.departure<t2)]),
         "policy_validation":set(meta.index[meta.partition.eq("policy_validation")&(meta['first']>=t2)&(meta.departure<t3)]),
         "test":set(meta.index[meta.partition.eq("test")&(meta['first']>=t3)])}
    out=ROOT/"trained/forward_diagnostic";out.mkdir(parents=True,exist_ok=True)
    stages={k:{"itineraries":len(v),"sources":int(meta.loc[list(v),"source_flight_id"].nunique())} for k,v in ids.items()}
    report={"protocol":"source_disjoint_and_time_ordered_internal_diagnostic","stage_counts":stages,
            "new_market_data":False,"original_generator_global_statistics_caveat":True,
            "data_previously_used_in_development":True,"test_used_to_select_parameters":False}
    if any(len(v)<5 for v in ids.values()):
        report.update(status="INSUFFICIENT_STAGE_COVERAGE")
        json_write(out/"metrics.json",report);print(json.dumps(report,indent=2));return report
    settings=m.Settings()
    train=m.supervised_pairs(quotes,ids["train"],settings)
    cal=m.supervised_pairs(quotes,ids["calibration"],settings)
    if cal.y_drop_within.nunique()<2:
        report.update(status="INSUFFICIENT_CALIBRATION_CLASSES")
        json_write(out/"metrics.json",report);return report
    bundle,_=fit_models(train,cal,quotes,settings,20261021,200)
    policy_eps,exc=m.prepare_episodes(bundle,quotes,ids["policy_validation"])
    if not policy_eps:
        report.update(status="NO_COVERED_POLICY_VALIDATION_EPISODES")
        json_write(out/"metrics.json",report);return report
    grid=tune_policy(bundle,policy_eps,settings,20261021)
    grid.to_csv(out/"validation_policy_grid.csv",index=False)
    bundle["model_id"]="airpulse-v4-forward-diagnostic"
    # The forward diagnostic is not the main model shipped for replay/integration.
    test_pairs=m.supervised_pairs(quotes,ids["test"],settings,max_snapshots=16)
    test_eps,exc=m.prepare_episodes(bundle,quotes,ids["test"])
    rows,stats=m.sequential_backtest(bundle,test_eps)
    rows.to_csv(out/"test_decisions.csv",index=False)
    if len(rows):stats["cluster_bootstrap_saving_95pct"]=m.cluster_interval(rows)
    report.update(status="COMPLETE",forecast=pair_metrics(bundle,test_pairs),sequential=stats,
        policy=bundle["policy"],policy_selection=bundle["policy_validation"],excluded_test_episodes=len(exc))
    assigned_rows=[{"itinerary_id":iid,"partition":k,"source_flight_id":meta.loc[iid,"source_flight_id"]} for k,v in ids.items() for iid in sorted(v)]
    pd.DataFrame(assigned_rows).to_csv(out/"source_time_partitions.csv",index=False)
    for a,b in [("train","calibration"),("calibration","policy_validation"),("policy_validation","test")]:
        if ids[a] and ids[b]:
            assert meta.loc[list(ids[a]),"departure"].max()<meta.loc[list(ids[b]),"first"].min()
    json_write(out/"metrics.json",report)
    print(json.dumps({"status":report["status"],"stages":stages,"sequential":stats},indent=2))
    return report


if __name__=="__main__":run()
