"""Reproducible V4 training. Test IDs are not used to select any parameter."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import sys
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, mean_absolute_error

import airpulse_v4 as m

ROOT=Path(__file__).resolve().parent
warnings.filterwarnings("ignore",message="X does not have valid feature names")


def json_write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding="utf-8")


def source_weights(pairs):
    counts=pairs.source_flight_id.value_counts()
    w=1/pairs.source_flight_id.map(counts).to_numpy(float)
    return w/w.mean()


def fixed_splits(quotes, split_csv, seed):
    original=pd.read_csv(split_csv)
    if original.itinerary_id.duplicated().any():raise ValueError("Duplicate itinerary assignments")
    sources=quotes[["itinerary_id","source_flight_id"]].drop_duplicates()
    joined=sources.merge(original,on="itinerary_id",validate="one_to_one")
    if len(joined)!=len(sources):raise ValueError("Every itinerary must have a split assignment")
    val_sources=np.array(sorted(joined.loc[joined.split.eq("validation"),"source_flight_id"].unique()))
    rng=np.random.default_rng(seed);rng.shuffle(val_sources)
    calibration_sources=set(val_sources[:len(val_sources)//2])
    joined["partition"]=joined["split"].replace({"validation":"policy_validation"})
    joined.loc[joined.source_flight_id.isin(calibration_sources),"partition"]="calibration"
    ids={p:set(joined.loc[joined.partition.eq(p),"itinerary_id"]) for p in
         ["train","calibration","policy_validation","test"]}
    for a in ids:
        for b in ids:
            if a>=b:continue
            ga=set(joined.loc[joined.itinerary_id.isin(ids[a]),"source_flight_id"])
            gb=set(joined.loc[joined.itinerary_id.isin(ids[b]),"source_flight_id"])
            if ga&gb:raise ValueError(f"Source leakage: {a}/{b}")
    return ids,joined


def fit_models(trainpairs, calpairs, quotes, settings, seed, trees):
    prep=ColumnTransformer([("categorical",OneHotEncoder(handle_unknown="ignore",sparse_output=False),m.CATS),
                            ("numerical","passthrough",m.NUMS)],remainder="drop")
    xt=prep.fit_transform(trainpairs[m.FEATURES]);xc=prep.transform(calpairs[m.FEATURES])
    weights=source_weights(trainpairs)
    params=dict(n_estimators=trees,learning_rate=.035,num_leaves=15,max_depth=5,
                min_child_samples=90,reg_lambda=8.,n_jobs=2,verbose=-1,random_state=seed)
    clf=LGBMClassifier(**params)
    clf.fit(xt,trainpairs.y_drop_within,sample_weight=weights)
    point=LGBMRegressor(**params,objective="regression_l1")
    # Price weighting aligns log-return training more closely with VND error.
    price_weights=np.clip(trainpairs.current_price_vnd.to_numpy()/trainpairs.current_price_vnd.median(),.25,4)
    point.fit(xt,trainpairs.y_point_log_return,sample_weight=weights*price_weights)
    opportunity=LGBMRegressor(**params,objective="regression")
    opportunity.fit(xt,trainpairs.y_opportunity_discount,sample_weight=weights)
    # A separate cash-cost signal penalizes waiting through expected price rises.
    # The opportunity minimum alone cannot represent that downside.
    near=trainpairs.horizon_hours.eq(6).to_numpy()
    cost=LGBMRegressor(**params,objective="regression")
    cost.fit(xt[near],(trainpairs.target_price_vnd.to_numpy()[near]-trainpairs.current_price_vnd.to_numpy()[near])/
             trainpairs.current_price_vnd.to_numpy()[near],sample_weight=(weights*price_weights)[near])
    raw=clf.predict_proba(xc)[:,1]
    logits=np.log(np.clip(raw,1e-6,1-1e-6)/np.clip(1-raw,1e-6,1))
    if calpairs.y_drop_within.nunique()!=2:raise ValueError("Calibration partition must contain both classes")
    calibrator=LogisticRegression(C=1.,max_iter=200,random_state=seed)
    calibrator.fit(logits.reshape(-1,1),calpairs.y_drop_within,sample_weight=source_weights(calpairs))
    raw_point=point.predict(xc)
    blend_grid=[]
    for w in [0.,.25,.5,.75,1.]:
        price=calpairs.current_price_vnd.to_numpy()*np.exp(np.clip(w*raw_point,-2,2))
        blend_grid.append({"weight":w,"calibration_mae_vnd":float(mean_absolute_error(calpairs.target_price_vnd,price))})
    # Tie -> the simpler persistence baseline. No test-based blend selection.
    best_blend=min(blend_grid,key=lambda r:(r["calibration_mae_vnd"],r["weight"]))
    baseline_mae=blend_grid[0]["calibration_mae_vnd"]
    # A tiny, unstable validation change does not justify departing from persistence.
    material_point_gain=(baseline_mae-best_blend["calibration_mae_vnd"])/baseline_mae
    blend=best_blend["weight"] if material_point_gain>=.01 else 0.
    residual=np.abs(calpairs.y_point_log_return.to_numpy()-blend*raw_point)
    radius=float(np.quantile(residual,.9,method="higher"))
    per_h={}
    for h in m.HORIZONS:
        values=residual[calpairs.horizon_hours.eq(h)]
        per_h[str(h)]=float(np.quantile(values,.9,method="higher")) if len(values)>=30 else radius
    scope=quotes.loc[quotes.itinerary_id.isin(set(trainpairs.itinerary_id)),m.CATS].drop_duplicates().values.tolist()
    origins=quotes.loc[quotes.itinerary_id.isin(set(trainpairs.itinerary_id)),"data_origin"].dropna().unique().tolist() if "data_origin" in quotes else []
    training_kind=str(origins[0]) if len(origins)==1 else ("MIXED_INPUT_PROVENANCE" if origins else "UNVERIFIED_INPUT_PROVENANCE")
    bundle={"schema_version":m.SCHEMA_VERSION,"model_version":m.VERSION,"model_id":"pending",
            "settings":asdict(settings),"preprocess":prep,"point_model":point,
            "opportunity_classifier":clf,"opportunity_model":opportunity,"calibrator":calibrator,
            "short_term_cost_model":cost,
            "point_blend_weight":blend,"point_validation_relative_gain":material_point_gain,
            "point_min_required_validation_gain":.01,"interval_abs_log_residual":per_h,"global_interval_radius":radius,
            "supported_route_airlines":scope,"validated_for_live":False,
            "training_data_kind":training_kind,
            "policy":{"enabled":False},"policy_validation":{"status":"pending"},
            "features":m.FEATURES,"calendar_seasonality_features_used":False}
    return bundle,blend_grid


def pair_metrics(bundle,pairs):
    point,prob,potential,_=m.predict_pairs(bundle,pairs)
    now=pairs.current_price_vnd.to_numpy();actual=pairs.target_price_vnd.to_numpy()
    pred=now*np.exp(point); y=pairs.y_drop_within.to_numpy()
    rows=[]
    for h,g in pairs.assign(prediction=pred,probability=prob).groupby("horizon_hours"):
        rows.append({"hours":int(h),"n_pairs":len(g),"model_mae_vnd":float(mean_absolute_error(g.target_price_vnd,g.prediction)),
                     "persistence_mae_vnd":float(mean_absolute_error(g.target_price_vnd,g.current_price_vnd)),
                     "observed_drop_within_rate":float(g.y_drop_within.mean())})
    bins=[]
    for interval,g in pairs.assign(probability=prob).groupby(pd.cut(prob,np.linspace(0,1,11),include_lowest=True),observed=True):
        bins.append({"bin":str(interval),"n_pairs":len(g),"predicted_probability":float(g.probability.mean()),
                     "observed_event_rate":float(g.y_drop_within.mean())})
    intervals=np.array([bundle["interval_abs_log_residual"].get(str(int(h)),bundle["global_interval_radius"]) for h in pairs.horizon_hours])
    coverage=float(((actual>=pred*np.exp(-intervals))&(actual<=pred*np.exp(intervals))).mean())
    return {"pairs":len(pairs),"itineraries":int(pairs.itinerary_id.nunique()),"source_groups":int(pairs.source_flight_id.nunique()),
            "price_mae_vnd":float(mean_absolute_error(actual,pred)),"persistence_mae_vnd":float(mean_absolute_error(actual,now)),
            "auc_drop_within":float(roc_auc_score(y,prob)) if len(set(y))==2 else None,
            "average_precision":float(average_precision_score(y,prob)),"brier":float(brier_score_loss(y,prob)),
            "event_prevalence":float(y.mean()),"empirical_interval_90pct_coverage":coverage,
            "by_horizon":rows,"calibration_bins":bins}


def tune_policy(bundle,episodes,settings,seed):
    baseline={"enabled":False}
    best=baseline;best_score=0.;rows=[];shadow=baseline;shadow_mean=0.
    for window in [12,24,48,72]:
        for proba in [.20,.35,.50,.65,.80]:
            for saving,short_cost in [(s,c) for s in [50_000,100_000] for c in [None,0,5000,15000]]:
                policy={"enabled":True,"window_hours":window,"min_probability":proba,"min_saving_vnd":saving,
                        "min_expected_6h_saving_vnd":short_cost}
                decisions,stats=m.sequential_backtest(bundle,episodes,policy=policy)
                if not len(decisions):continue
                waits=decisions[decisions.initial_action.eq("WAIT_AND_RECHECK")]
                n_wait_sources=waits.source_flight_id.nunique()
                ci=m.cluster_interval(decisions,seed=seed,samples=400,alpha=.20)
                score=float(ci[0])-0.5*stats.get("mean_downside_vnd",0) if ci[0] is not None else -float("inf")
                eligible=n_wait_sources>=3 and stats.get("unexecuted",1)==0
                rows.append({**policy,"validation_score_vnd":score,"eligible":bool(eligible),
                             "independent_wait_sources":int(n_wait_sources),"lower_cluster_80pct_vnd":ci[0],**stats})
                if eligible and stats["mean_saving_vnd"]>shadow_mean:
                    shadow_mean=stats["mean_saving_vnd"];shadow=policy
                if eligible and score>best_score+1e-8:
                    best=policy;best_score=score
    bundle["policy"]=best
    bundle["shadow_policy"]=shadow
    bundle["policy_validation"]={"status":"WAIT_ENABLED_BY_VALIDATION" if best.get("enabled") else "WAIT_DISABLED_NO_ROBUST_VALIDATION_EDGE",
        "selection_partition":"policy_validation_only","selection_score":"cluster_bootstrap_80pct_lower_bound - 0.5 * mean_downside",
        "selected_score_vnd":best_score,"requires_at_least_wait_source_groups":3,
        "test_used_for_selection":False,"shadow_policy_is_not_enabled_automatically":True}
    return pd.DataFrame(rows)


def evaluate_partition(bundle,quotes,ids,out,name,v3=None):
    pairs=m.supervised_pairs(quotes,ids,m.Settings(**bundle["settings"]),max_snapshots=16)
    episodes,excluded=m.prepare_episodes(bundle,quotes,ids)
    rows,policy=m.sequential_backtest(bundle,episodes)
    buy,buy_stats=m.sequential_backtest(bundle,episodes,strategy="buy_now")
    rows.to_csv(out/f"{name}_sequential_decisions.csv",index=False)
    pd.DataFrame(excluded).to_csv(out/f"{name}_excluded_episodes.csv",index=False)
    if len(rows):policy["cluster_bootstrap_saving_95pct"]=m.cluster_interval(rows)
    result={"forecast":pair_metrics(bundle,pairs),"sequential":policy,"buy_now":buy_stats,
            "coverage":{"requested_itineraries":len(ids),"eligible_episodes":len(episodes),
                        "eligible_itineraries":len({ep["itinerary_id"] for ep in episodes}),"excluded_entries":len(excluded)}}
    if v3:
        ref_rows,ref_stats=m.sequential_backtest(bundle,episodes,strategy="v3",v3_bundle=v3[0],v3_module=v3[1])
        ref_rows.to_csv(out/f"{name}_v3_sequential_decisions.csv",index=False)
        result["v3_same_episodes_sequential"]=ref_stats
        # Legacy diagnostic; its original sampling/target matching differs from V4.
        old_pairs=v3[1].build_training_pairs(quotes,ids,max_snapshots=16)
        xo=v3[0]["preprocess"].transform(old_pairs[v3[1].CATS+v3[1].NUMS])
        old_pred=old_pairs.current_price_vnd.to_numpy()*(1-np.clip(v3[0]["discount_model"].predict(xo),-1.5,.9))
        result["v3_original_pair_diagnostic"]={"pairs":len(old_pairs),"price_mae_vnd":float(mean_absolute_error(old_pairs.actual_price_vnd,old_pred)),
                                               "not_identical_targets_to_v4":True}
        common_features=[]
        for iid,pp in pairs.groupby("itinerary_id",sort=False):
            g=quotes[quotes.itinerary_id.eq(iid)].sort_values("observed_at").reset_index(drop=True)
            positions={t:i for i,t in enumerate(g.observed_at)}
            cached={}
            for r in pp.itertuples():
                if r.as_of not in cached:
                    cached[r.as_of]=v3[1]._state(g,positions[r.as_of])[0]
                f=v3[1]._h_feature(cached[r.as_of],r.as_of,g.departure_at.iloc[0],r.horizon_hours)
                common_features.append({**f,"actual":r.target_price_vnd,"current":r.current_price_vnd})
        cf=pd.DataFrame(common_features)
        if not np.array_equal(cf.actual.to_numpy(),pairs.target_price_vnd.to_numpy()):
            raise ValueError("Common benchmark row alignment differs")
        cx=v3[0]["preprocess"].transform(cf[v3[1].CATS+v3[1].NUMS])
        cp=cf.current.to_numpy()*(1-np.clip(v3[0]["discount_model"].predict(cx),-1.5,.9))
        result["same_target_price_comparison"]={"pairs":len(cf),
            "v3_price_mae_vnd":float(mean_absolute_error(cf.actual,cp)),
            "v4_price_mae_vnd":float(mean_absolute_error(cf.actual,cf.current.to_numpy()*np.exp(m.predict_pairs(bundle,pairs)[0]))),
            "persistence_mae_vnd":float(mean_absolute_error(cf.actual,cf.current))}
    return result


def run(data,outdir,split_csv,trees=200,seed=20261021):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    quotes=m.load_quotes(data);settings=m.Settings()
    if "source_flight_id" not in pd.read_csv(data,nrows=1):raise ValueError("Explicit source_flight_id required")
    ids,split=fixed_splits(quotes,split_csv,seed)
    split.to_csv(out/"source_partitions.csv",index=False)
    print("BUILDING_TRAIN_AND_CALIBRATION",flush=True)
    trainpairs=m.supervised_pairs(quotes,ids["train"],settings)
    calpairs=m.supervised_pairs(quotes,ids["calibration"],settings)
    print("FITTING_MODELS",len(trainpairs),len(calpairs),flush=True)
    bundle,blend=fit_models(trainpairs,calpairs,quotes,settings,seed,trees)
    print("CACHING_POLICY_VALIDATION",flush=True)
    episodes,excluded=m.prepare_episodes(bundle,quotes,ids["policy_validation"])
    grid=tune_policy(bundle,episodes,settings,seed)
    grid.to_csv(out/"validation_policy_grid.csv",index=False)
    pd.DataFrame(blend).to_csv(out/"calibration_point_blend.csv",index=False)
    # Seal model/policy BEFORE evaluating any test labels.
    code_hashes={n:m.sha256(ROOT/n) for n in ["airpulse_v4.py","train.py"]}
    fingerprint={"data":m.sha256(data),"policy":bundle["policy"],"seed":seed,"trees":trees,
                 "settings":asdict(settings),"point_blend":bundle["point_blend_weight"],"code":code_hashes}
    digest=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()[:16]
    bundle["model_id"]="airpulse-v4-"+digest
    bundle["split_seed"]=seed
    m.save_bundle(bundle,out/"airpulse_v4_model.joblib")
    model_hash=m.sha256(out/"airpulse_v4_model.joblib")
    v3=None
    if (ROOT/"legacy/airpulse_v3_model.joblib").exists():
        sys.path.insert(0,str(ROOT/"legacy"))
        import airpulse_v3
        v3=(airpulse_v3.joblib.load(ROOT/"legacy/airpulse_v3_model.joblib"),airpulse_v3)
    reference_data=ROOT/"data/AirPulse_V2_SEMISYNTHETIC_BASE.csv"
    reused=reference_data.is_file() and m.sha256(data)==m.sha256(reference_data)
    metrics={"protocol":"source_grouped_development_benchmark",
             "test_was_previously_audited_in_v3":True if reused else None,
             "benchmark_reused_during_development":True if reused else None,
             "new_real_world_validation":False,
             "policy":bundle["policy"],"policy_selection":bundle["policy_validation"],
             "calibration":pair_metrics(bundle,calpairs),
             "dataset":{"rows":len(quotes),"itineraries":int(quotes.itinerary_id.nunique()),
                        "source_groups":int(quotes.source_flight_id.nunique()),
                        "partitions":{k:{"itineraries":len(v),"sources":int(quotes.loc[quotes.itinerary_id.isin(v),"source_flight_id"].nunique())} for k,v in ids.items()}}}
    for name in ["policy_validation","test"]:
        print("EVALUATING",name,flush=True)
        metrics[name]=evaluate_partition(bundle,quotes,ids[name],out,name,v3=v3 if name=="test" else None)
    stress_path=Path(data).with_name("AirPulse_V2_SEMISYNTHETIC_STRESS.csv")
    if stress_path.exists():
        print("STRESS_DIAGNOSTIC",flush=True)
        stress=m.load_quotes(stress_path)
        eps,exc=m.prepare_episodes(bundle,stress,ids["test"])
        srows,sm=m.sequential_backtest(bundle,eps)
        srows.to_csv(out/"stress_sequential_decisions.csv",index=False)
        metrics["stress_assumption_only"]={"sequential":sm,"excluded":len(exc),"independent_market_test":False}
    json_write(out/"metrics.json",metrics)
    versions={n:importlib.metadata.version(n) for n in ["numpy","pandas","scikit-learn","lightgbm","joblib"]}
    manifest={"model_version":m.VERSION,"model_id":bundle["model_id"],"schema_version":m.SCHEMA_VERSION,
              "training_data_sha256":m.sha256(data),"source_split_sha256":m.sha256(split_csv),
              "model_sha256":model_hash,"python":platform.python_version(),"dependencies":versions,
              "settings":asdict(settings),"trees":trees,"seed":seed,"features":m.FEATURES,"code_sha256":code_hashes,
              "live_validated":False,"model_and_policy_sealed_before_test":True,
              "test_reused_from_prior_review":True if reused else None,
              "benchmark_reused_during_development":True if reused else None,
              "new_independent_market_data_available":False,
              "generator_code_available":False,"seasonality_claims_supported":False}
    json_write(out/"manifest.json",manifest)
    print(json.dumps({"policy":bundle["policy"],"test":metrics["test"]["sequential"],
                      "point_mae":metrics["test"]["forecast"]["price_mae_vnd"],
                      "persistence_mae":metrics["test"]["forecast"]["persistence_mae_vnd"]},indent=2),flush=True)
    return bundle,metrics


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=ROOT/"data/AirPulse_V2_SEMISYNTHETIC_BASE.csv")
    p.add_argument("--out",type=Path,default=ROOT/"trained")
    p.add_argument("--source-split",type=Path,default=ROOT/"data/v3_source_split.csv")
    p.add_argument("--trees",type=int,default=200)
    a=p.parse_args();run(a.data,a.out,a.source_split,trees=a.trees)
