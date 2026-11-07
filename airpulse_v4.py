"""AirPulse V4: calibrated price-drop opportunities and a rechecking policy.

Research software. Forecasts are estimates, never payable fare quotes.
All training features use observations <= as_of. Future minima are LABELS only.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import re
import warnings

import joblib
import numpy as np
import pandas as pd

VERSION = "4.0.0"
SCHEMA_VERSION = "airpulse.insight.v4"
TZ = "Asia/Ho_Chi_Minh"
HORIZONS = (6, 12, 24, 48, 72)
CATS = ["origin", "destination", "airline"]
NUMS = ["log_price", "days_to_departure", "hours_to_deadline", "history_hours",
        "quote_age_hours", "quote_count_24h", "quote_count_72h", "changes_24h",
        "changes_72h", "hours_since_last_change", "last_return", "return_24h",
        "return_72h", "missing_lag_24h", "missing_lag_72h", "rel_min_72h",
        "rel_median_72h", "rel_max_72h", "departure_hour_sin", "departure_hour_cos",
        "snapshot_hour_sin", "snapshot_hour_cos", "horizon_hours"]
FEATURES = CATS + NUMS
REQUIRED = ["itinerary_id", "observed_at", "departure_at", "origin", "destination", "airline", "price_vnd"]


@dataclass(frozen=True)
class Settings:
    min_quotes: int = 4
    min_history_hours: float = 12.0
    max_quote_age_hours: float = 6.0
    min_lead_hours: float = 36.0
    recheck_hours: float = 3.0
    execution_tolerance_hours: float = 3.1  # 6-minute scheduling grace; never use an earlier quote
    max_label_gap_hours: float = 6.1
    drop_fraction: float = .04
    min_drop_vnd: int = 50_000


class InputError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def timestamp(value, name="timestamp"):
    try:
        t = pd.Timestamp(value)
        if pd.isna(t) or t.tz is None:
            raise ValueError()
        return t.tz_convert(TZ)
    except (ValueError, TypeError, OverflowError) as exc:
        raise InputError("INVALID_TIMESTAMP", f"{name} must be a valid timestamp with timezone offset") from exc


def load_quotes(value):
    d = pd.read_csv(value) if not isinstance(value, pd.DataFrame) else value.copy()
    missing = set(REQUIRED) - set(d.columns)
    if missing:
        raise InputError("MISSING_FIELDS", f"Missing fields: {sorted(missing)}")
    if not len(d):
        raise InputError("EMPTY_HISTORY", "History is empty")
    for col in ("observed_at", "departure_at"):
        if not d[col].astype(str).str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True).all():
            raise InputError("INVALID_TIMESTAMP", f"{col} requires a timezone offset")
        try:
            d[col] = pd.to_datetime(d[col], utc=True, format="ISO8601").dt.tz_convert(TZ)
        except (ValueError, TypeError, OverflowError) as exc:
            raise InputError("INVALID_TIMESTAMP", f"Invalid {col}") from exc
    p = pd.to_numeric(d.price_vnd, errors="coerce")
    if not (np.isfinite(p) & (p > 0)).all():
        raise InputError("INVALID_PRICE", "Prices must be finite, positive VND totals")
    d["price_vnd"] = p.astype(float)
    if not (d.observed_at < d.departure_at).all():
        raise InputError("QUOTE_AFTER_DEPARTURE", "Quotes must precede departure")
    for col in CATS:
        if d[col].isna().any() or d[col].astype(str).str.strip().eq("").any():
            raise InputError("INVALID_CATEGORY", f"Empty {col}")
        d[col] = d[col].astype(str).str.upper().str.strip()
    for col in ("origin", "destination"):
        if not d[col].str.fullmatch("[A-Z]{3}").all():
            raise InputError("INVALID_AIRPORT", "Airport codes must have three letters")
    if d.origin.eq(d.destination).any():
        raise InputError("INVALID_ROUTE", "Origin and destination must differ")
    if d.itinerary_id.isna().any() or d.itinerary_id.astype(str).str.strip().eq("").any():
        raise InputError("INVALID_ITINERARY", "itinerary_id must not be empty")
    d["itinerary_id"] = d.itinerary_id.astype(str)
    if "source_flight_id" not in d:
        d["source_flight_id"] = d.itinerary_id
        d.attrs["missing_source_ids"] = True
    if d.source_flight_id.isna().any() or d.source_flight_id.astype(str).str.strip().eq("").any():
        raise InputError("INVALID_SOURCE_ID", "source_flight_id must be present and nonempty when supplied")
    for col in ("fare_family", "seller", "cabin", "baggage"):
        if col not in d:
            d[col] = "UNSPECIFIED"
        d[col] = d[col].fillna("UNSPECIFIED").astype(str)
    if "currency" not in d:
        d["currency"] = "VND"
    if not d.currency.eq("VND").all():
        raise InputError("UNSUPPORTED_CURRENCY", "Convert and verify currencies upstream; this model accepts VND only")
    for col in ["departure_at", *CATS, "source_flight_id", "fare_family", "seller", "cabin", "baggage", "currency"]:
        if (d.groupby("itinerary_id")[col].nunique(dropna=False) > 1).any():
            raise InputError("MIXED_FARE_HISTORY", f"Inconsistent {col} within itinerary")
    if (d.groupby(["itinerary_id", "observed_at"]).price_vnd.nunique() > 1).any():
        raise InputError("CONFLICTING_QUOTES", "Different prices at the same itinerary timestamp")
    return d.sort_values(["itinerary_id", "observed_at"]).drop_duplicates(["itinerary_id", "observed_at"]).reset_index(drop=True)


def context(g):
    g = g.sort_values("observed_at").reset_index(drop=True)
    first = g.iloc[0]
    return {"g": g, "times": g.observed_at.astype("int64").to_numpy() / 3.6e12,
            "prices": g.price_vnd.to_numpy(float), "departure": first.departure_at,
            "origin": first.origin, "destination": first.destination, "airline": first.airline}


def state_features(c, i, now, settings):
    """The full context may contain labels; only arrays ending at i are read here."""
    ts, prices = c["times"][:i+1], c["prices"][:i+1]
    nh = now.value / 3.6e12
    cur = prices[-1]
    a24 = int(np.searchsorted(ts, nh - 24))
    a72 = int(np.searchsorted(ts, nh - 72))
    p24, p72 = prices[a24:], prices[a72:]
    if not len(p72):
        raise InputError("STALE_QUOTES", "No quote in the last 72 hours")
    changes = np.flatnonzero(np.diff(prices) != 0)
    last_change = ts[changes[-1]+1] if len(changes) else ts[0]
    def lag(h):
        k = int(np.searchsorted(ts, nh-h, side="right")-1)
        if k < 0 or nh-h-ts[k] > settings.max_label_gap_hours:
            return 0.0, 1
        return float(np.log(cur/prices[k])), 0
    r24, m24 = lag(24); r72, m72 = lag(72)
    dep = c["departure"]
    dh = dep.hour + dep.minute/60
    sh = now.hour + now.minute/60
    return {"origin": c["origin"], "destination": c["destination"], "airline": c["airline"],
            "log_price": float(np.log(cur)), "days_to_departure": (dep-now).total_seconds()/86400,
            "hours_to_deadline": (dep-now).total_seconds()/3600-settings.min_lead_hours,
            "history_hours": min(nh-ts[0], 240), "quote_age_hours": nh-ts[-1],
            "quote_count_24h": len(p24), "quote_count_72h": len(p72),
            "changes_24h": int(np.count_nonzero(np.diff(p24))),
            "changes_72h": int(np.count_nonzero(np.diff(p72))),
            "hours_since_last_change": min(nh-last_change, 240),
            "last_return": float(np.log(cur/prices[-2])) if len(prices)>1 else 0,
            "return_24h": r24, "return_72h": r72, "missing_lag_24h": m24, "missing_lag_72h": m72,
            "rel_min_72h": float(np.log(cur/np.min(p72))),
            "rel_median_72h": float(np.log(cur/np.median(p72))),
            "rel_max_72h": float(np.log(cur/np.max(p72))),
            "departure_hour_sin": math.sin(dh*math.pi/12), "departure_hour_cos": math.cos(dh*math.pi/12),
            "snapshot_hour_sin": math.sin(sh*math.pi/12), "snapshot_hour_cos": math.cos(sh*math.pi/12)}


def supervised_pairs(quotes, ids, settings, max_snapshots=24):
    rows = []
    for iid, g in quotes[quotes.itinerary_id.isin(ids)].groupby("itinerary_id", sort=False):
        c = context(g); ts, p = c["times"], c["prices"]
        indices = np.arange(settings.min_quotes-1, len(g)-1)
        if len(indices) > max_snapshots:
            indices = np.unique(np.linspace(indices[0], indices[-1], max_snapshots).astype(int))
        for i in indices:
            if ts[i]-ts[0] < settings.min_history_hours:
                continue
            now = c["g"].observed_at.iloc[i]
            remaining = (c["departure"]-now).total_seconds()/3600-settings.min_lead_hours
            f = state_features(c, int(i), now, settings)
            for h in HORIZONS:
                end = ts[i]+h
                if h > remaining or ts[-1] < end:
                    continue  # censored future windows are never negative examples
                j = int(np.searchsorted(ts, end, side="right")-1)
                if j <= i or end-ts[j] > settings.max_label_gap_hours:
                    continue
                segment_times = np.r_[ts[i:j+1], end]
                if np.diff(segment_times).max() > settings.max_label_gap_hours:
                    continue
                minimum = float(p[i+1:j+1].min())
                threshold = max(settings.drop_fraction*p[i], settings.min_drop_vnd)
                rows.append({**f, "horizon_hours": h, "itinerary_id": iid,
                             "source_flight_id": str(c["g"].source_flight_id.iloc[0]), "as_of": now,
                             "current_price_vnd": p[i], "target_price_vnd": p[j],
                             "y_point_log_return": float(np.log(p[j]/p[i])),
                             "y_opportunity_discount": max(0., (p[i]-minimum)/p[i]),
                             "y_drop_within": int(minimum <= p[i]-threshold)})
    if not rows:
        raise ValueError("No complete training windows with the configured history/cadence/deadline")
    return pd.DataFrame(rows)


def predict_pairs(bundle, pairs):
    x = bundle["preprocess"].transform(pairs[FEATURES])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        raw = bundle["opportunity_classifier"].predict_proba(x)[:, 1]
        point = bundle["point_model"].predict(x) if bundle["point_blend_weight"] else np.zeros(len(pairs))
        potential = np.clip(bundle["opportunity_model"].predict(x), 0, .75)
    logits = np.log(np.clip(raw, 1e-6, 1-1e-6)/np.clip(1-raw, 1e-6, 1))
    proba = bundle["calibrator"].predict_proba(logits.reshape(-1,1))[:,1]
    point = np.clip(point * bundle["point_blend_weight"], -2, 2)
    short=np.zeros(len(pairs))
    mask=pairs.horizon_hours.eq(6).to_numpy()
    if mask.any():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="X does not have valid feature names")
            short[mask]=bundle["short_term_cost_model"].predict(x[mask])
    return point, np.clip(proba, 0, 1), potential, short


def make_curve(bundle, frame, current, now):
    point, proba, potential, short = predict_pairs(bundle, frame)
    result=[]
    for i,h in enumerate(frame.horizon_hours.astype(int)):
        radius = float(bundle["interval_abs_log_residual"].get(str(h), bundle["global_interval_radius"]))
        median = current*math.exp(float(point[i]))
        result.append({"horizon_hours": h, "window_end": (now+pd.Timedelta(hours=h)).isoformat(),
                       "forecast_price_vnd": round(median),
                       "empirical_90pct_lower_vnd": round(median*math.exp(-radius)),
                       "empirical_90pct_upper_vnd": round(median*math.exp(radius)),
                       "prob_drop_within_window": round(float(proba[i]),6),
                       "estimated_opportunity_saving_vnd": round(float(current*potential[i])),
                       "estimated_opportunity_low_vnd": round(float(current*(1-potential[i]))),
                       "expected_6h_saving_vnd": round(float(-current*short[i])) if h==6 else None})
    return result


def decide(curve, current, now, deadline, policy, settings):
    # The deadline decision is made BEFORE waiting, never by retroactively buying.
    if (deadline-now).total_seconds()/3600 <= settings.recheck_hours+settings.execution_tolerance_hours:
        return "BUY_NOW", "PURCHASE_DEADLINE_NEAR", None
    if policy.get("enabled", False):
        required=policy.get("min_expected_6h_saving_vnd")
        if required is not None:
            near=next((s for s in curve if s["horizon_hours"]==6),None)
            if near is None or near.get("expected_6h_saving_vnd",-float("inf"))<required:
                return "BUY_NOW", "NO_NEAR_TERM_EXPECTED_SAVING", None
        wanted = int(policy["window_hours"])
        slots = [s for s in curve if s["horizon_hours"] <= wanted
                 and s["prob_drop_within_window"] >= policy["min_probability"]
                 and s["estimated_opportunity_saving_vnd"] >= policy["min_saving_vnd"]]
        if slots:
            # Filter BEFORE ranking. A low-confidence slot cannot mask a valid slot.
            slot=max(slots,key=lambda s:(s["estimated_opportunity_saving_vnd"],-s["horizon_hours"]))
            return "WAIT_AND_RECHECK", "QUALIFIED_DROP_OPPORTUNITY", slot
    return "BUY_NOW", "NO_VALIDATED_WAIT_ADVANTAGE", None


def recommend(bundle, history, *, mode="replay", as_of=None, clock=None):
    settings=Settings(**bundle["settings"])
    if mode not in {"replay", "live"}:
        raise InputError("INVALID_MODE", "mode must be replay or live")
    if mode=="replay" and as_of is None:
        raise InputError("AS_OF_REQUIRED", "Replay requires an explicit as_of")
    server_now=timestamp(clock or pd.Timestamp.now(tz=TZ),"clock")
    now=timestamp(as_of,"as_of") if mode=="replay" else server_now
    if mode=="live" and as_of is not None:
        raise InputError("SERVER_CLOCK_REQUIRED", "Live uses server time; omit as_of")
    d=load_quotes(history)
    if d.itinerary_id.nunique()!=1:
        raise InputError("MULTIPLE_ITINERARIES", "One request must contain one comparable itinerary")
    if (d.observed_at>now).any():
        raise InputError("FUTURE_QUOTES", "History contains observations after as_of/server time")
    c=context(d); dep=c["departure"]; last=c["g"].observed_at.iloc[-1]
    current=float(c["prices"][-1]); deadline=dep-pd.Timedelta(hours=settings.min_lead_hours)
    info={"itinerary_id": str(d.itinerary_id.iloc[0]), "as_of":now.isoformat(),
          "departure_at":dep.isoformat(), "latest_quote_at":last.isoformat(),
          "quote_age_hours":round((now-last).total_seconds()/3600,4),
          "current_observed_price_vnd":round(current), "purchase_deadline":deadline.isoformat(),
          "quotes_seen":len(d), "history_hours":round((last-d.observed_at.iloc[0]).total_seconds()/3600,4)}
    base={"schema_version":SCHEMA_VERSION, "model_version":VERSION, "mode":mode,
          "model_id":bundle["model_id"], "training_data_kind":bundle["training_data_kind"],
          "status":"OK", "recommendation":{**info,"action":None,"recommended_buy_at":None,
          "next_check_at":None,"reason_code":None}, "forecast_slots":[],
          "probability_event":{"within_window":True,"relative_drop_at_least":settings.drop_fraction,
                               "absolute_drop_at_least_vnd":settings.min_drop_vnd},
          "warnings":["SEMI_SYNTHETIC_RESEARCH_MODEL" if "SYNTHETIC" in bundle["training_data_kind"] else "RESEARCH_MODEL_REQUIRES_MARKET_VALIDATION", "FARE_TOTAL_NOT_VERIFIED",
                      "FORECASTS_ARE_NOT_PAYABLE_FARES", "INTERVALS_HAVE_NO_MARKET_COVERAGE_GUARANTEE"]}
    def abstain(code):
        base["status"]=code; base["recommendation"]["reason_code"]=code
        return base
    if dep<=now: return abstain("DEPARTURE_PASSED")
    if deadline<=now: return abstain("OUTSIDE_SUPPORTED_PURCHASE_WINDOW")
    if len(d)<settings.min_quotes or info["history_hours"]<settings.min_history_hours:
        return abstain("INSUFFICIENT_HISTORY")
    if info["quote_age_hours"]>settings.max_quote_age_hours: return abstain("STALE_QUOTES")
    scope=(c["origin"],c["destination"],c["airline"])
    if list(scope) not in bundle["supported_route_airlines"]: return abstain("OUT_OF_SCOPE")
    if mode=="live" and not bundle.get("validated_for_live",False):
        return abstain("NOT_VALIDATED_FOR_LIVE")
    f=state_features(c,len(d)-1,now,settings)
    hours_left=(deadline-now).total_seconds()/3600
    hs=[h for h in HORIZONS if h<=hours_left]
    curve=make_curve(bundle,pd.DataFrame([{**f,"horizon_hours":h} for h in hs]),current,now) if hs else []
    action,reason,slot=decide(curve,current,now,deadline,bundle["policy"],settings)
    base["forecast_slots"]=curve
    base["valid_until"]=(now+pd.Timedelta(hours=settings.recheck_hours)).isoformat()
    base["recommendation"].update(action=action,reason_code=reason,
        recommended_buy_at=now.isoformat() if action=="BUY_NOW" else None,
        next_check_at=(now+pd.Timedelta(hours=settings.recheck_hours)).isoformat() if action=="WAIT_AND_RECHECK" else None,
        opportunity_window=slot)
    base["policy_validation"]=bundle["policy_validation"]
    return base


def prepare_episodes(bundle, quotes, ids):
    """Build forecast caches once; no test outcomes are used in deciding actions."""
    settings=Settings(**bundle["settings"])
    episodes=[]; excluded=[]; frames=[]; references=[]
    for iid,g in quotes[quotes.itinerary_id.isin(ids)].groupby("itinerary_id",sort=False):
        c=context(g);ts,p=c["times"],c["prices"]
        deadline=c["departure"]-pd.Timedelta(hours=settings.min_lead_hours)
        dh=deadline.value/3.6e12
        first=next((i for i in range(settings.min_quotes-1,len(g)) if ts[i]-ts[0]>=settings.min_history_hours),None)
        if first is None:
            excluded.append({"itinerary_id":iid,"reason":"NO_START_WINDOW"});continue
        # Fixed entry-time grid; selected from timestamps, never prices or outcomes.
        # These are correlated scenarios on each itinerary, not independent flights.
        starts={first}
        dep_hours=c["departure"].value/3.6e12
        for days_before_departure in (2,4,6,8):
            target=dep_hours-days_before_departure*24
            idx=int(np.searchsorted(ts,target,side="left"))
            if first<=idx<len(g) and ts[idx]-target<=settings.max_label_gap_hours:
                starts.add(idx)
        end=int(np.searchsorted(ts,dh,side="right")-1)
        cache={};valid_starts=[];eindex=len(episodes)
        for start in sorted(starts):
            reason=None
            if ts[start]+settings.recheck_hours+settings.execution_tolerance_hours>=dh:
                reason="NO_START_WINDOW"
            elif ts[-1]<dh:
                reason="RIGHT_CENSORED_BEFORE_DEADLINE"
            elif np.diff(np.r_[ts[start:end+1],dh]).max()>settings.max_label_gap_hours:
                reason="FUTURE_COVERAGE_GAP"
            if reason:
                excluded.append({"itinerary_id":iid,"entry_index":start,"reason":reason});continue
            valid_starts.append(start)
            episodes.append({"itinerary_id":iid,"source_flight_id":str(c["g"].source_flight_id.iloc[0]),
                "context":c,"deadline":deadline,"start":start,"end":end,"curves":cache})
        if not valid_starts:continue
        for i in range(min(valid_starts),end+1):
            now=c["g"].observed_at.iloc[i]
            f=state_features(c,i,now,settings)
            for h in HORIZONS:
                if ts[i]+h<=dh:
                    frames.append({**f,"horizon_hours":h})
                    references.append((eindex,i,h))
    if frames:
        frame=pd.DataFrame(frames);point,proba,potential,short=predict_pairs(bundle,frame)
        for k,(ei,i,h) in enumerate(references):
            ep=episodes[ei];current=ep["context"]["prices"][i]
            ep["curves"].setdefault(i,[]).append({"horizon_hours":h,
                "prob_drop_within_window":float(proba[k]),
                "estimated_opportunity_saving_vnd":float(current*potential[k]),
                "forecast_price_vnd":float(current*np.exp(point[k])),
                "expected_6h_saving_vnd":float(-current*short[k]) if h==6 else None})
    return episodes,excluded


def sequential_backtest(bundle, episodes, policy=None, strategy="v4", v3_bundle=None, v3_module=None):
    settings=Settings(**bundle["settings"]);policy=policy or bundle["policy"]
    rows=[]
    for ep in episodes:
        c=ep["context"];ts,p=c["times"],c["prices"];i=ep["start"];waits=0
        unknown=None;dh=ep["deadline"].value/3.6e12;reason=""
        while True:
            now=c["g"].observed_at.iloc[i]
            if strategy=="buy_now":
                action,reason="BUY_NOW","BASELINE"
            elif dh-ts[i]<=settings.recheck_hours+settings.execution_tolerance_hours:
                action,reason="BUY_NOW","PURCHASE_DEADLINE_NEAR"
            elif strategy=="v3":
                try:
                    rec,_=v3_module.recommend(v3_bundle,c["g"].iloc[:i+1])
                    action,reason=rec["action"],"V3_RECHECKED_3H"
                except ValueError:
                    action,reason="BUY_NOW","V3_NO_FORECAST"
            else:
                action,reason,_=decide(ep["curves"].get(i,[]),p[i],now,ep["deadline"],policy,settings)
            if action=="BUY_NOW":break
            waits+=1
            target=ts[i]+settings.recheck_hours
            nxt=int(np.searchsorted(ts,target,side="left"))
            if nxt>ep["end"] or ts[nxt]>dh or ts[nxt]-target>settings.execution_tolerance_hours:
                unknown="NO_EXECUTABLE_QUOTE_AFTER_RECHECK";break
            i=nxt
        current=p[ep["start"]]
        oracle=float(np.min(p[ep["start"]:ep["end"]+1]))
        cost=float(p[i]) if unknown is None else None
        rows.append({"itinerary_id":ep["itinerary_id"],"source_flight_id":ep["source_flight_id"],
                     "entry_id":ep["itinerary_id"]+"@"+c["g"].observed_at.iloc[ep["start"]].isoformat(),
                     "entry_days_to_departure":round((c["departure"]-c["g"].observed_at.iloc[ep["start"]]).total_seconds()/86400,4),
                     "as_of":c["g"].observed_at.iloc[ep["start"]].isoformat(),
                     "purchase_deadline":ep["deadline"].isoformat(),
                     "initial_action":"WAIT_AND_RECHECK" if waits else "BUY_NOW",
                     "rechecks":waits,"executed":unknown is None,"failure_reason":unknown,
                     "bought_at":c["g"].observed_at.iloc[i].isoformat() if cost is not None else None,
                     "buy_now_vnd":float(current),"paid_vnd":cost,
                     "saving_vs_now_vnd":float(current-cost) if cost is not None else None,
                     "oracle_observed_min_vnd":oracle,"regret_vnd":float(cost-oracle) if cost is not None else None,
                     "stop_reason":reason})
    df=pd.DataFrame(rows)
    if not len(df):return df,{"episodes":0,"error":"NO_COVERED_EPISODES"}
    valid=df[df.executed]
    if not len(valid):return df,{"episodes":len(df),"executed":0,"unexecuted":len(df)}
    metrics={"episodes":len(df),"unique_itineraries":int(df.itinerary_id.nunique()),
             "executed":len(valid),"unexecuted":int((~df.executed).sum()),
             "source_groups":int(df.source_flight_id.nunique()),"initial_waits":int(df.initial_action.eq("WAIT_AND_RECHECK").sum()),
             "mean_saving_vnd":float(valid.saving_vs_now_vnd.mean()),"total_saving_vnd":float(valid.saving_vs_now_vnd.sum()),
             "mean_paid_vnd":float(valid.paid_vnd.mean()),"mean_regret_vnd":float(valid.regret_vnd.mean()),
             "win_rate":float(valid.saving_vs_now_vnd.gt(0).mean()),"loss_rate":float(valid.saving_vs_now_vnd.lt(0).mean()),
             "mean_downside_vnd":float(np.maximum(-valid.saving_vs_now_vnd,0).mean()),
             "minimum_saving_vnd":float(valid.saving_vs_now_vnd.min()),
             "disclaimer":bundle.get("training_data_kind","UNVERIFIED_INPUT_PROVENANCE")+"; internal benchmark; oracle uses observed prices only, not guaranteed market minima"}
    return df,metrics


def cluster_interval(decisions, seed=42, samples=2000, alpha=.05):
    d=decisions[decisions.executed]
    if not len(d):return [None,None]
    g=d.groupby("source_flight_id").saving_vs_now_vnd.agg(["sum","count"])
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(g),size=(samples,len(g)))
    means=g["sum"].to_numpy()[ix].sum(1)/g["count"].to_numpy()[ix].sum(1)
    return np.quantile(means,[alpha/2,1-alpha/2]).tolist()


def save_bundle(bundle,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,path,compress=3)


def load_bundle(path):
    # Only load trusted, locally supplied model artifacts.
    b=joblib.load(path)
    if b.get("schema_version")!=SCHEMA_VERSION:
        raise ValueError("Unsupported model schema")
    return b


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
