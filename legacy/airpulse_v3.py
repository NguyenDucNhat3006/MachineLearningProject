"""AirPulse V3 | Direct future-quote forecast + meaningful-price-drop probability.

Research prototype. Training/testing only on semi-synthetic quotes; no claims of
real-market reliability. SOURCE FLIGHT GROUP HOLDOUT is mandatory when available.

Input quotes: itinerary_id, observed_at, departure_at, origin, destination,
              airline, price_vnd; source_flight_id for grouped evaluation.
Timestamp inputs require timezone offsets. Extra columns are ignored.

Forecast horizons are delays from NOW in hours. A recommendation at +h is an
approximate quote-check time, not a guaranteed optimal transaction timestamp.
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, brier_score_loss, roc_auc_score
from lightgbm import LGBMRegressor, LGBMClassifier

TZ = 'Asia/Ho_Chi_Minh'
HORIZONS = (6, 12, 18, 24, 36, 48, 72, 96, 120, 144, 168)
CATS = ['origin', 'destination', 'airline']
NUMS = [
    'current_logprice', 'days_to_departure', 'departure_hour', 'departure_dow',
    'departure_month', 'snapshot_hour', 'snapshot_dow', 'snapshot_month',
    'history_hours', 'quote_count_24h', 'quote_count_72h',
    'changes_24h', 'changes_72h', 'hours_since_last_change',
    'last_return', 'return_24h', 'return_72h',
    'rel_min_72h', 'rel_median_72h', 'rel_max_72h',
    'horizon_hours', 'future_days_to_departure', 'future_hour', 'future_dow',
]
TOLERANCE_HOURS = 2.05
MIN_LEAD_HOURS = 24
MAX_FORECAST_HOURS = 168
DROP_FRACTION = 0.04


def load_quotes(path_or_df):
    d = pd.read_csv(path_or_df) if not isinstance(path_or_df, pd.DataFrame) else path_or_df.copy()
    required = ['itinerary_id','observed_at','departure_at','origin','destination','airline','price_vnd']
    missing = set(required) - set(d.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    for c in ['observed_at','departure_at']:
        if not d[c].astype(str).str.contains(r'(?:Z|[+-]\d\d:\d\d)$', regex=True).all():
            raise ValueError(f'{c} requires an ISO8601 timezone offset')
        d[c] = pd.to_datetime(d[c], utc=True, format='ISO8601').dt.tz_convert(TZ)
    d['price_vnd'] = pd.to_numeric(d.price_vnd,errors='coerce')
    d=d.dropna(subset=['price_vnd'])
    d=d[(d.price_vnd>0)&(d.observed_at<d.departure_at)].copy()
    for c in CATS: d[c]=d[c].fillna('UNKNOWN').astype(str).str.upper().str.strip()
    d['itinerary_id']=d.itinerary_id.astype(str)
    if 'source_flight_id' not in d.columns:
        # Valid for live inference; training group-holdout is not honest without source IDs.
        d['source_flight_id']=d.itinerary_id
        d.attrs['missing_source_ids']=True
    d['source_flight_id']=d.source_flight_id.astype(str)
    d=d.sort_values(['itinerary_id','observed_at']).reset_index(drop=True)
    for c in ['departure_at','origin','destination','airline','source_flight_id']:
        if (d.groupby('itinerary_id')[c].nunique()>1).any():
            raise ValueError(f'Inconsistent {c} inside one itinerary_id')
    conflict=d.groupby(['itinerary_id','observed_at']).price_vnd.nunique()
    if (conflict>1).any(): raise ValueError('Conflicting quotes at identical itinerary/time')
    return d.drop_duplicates(['itinerary_id','observed_at']).reset_index(drop=True)


def _state(g, i):
    t = g.observed_at
    now=t.iloc[i]; dep=g.departure_at.iloc[0]
    p=g.price_vnd.to_numpy(dtype=float)
    hours=(t.astype('int64').to_numpy()/3.6e12)
    ix24=int(np.searchsorted(hours, hours[i]-24, side='left'))
    ix72=int(np.searchsorted(hours, hours[i]-72, side='left'))
    values24=p[ix24:i+1]; values72=p[ix72:i+1]
    change_idxs=np.where(np.diff(p[:i+1])!=0)[0]
    hours_since_change=float(hours[i]-hours[change_idxs[-1]+1]) if len(change_idxs) else float(hours[i]-hours[0])

    def lag_return(lag, tolerance=8.):
        target=hours[i]-lag
        j=int(np.searchsorted(hours[:i],target,side='right')-1)
        if j<0 or abs(hours[i]-hours[j]-lag)>tolerance: return 0.0
        return float(np.log(p[i]/p[j]))

    f={
        'origin':g.origin.iloc[0],'destination':g.destination.iloc[0],
        'airline':g.airline.iloc[0],
        'current_logprice':float(np.log(p[i])),
        'days_to_departure':float((dep-now).total_seconds()/86400),
        'departure_hour':int(dep.hour),'departure_dow':int(dep.dayofweek),
        'departure_month':int(dep.month),'snapshot_hour':int(now.hour),
        'snapshot_dow':int(now.dayofweek),'snapshot_month':int(now.month),
        'history_hours':float(hours[i]-hours[0]),
        'quote_count_24h':len(values24),'quote_count_72h':len(values72),
        'changes_24h':int(np.count_nonzero(np.diff(values24))),
        'changes_72h':int(np.count_nonzero(np.diff(values72))),
        'hours_since_last_change':min(hours_since_change,240),
        'last_return':float(np.log(p[i]/p[i-1])) if i>=1 else 0.0,
        'return_24h':lag_return(24),'return_72h':lag_return(72),
        'rel_min_72h':float(np.log(p[i]/np.min(values72))),
        'rel_median_72h':float(np.log(p[i]/np.median(values72))),
        'rel_max_72h':float(np.log(p[i]/np.max(values72))),
    }
    return f, hours, p


def _h_feature(base,now,dep,h):
    future=now+pd.Timedelta(hours=int(h))
    f=base.copy()
    f.update(horizon_hours=float(h),future_days_to_departure=float((dep-future).total_seconds()/86400),
             future_hour=int(future.hour),future_dow=int(future.dayofweek))
    return f


def _valid_horizons(now,dep):
    max_allowed=min(MAX_FORECAST_HOURS,(dep-now).total_seconds()/3600-MIN_LEAD_HOURS)
    return [h for h in HORIZONS if h <= max_allowed]


def _nearest_future(hours,i,target,tolerance=TOLERANCE_HOURS):
    j=int(np.searchsorted(hours,target,side='left'))
    js=[x for x in (j-1,j) if i<x<len(hours)]
    if not js: return None
    best=min(js,key=lambda x:abs(hours[x]-target))
    return best if abs(hours[best]-target)<=tolerance else None


def build_training_pairs(quotes, allowed_ids, max_snapshots=18):
    result=[]
    eligible=quotes[quotes.itinerary_id.isin(allowed_ids)]
    for iid,g in eligible.groupby('itinerary_id',sort=False):
        g=g.sort_values('observed_at').reset_index(drop=True)
        if len(g)<12: continue
        # Starting from 4th past quote: enough history for a new-user demo.
        idxs=np.arange(3,len(g)-2)
        if len(idxs)>max_snapshots:
            idxs=np.unique(np.linspace(idxs[0],idxs[-1],max_snapshots).astype(int))
        dep=g.departure_at.iloc[0]
        for i in idxs:
            base,hours,prices=_state(g,int(i))
            now=g.observed_at.iloc[i]
            for h in _valid_horizons(now,dep):
                j=_nearest_future(hours,int(i),hours[i]+h)
                if j is None: continue
                future=float(prices[j]); cur=float(prices[i])
                f=_h_feature(base,now,dep,h)
                f.update(itinerary_id=iid,source_flight_id=g.source_flight_id.iloc[0],
                         asof=now,target_at=g.observed_at.iloc[j],
                         current_price_vnd=cur,actual_price_vnd=future,
                         y_discount=float(np.clip((cur-future)/cur,-1.5,.9)),
                         y_drop=int(future<=cur*(1-DROP_FRACTION)))
                result.append(f)
    if not result: raise ValueError('No supervised examples: check quote history and cadence')
    return pd.DataFrame(result)


def split_sources(df,seed=42):
    ids=df[['itinerary_id','source_flight_id']].drop_duplicates().reset_index(drop=True)
    if ids.source_flight_id.nunique()<15: raise ValueError('Need >=15 distinct source_flight_id for grouped validation')
    train_idx, tmp_idx=next(GroupShuffleSplit(n_splits=1,test_size=.30,random_state=seed).split(ids,groups=ids.source_flight_id))
    trainids=ids.iloc[train_idx]
    temp=ids.iloc[tmp_idx]
    vi,ti=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=seed+1).split(temp,groups=temp.source_flight_id))
    valids=temp.iloc[vi]; testids=temp.iloc[ti]
    groups={'train':set(trainids.itinerary_id),'validation':set(valids.itinerary_id),'test':set(testids.itinerary_id)}
    for a in groups:
        for b in groups:
            if a>=b: continue
            sa=set(ids[ids.itinerary_id.isin(groups[a])].source_flight_id)
            sb=set(ids[ids.itinerary_id.isin(groups[b])].source_flight_id)
            assert sa.isdisjoint(sb),f'Source leakage between {a}/{b}'
    return groups


def _prep():
    return ColumnTransformer([
        ('cat',OneHotEncoder(handle_unknown='ignore'),CATS),
        ('num','passthrough',NUMS),
    ],remainder='drop')


def train(quotes,max_snapshots=18,n_estimators=160,seed=42):
    if quotes.attrs.get('missing_source_ids',False):
        raise ValueError('source_flight_id required for honest group holdout during training')
    groups=split_sources(quotes,seed)
    trainpairs=build_training_pairs(quotes,groups['train'],max_snapshots=max_snapshots)
    valpairs=build_training_pairs(quotes,groups['validation'],max_snapshots=max(12,max_snapshots//2))
    testpairs=build_training_pairs(quotes,groups['test'],max_snapshots=max(12,max_snapshots//2))
    prep=_prep()
    Xt=prep.fit_transform(trainpairs[CATS+NUMS]);Xv=prep.transform(valpairs[CATS+NUMS]);Xs=prep.transform(testpairs[CATS+NUMS])
    reg=LGBMRegressor(n_estimators=n_estimators,learning_rate=.05,num_leaves=15,
                      min_child_samples=70,max_depth=6,objective='regression',verbose=-1,random_state=seed,n_jobs=2)
    clf=LGBMClassifier(n_estimators=n_estimators,learning_rate=.05,num_leaves=15,
                       min_child_samples=70,max_depth=6,verbose=-1,random_state=seed+1,n_jobs=2)
    reg.fit(Xt,trainpairs.y_discount.to_numpy());clf.fit(Xt,trainpairs.y_drop.to_numpy())
    bundle={'preprocess':prep,'discount_model':reg,'drop_model':clf,
            'horizons':HORIZONS,'min_lead_hours':MIN_LEAD_HOURS,
            'max_forecast_hours':MAX_FORECAST_HOURS,'price_drop_threshold':DROP_FRACTION,
            'training_kind':'SEMI_SYNTHETIC; EVALUATE ON REAL NEW DEPARTURES BEFORE LIVE USE',
            'feature_columns':CATS+NUMS,'source_group_split_seed':seed,
            'decision_min_discount':.03,'decision_min_probability':.20}
    metrics={}
    for name,pairs,tx in [('validation',valpairs,Xv),('test',testpairs,Xs)]:
        pred_disc=np.clip(reg.predict(tx),-1.5,.9)
        prob=np.clip(clf.predict_proba(tx)[:,1],0,1)
        real=pairs.actual_price_vnd.to_numpy()
        now=pairs.current_price_vnd.to_numpy()
        model_price=now*(1-pred_disc)
        metrics[name]={
            'pair_count':int(len(pairs)),'distinct_itineraries':int(pairs.itinerary_id.nunique()),
            'price_mae_vnd':int(round(mean_absolute_error(real,model_price))),
            'buy_now_price_mae_vnd':int(round(mean_absolute_error(real,now))),
            'drop_prevalence':round(float(pairs.y_drop.mean()),4),
            'drop_brier':round(float(brier_score_loss(pairs.y_drop,prob)),5),
            'drop_auc':round(float(roc_auc_score(pairs.y_drop,prob)),4) if pairs.y_drop.nunique()==2 else None,
        }
    return bundle,groups,metrics


def forecast(bundle,history,asof=None):
    """History must contain actual prices at <=asof; no target timestamps after asof."""
    d=load_quotes(history)
    if d.itinerary_id.nunique()!=1:raise ValueError('Use a single itinerary_id in live inference')
    g=d.sort_values('observed_at').reset_index(drop=True)
    if asof is not None:
        at=pd.Timestamp(asof)
        if at.tz is None: raise ValueError('asof must have timezone')
        g=g[g.observed_at<=at.tz_convert(TZ)].reset_index(drop=True)
    if len(g)<4:raise ValueError('Need at least 4 past quotes for forecast')
    i=len(g)-1;now=g.observed_at.iloc[i];dep=g.departure_at.iloc[0]
    horizon=_valid_horizons(now,dep)
    if not horizon: raise ValueError('Departure too soon or no valid forecast slots')
    base,_,prices=_state(g,i)
    xf=pd.DataFrame([_h_feature(base,now,dep,h) for h in horizon])
    x=bundle['preprocess'].transform(xf[CATS+NUMS])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='X does not have valid feature names.*',category=UserWarning)
        discount=np.clip(bundle['discount_model'].predict(x),-1.5,.9)
        proba=bundle['drop_model'].predict_proba(x)[:,1]
    out=pd.DataFrame({'horizon_hours':horizon,
                      'buy_at':[now+pd.Timedelta(hours=int(h)) for h in horizon],
                      'predicted_price_vnd':np.round(prices[i]*(1-discount)).astype(int),
                      'predicted_discount_pct':np.round(discount*100,2),
                      'prob_at_least_4pct_drop':np.round(proba,4)})
    return out,{'itinerary_id':g.itinerary_id.iloc[0],'asof':now,'departure_at':dep,
                'now_price_vnd':int(prices[i]),'quotes_seen':len(g)}


def recommend(bundle,history,asof=None,forced_thresholds=None):
    curve,info=forecast(bundle,history,asof=asof)
    # Choose horizon with highest estimated discount, adjusted for drop likelihood.
    # All score inputs are predictions, never future prices.
    score=curve.predicted_discount_pct.to_numpy()+3.*curve.prob_at_least_4pct_drop.to_numpy()
    chosen=curve.iloc[int(np.argmax(score))]
    min_discount=float(bundle['decision_min_discount'])
    min_prob=float(bundle['decision_min_probability'])
    if forced_thresholds is not None:min_discount,min_prob=forced_thresholds
    wait=(chosen.predicted_discount_pct>=100*min_discount and chosen.prob_at_least_4pct_drop>=min_prob)
    result={**info,'action':'WAIT_AND_RECHECK' if wait else 'BUY_NOW',
           'best_window_start':chosen.buy_at.isoformat(),
           'best_horizon_hours':int(chosen.horizon_hours),
           'forecast_price_vnd':int(chosen.predicted_price_vnd),
           'forecast_discount_pct':float(chosen.predicted_discount_pct),
           'prob_at_least_4pct_drop':float(chosen.prob_at_least_4pct_drop),
           'recheck_interval_hours':3 if wait else 6,
           'note':'Research estimate for one price-quote window. Not guaranteed cheapest fare or verified payable total.'}
    return result,curve


def _select_one_snapshot(g):
    if len(g)<12:return None
    # First snapshot after some price-history context, before last 24h cutoff.
    i=min(12,len(g)//4)
    if i<3:i=3
    dep=g.departure_at.iloc[0]
    for k in range(i,len(g)-3):
        now=g.observed_at.iloc[k]
        if _valid_horizons(now,dep) and (dep-now).total_seconds()/3600>=48:
            return k
    return None


def one_shot_backtest(bundle,quotes,allowed_ids,thresholds=None):
    rows=[]
    for iid,g in quotes[quotes.itinerary_id.isin(allowed_ids)].groupby('itinerary_id',sort=False):
        g=g.sort_values('observed_at').reset_index(drop=True)
        i=_select_one_snapshot(g)
        if i is None:continue
        hist=g.iloc[:i+1]
        rec,curve=recommend(bundle,hist,forced_thresholds=thresholds)
        now=hist.observed_at.iloc[-1];p=float(hist.price_vnd.iloc[-1]);dep=hist.departure_at.iloc[0]
        future=g[(g.observed_at>now)&(g.observed_at<=now+pd.Timedelta(hours=MAX_FORECAST_HOURS+TOLERANCE_HOURS))
                 &(g.observed_at<=dep-pd.Timedelta(hours=MIN_LEAD_HOURS))]
        if len(future)<2: continue
        oracle=min(p,float(future.price_vnd.min()))
        target=pd.Timestamp(rec['best_window_start'])
        diffs=(future.observed_at-target).abs().dt.total_seconds()/3600
        available=bool(len(diffs)>0 and diffs.min()<=TOLERANCE_HOURS)
        chosen_actual=float(future.loc[diffs.idxmin(),'price_vnd']) if available else np.nan
        wants_wait=rec['action']=='WAIT_AND_RECHECK'
        # If no price near chosen check time: conservatively count as buying now,
        # not a magic successful purchase. Track as unexecutable prediction.
        executable_wait=bool(wants_wait and available)
        cost=chosen_actual if executable_wait else p
        peak_prices=future.loc[future.price_vnd==future.price_vnd.min(),'observed_at']
        best_obs=now if p<=float(future.price_vnd.min()) else peak_prices.iloc[0]
        timing_err=abs((target-best_obs).total_seconds()/3600)
        actual_buy_at=target if executable_wait else now
        decision_timing_err=abs((actual_buy_at-best_obs).total_seconds()/3600)
        rows.append({'itinerary_id':iid,'source_flight_id':g.source_flight_id.iloc[0],
                     'asof':now.isoformat(),'action':rec['action'],
                     'best_slot':target.isoformat(),
                     'forecast_discount_pct':rec['forecast_discount_pct'],
                     'prob_at_least_4pct_drop':rec['prob_at_least_4pct_drop'],
                     'buy_now_vnd':p,'selected_slot_observed':available,
                     'selected_slot_actual_vnd':chosen_actual,
                     'one_shot_cost_vnd':cost,'saving_vs_now_vnd':p-cost,
                     'oracle_min_vnd':oracle,'regret_vnd':cost-oracle,
                     'best_slot_time_error_hours':timing_err,
                     'executed_decision_time_error_hours':decision_timing_err,
                     'number_future_observations':len(future)})
    result=pd.DataFrame(rows)
    if result.empty:return result,{'error':'No evaluable test flights'}
    waits=result.action.eq('WAIT_AND_RECHECK')
    info={
      'n_itineraries':int(len(result)),
      'n_source_templates':int(result.source_flight_id.nunique()),
      'wait_recommendations':int(waits.sum()),
      'wait_not_observable_at_selected_slot':int((waits&~result.selected_slot_observed).sum()),
      'mean_saving_vnd':round(float(result.saving_vs_now_vnd.mean())),
      'beat_buy_now_rate':round(float((result.saving_vs_now_vnd>0).mean()),4),
      'worse_than_buy_now_rate':round(float((result.saving_vs_now_vnd<0).mean()),4),
      'mean_regret_vnd':round(float(result.regret_vnd.mean())),
      'median_timing_error_hours':round(float(result.best_slot_time_error_hours.median()),2),
      'within_24h_of_hindsight_min_rate':round(float((result.best_slot_time_error_hours<=24).mean()),4),
      'decision_median_timing_error_hours':round(float(result.executed_decision_time_error_hours.median()),2),
      'mean_buy_now_vnd':round(float(result.buy_now_vnd.mean())),
      'mean_booked_vnd':round(float(result.one_shot_cost_vnd.mean())),
      'caveat':'ALL EVALUATION IS ON SEMI-SYNTHETIC DATA. Grouped template holdout, not new real market departures. Time error uses earliest observed future min or NOW if cheaper; price at chosen slot is observed only if within tolerance.'}
    return result,info


def tune_policy(bundle,quotes,ids):
    # Threshold selection uses ONLY validation itineraries and never the test set.
    candidates=[(.01,.10),(.02,.15),(.03,.20),(.04,.20),(.04,.30),(.06,.30),(.08,.40),(.10,.50)]
    best=None;best_score=-1e15;allrows=[]
    for th in candidates:
        pred,stats=one_shot_backtest(bundle,quotes,ids,thresholds=th)
        if pred.empty:continue
        # Penalize downside losses, but allow all-BUY baseline when no policy beats it.
        score=float(pred.saving_vs_now_vnd.mean())-0.5*float(np.maximum(-pred.saving_vs_now_vnd,0).mean())
        allrows.append({'min_predicted_discount':th[0],'min_drop_probability':th[1],
                        'score_vnd':round(score),**{k:v for k,v in stats.items() if k!='caveat'}})
        if score>best_score+1e-9:
            best_score=score;best=th
    if best is None:raise ValueError('No validation decision samples')
    # If all candidates lose, select a stricter default that buys now.
    if best_score<0:best=(1.0,1.0)
    bundle['decision_min_discount'],bundle['decision_min_probability']=best
    return best,pd.DataFrame(allrows)


def run_pipeline(base_csv,outdir,seed=42,n_estimators=160):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    quotes=load_quotes(base_csv)
    if 'source_flight_id' not in pd.read_csv(base_csv,nrows=1).columns:
        raise ValueError('This training recipe requires source_flight_id to prevent template contamination')
    bundle,groups,metrics=train(quotes,n_estimators=n_estimators,seed=seed)
    thresholds,val_grid=tune_policy(bundle,quotes,groups['validation'])
    val_df,val_stats=one_shot_backtest(bundle,quotes,groups['validation'])
    test_df,test_stats=one_shot_backtest(bundle,quotes,groups['test'])
    metrics['booking_policy']={'thresholds':{'min_expected_discount':thresholds[0],'min_probability_4pct_drop':thresholds[1]},
                               'validation':val_stats,'test':test_stats}
    metrics['dataset']={'rows':len(quotes),'itineraries':quotes.itinerary_id.nunique(),
                        'source_templates':quotes.source_flight_id.nunique(),
                        'mean_quote_change_rate':float(quotes.sort_values(['itinerary_id','observed_at']).groupby('itinerary_id').price_vnd.diff().fillna(0).ne(0).mean()),
                        'split_itineraries':{k:len(v) for k,v in groups.items()},
                        'split_source_templates':{k:int(quotes[quotes.itinerary_id.isin(v)].source_flight_id.nunique()) for k,v in groups.items()}}
    joblib.dump(bundle,out/'airpulse_v3_model.joblib',compress=3)
    val_grid.to_csv(out/'validation_policy_grid.csv',index=False)
    val_df.to_csv(out/'validation_booking_decisions.csv',index=False)
    test_df.to_csv(out/'test_booking_decisions.csv',index=False)
    pd.DataFrame([{'itinerary_id':i,'split':part} for part,ids in groups.items() for i in sorted(ids)]).to_csv(out/'source_group_split.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    return bundle,metrics,test_df


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True,help='BASE CSV including source_flight_id')
    parser.add_argument('--out',default='airpulse_v3_outputs')
    parser.add_argument('--trees',type=int,default=160)
    a=parser.parse_args()
    _,metrics,_=run_pipeline(a.data,a.out,n_estimators=a.trees)
    print(json.dumps(metrics,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':main()
