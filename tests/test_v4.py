from pathlib import Path
from dataclasses import asdict
import copy
import json
import sys
import unittest

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import airpulse_v4 as m
import api


def toy_quotes():
    start=pd.Timestamp("2026-12-01T00:00:00+07:00")
    hours=[0,3,6,12,18,24,30,36,42,48]
    prices=[1_000_000]*4+[700_000,1_100_000,1_100_000,1_100_000,1_100_000,1_100_000]
    return m.load_quotes(pd.DataFrame([{"itinerary_id":"toy","source_flight_id":"toy-source",
        "observed_at":(start+pd.Timedelta(hours=h)).isoformat(),"departure_at":"2026-12-06T00:00:00+07:00",
        "origin":"SGN","destination":"HAN","airline":"VIETJET","price_vnd":price}
        for h,price in zip(hours,prices)]))


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=m.load_bundle(ROOT/"trained/airpulse_v4_model.joblib")
        cls.payload=json.loads((ROOT/"examples/request_replay.json").read_text())

    def test_dip_then_rebound_is_positive_opportunity(self):
        d=toy_quotes();pairs=m.supervised_pairs(d,{"toy"},m.Settings(),max_snapshots=50)
        r=pairs[(pairs.as_of==pd.Timestamp("2026-12-01T12:00:00+07:00"))&pairs.horizon_hours.eq(12)].iloc[0]
        self.assertEqual(r.y_drop_within,1)
        self.assertEqual(r.target_price_vnd,1_100_000)
        self.assertAlmostEqual(r.y_opportunity_discount,.3)

    def test_future_changes_do_not_change_features(self):
        d=toy_quotes();changed=d.copy();changed.loc[changed.index>=4,"price_vnd"]=9_000_000
        now=d.observed_at.iloc[3]
        self.assertEqual(m.state_features(m.context(d),3,now,m.Settings()),
                         m.state_features(m.context(changed),3,now,m.Settings()))

    def test_censored_window_is_not_labelled_as_no_drop(self):
        d=toy_quotes().iloc[:5]
        pairs=m.supervised_pairs(d,{"toy"},m.Settings(),max_snapshots=50)
        self.assertFalse(pairs.horizon_hours.ge(12).any())

    def test_short_history_is_abstention(self):
        history=pd.DataFrame(self.payload["history"][:4])
        t=pd.Timestamp(history.observed_at.iloc[0])
        history["observed_at"]=[(t+pd.Timedelta(seconds=i)).isoformat() for i in range(4)]
        ans=m.recommend(self.bundle,history,as_of=history.observed_at.iloc[-1])
        self.assertEqual(ans["status"],"INSUFFICIENT_HISTORY")
        self.assertIsNone(ans["recommendation"]["action"])

    def test_stale_history_is_not_buy_now(self):
        history=pd.DataFrame(self.payload["history"])
        at=pd.Timestamp(self.payload["as_of"])+pd.Timedelta(hours=7)
        ans=m.recommend(self.bundle,history,as_of=at)
        self.assertEqual(ans["status"],"STALE_QUOTES")

    def test_unknown_route_is_out_of_scope(self):
        d=pd.DataFrame(self.payload["history"]);d["origin"]="XXX";d["destination"]="YYY"
        ans=m.recommend(self.bundle,d,as_of=self.payload["as_of"])
        self.assertEqual(ans["status"],"OUT_OF_SCOPE")

    def test_future_quote_is_rejected(self):
        with self.assertRaises(m.InputError) as exc:
            m.recommend(self.bundle,pd.DataFrame(self.payload["history"]),as_of=self.payload["history"][3]["observed_at"])
        self.assertEqual(exc.exception.code,"FUTURE_QUOTES")

    def test_past_departure_is_abstention(self):
        d=pd.DataFrame(self.payload["history"])
        at=pd.Timestamp(d.departure_at.iloc[0])+pd.Timedelta(hours=1)
        self.assertEqual(m.recommend(self.bundle,d,as_of=at)["status"],"DEPARTURE_PASSED")

    def test_mixed_seller_or_fare_rejected(self):
        d=pd.DataFrame(self.payload["history"]);d["seller"]="A";d.loc[0,"seller"]="B"
        with self.assertRaises(m.InputError) as exc:m.load_quotes(d)
        self.assertEqual(exc.exception.code,"MIXED_FARE_HISTORY")

    def test_nonfinite_prices_rejected(self):
        d=pd.DataFrame(self.payload["history"]);d.loc[0,"price_vnd"]=float("inf")
        with self.assertRaises(m.InputError):m.load_quotes(d)

    def test_buy_now_timestamp_is_now(self):
        b={**self.bundle,"policy":{"enabled":False}}
        ans=m.recommend(b,pd.DataFrame(self.payload["history"]),as_of=self.payload["as_of"])
        rec=ans["recommendation"]
        self.assertEqual(rec["action"],"BUY_NOW")
        self.assertEqual(rec["recommended_buy_at"],rec["as_of"])
        self.assertIsNone(rec["next_check_at"])

    def test_replay_requires_explicit_clock(self):
        with self.assertRaises(m.InputError) as exc:m.recommend(self.bundle,pd.DataFrame(self.payload["history"]))
        self.assertEqual(exc.exception.code,"AS_OF_REQUIRED")

    def test_live_is_not_enabled_by_demo_request(self):
        d=pd.DataFrame(self.payload["history"])
        ans=m.recommend(self.bundle,d,mode="live",clock=self.payload["as_of"])
        self.assertEqual(ans["status"],"NOT_VALIDATED_FOR_LIVE")
        self.assertIsNone(ans["recommendation"]["action"])

    def test_eligible_slot_is_not_masked(self):
        now=pd.Timestamp("2026-12-01T00:00:00+07:00")
        curve=[{"horizon_hours":12,"prob_drop_within_window":.1,"estimated_opportunity_saving_vnd":200000},
               {"horizon_hours":24,"prob_drop_within_window":.6,"estimated_opportunity_saving_vnd":100000}]
        policy={"enabled":True,"window_hours":24,"min_probability":.5,"min_saving_vnd":50000}
        action,_,chosen=m.decide(curve,1_000_000,now,now+pd.Timedelta(days=3),policy,m.Settings())
        self.assertEqual(action,"WAIT_AND_RECHECK");self.assertEqual(chosen["horizon_hours"],24)

    def test_backtest_never_buys_in_the_past_when_quote_missing(self):
        d=toy_quotes().iloc[:4].copy();c=m.context(d)
        ep={"itinerary_id":"toy","source_flight_id":"toy-source","context":c,
            "deadline":c["departure"]-pd.Timedelta(hours=36),"start":3,"end":3,
            "curves":{3:[{"horizon_hours":24,"prob_drop_within_window":.9,"estimated_opportunity_saving_vnd":200000}]}}
        b={"settings":asdict(m.Settings()),"policy":{"enabled":True,"window_hours":24,"min_probability":.5,"min_saving_vnd":50000}}
        rows,_=m.sequential_backtest(b,[ep])
        self.assertFalse(rows.executed.iloc[0]);self.assertTrue(pd.isna(rows.paid_vnd.iloc[0]))
        self.assertIsNone(rows.bought_at.iloc[0])

    def test_four_partitions_have_no_shared_sources(self):
        p=pd.read_csv(ROOT/"trained/source_partitions.csv")
        self.assertTrue((p.groupby("source_flight_id").partition.nunique()==1).all())
        self.assertEqual(set(p.partition),{"train","calibration","policy_validation","test"})

    def test_wait_response_is_a_recheck_not_a_promised_bottom(self):
        policy={"enabled":True,"window_hours":72,"min_probability":0.,"min_saving_vnd":0}
        b={**self.bundle,"policy":policy}
        ans=m.recommend(b,pd.DataFrame(self.payload["history"]),as_of=self.payload["as_of"])
        rec=ans["recommendation"]
        self.assertEqual(rec["action"],"WAIT_AND_RECHECK")
        self.assertIsNone(rec["recommended_buy_at"])
        self.assertGreater(pd.Timestamp(rec["next_check_at"]),pd.Timestamp(rec["as_of"]))

    def test_economic_gate_rejects_expected_price_increase(self):
        now=pd.Timestamp("2026-12-01T00:00:00+07:00")
        curve=[{"horizon_hours":6,"expected_6h_saving_vnd":-30000,
                "prob_drop_within_window":.9,"estimated_opportunity_saving_vnd":200000}]
        policy={"enabled":True,"window_hours":24,"min_probability":.5,"min_saving_vnd":50000,
                "min_expected_6h_saving_vnd":0}
        action,reason,_=m.decide(curve,1000000,now,now+pd.Timedelta(days=3),policy,m.Settings())
        self.assertEqual(action,"BUY_NOW");self.assertEqual(reason,"NO_NEAR_TERM_EXPECTED_SAVING")

    def test_backtest_covers_multiple_entry_times_without_new_independent_flights(self):
        start=pd.Timestamp("2026-12-01T00:00:00+07:00")
        d=m.load_quotes(pd.DataFrame([{"itinerary_id":"entry-demo","source_flight_id":"one-source",
             "observed_at":(start+pd.Timedelta(hours=h)).isoformat(),
             "departure_at":(start+pd.Timedelta(hours=240)).isoformat(),"origin":"SGN","destination":"HAN",
             "airline":"VIETJET","price_vnd":1000000} for h in range(0,214,3)]))
        episodes,_=m.prepare_episodes(self.bundle,d,{"entry-demo"})
        self.assertGreater(len({ep["start"] for ep in episodes}),1)
        self.assertEqual(len({ep["itinerary_id"] for ep in episodes}),1)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload=json.loads((ROOT/"examples/request_replay.json").read_text())
        cls.ctx=TestClient(api.app);cls.client=cls.ctx.__enter__()

    @classmethod
    def tearDownClass(cls):cls.ctx.__exit__(None,None,None)

    def test_both_routes_share_one_contract(self):
        a=self.client.post("/predict",json=self.payload);b=self.client.post("/insight",json=self.payload)
        self.assertEqual(a.status_code,200);self.assertEqual(a.json(),b.json())
        self.assertEqual(a.json()["schema_version"],m.SCHEMA_VERSION)

    def test_cors_preflight_for_demo(self):
        r=self.client.options("/insight",headers={"Origin":"https://airpulse-demo.vercel.app",
             "Access-Control-Request-Method":"POST","Access-Control-Request-Headers":"content-type"})
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.headers.get("access-control-allow-origin"),"https://airpulse-demo.vercel.app")

    def test_unexpected_origin_not_allowed(self):
        r=self.client.options("/insight",headers={"Origin":"https://not-allowed.example",
             "Access-Control-Request-Method":"POST"})
        self.assertEqual(r.status_code,400)
        self.assertIsNone(r.headers.get("access-control-allow-origin"))

    def test_naive_timestamp_rejected(self):
        p=copy.deepcopy(self.payload);p["history"][0]["observed_at"]="2026-12-01T00:00:00"
        self.assertEqual(self.client.post("/insight",json=p).status_code,422)

    def test_extra_unverified_fields_not_silently_ignored(self):
        p=copy.deepcopy(self.payload);p["history"][0]["invented_price_guarantee"]=True
        self.assertEqual(self.client.post("/insight",json=p).status_code,422)

    def test_missing_replay_asof_is_clear_error(self):
        p=copy.deepcopy(self.payload);p.pop("as_of")
        r=self.client.post("/insight",json=p)
        self.assertEqual(r.status_code,422);self.assertEqual(r.json()["detail"]["code"],"AS_OF_REQUIRED")

    def test_health_exposes_research_status(self):
        r=self.client.get("/health")
        self.assertTrue(r.json()["model_loaded"]);self.assertFalse(r.json()["live_validated"])


if __name__=="__main__":unittest.main(verbosity=2)
