"""Create an explicit, semi-synthetic replay request from held-out source IDs."""
from pathlib import Path
import json
import pandas as pd
import airpulse_v4 as m

ROOT=Path(__file__).resolve().parent


def create_example():
    bundle=m.load_bundle(ROOT/"trained/airpulse_v4_model.joblib")
    quotes=m.load_quotes(ROOT/"data/AirPulse_V2_SEMISYNTHETIC_BASE.csv")
    parts=pd.read_csv(ROOT/"trained/source_partitions.csv")
    ids=set(parts.loc[parts.partition.eq("test"),"itinerary_id"])
    iid=sorted(ids)[0]
    g=quotes[quotes.itinerary_id.eq(iid)].iloc[:13].copy()
    columns=[*m.REQUIRED,"fare_family","seller","cabin","baggage","currency"]
    for c in ["observed_at","departure_at"]:g[c]=g[c].map(lambda t:t.isoformat())
    payload={"mode":"replay","as_of":g.observed_at.iloc[-1],"history":g[columns].to_dict(orient="records")}
    request_path=ROOT/"examples/request_replay.json"
    request_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    answer=m.recommend(bundle,pd.DataFrame(payload["history"]),mode="replay",as_of=payload["as_of"])
    (ROOT/"examples/response_replay.json").write_text(json.dumps(answer,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(answer,ensure_ascii=False,indent=2))
    return payload,answer


if __name__=="__main__":create_example()
