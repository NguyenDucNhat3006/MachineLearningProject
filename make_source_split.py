"""Create a source-group-disjoint split for a NEW input CSV, before fitting."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from airpulse_v4 import load_quotes


def run(data,out,seed=42):
    if "source_flight_id" not in pd.read_csv(data,nrows=1):
        raise ValueError("Explicit original source_flight_id is required")
    d=load_quotes(data)[["itinerary_id","source_flight_id"]].drop_duplicates().reset_index(drop=True)
    if d.source_flight_id.nunique()<30:
        raise ValueError("This four-part training workflow requires at least 30 source groups; this does not guarantee good performance")
    tr,tmp=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=seed).split(d,groups=d.source_flight_id))
    temp=d.iloc[tmp]
    vi,ti=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=seed+1).split(temp,groups=temp.source_flight_id))
    d["split"]=""
    d.loc[tr,"split"]="train"
    d.loc[temp.iloc[vi].index,"split"]="validation"
    d.loc[temp.iloc[ti].index,"split"]="test"
    assert (d.groupby("source_flight_id").split.nunique()==1).all()
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    d[["itinerary_id","split"]].to_csv(out,index=False)
    print(d.groupby("split").source_flight_id.nunique().to_dict())


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",required=True,type=Path);p.add_argument("--out",required=True,type=Path)
    a=p.parse_args();run(a.data,a.out)
