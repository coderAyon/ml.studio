"""Typed project snapshots, preserving infinities separately from JSON missing values."""
import hashlib
import io
import json
import base64
import numpy as np
import pandas as pd


def pack_frame(frame):
    raw=frame.to_json(orient='table',index=False,date_format='iso',date_unit='ns',double_precision=15)
    infinity=[]
    for col in frame.select_dtypes(include='number'):
        values=frame[col].to_numpy(dtype=float,na_value=np.nan)
        for pos in np.flatnonzero(np.isinf(values)):
            infinity.append([int(pos),str(col),1 if values[pos]>0 else -1])
    floats=[]
    for col in frame:
        if str(frame[col].dtype) in {'float64','float32','Float64','Float32'}:
            values=frame[col].to_numpy(dtype='<f8',na_value=np.nan)
            floats.append({'column':str(col),'dtype':str(frame[col].dtype),'data':base64.b64encode(values.tobytes()).decode()})
    bundle={'data':raw,'infinity':infinity,'floats':floats}
    bundle['sha256']=hashlib.sha256(json.dumps(bundle,sort_keys=True).encode()).hexdigest()
    bundle['encoding']='typed-v3'
    return bundle


def unpack_frame(bundle):
    if bundle.get('encoding') in {'typed-v2','typed-v3'}:
        core={k:bundle[k] for k in (['data','infinity','floats'] if bundle['encoding']=='typed-v3' else ['data','infinity'])}
        digest=hashlib.sha256(json.dumps(core,sort_keys=True).encode()).hexdigest()
    else:digest=hashlib.sha256(bundle['data'].encode()).hexdigest()
    if digest!=bundle['sha256']:raise ValueError('Embedded dataset integrity check failed.')
    frame=pd.read_json(io.StringIO(bundle['data']),orient='table')
    for pos,col,sign in bundle.get('infinity',[]):
        if type(pos) is not int or not 0<=pos<len(frame) or col not in frame or sign not in [-1,1]:raise ValueError('Invalid stored infinity position.')
        frame.iat[pos,frame.columns.get_loc(col)]=np.inf*sign
    for block in bundle.get('floats',[]):
        if block['column'] not in frame or block['dtype'] not in {'float64','float32','Float64','Float32'}:raise ValueError('Invalid numeric snapshot column.')
        values=np.frombuffer(base64.b64decode(block['data'],validate=True),dtype='<f8')
        if len(values)!=len(frame):raise ValueError('Invalid numeric snapshot length.')
        frame[block['column']]=pd.Series(values,index=frame.index,dtype=block['dtype'])
    return frame
