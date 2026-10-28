"""Finite synthetic coverage/power diagnostic; never evidence for real-model H2."""
import argparse
import math
from pathlib import Path
import numpy as np
from h2.common import ROOT,now,require_pbs,write_json
from h2.geometry import (affinity_matrix,pair_vector,differences,joint_group_bootstrap_indices,
                         simultaneous_intervals,inversion_flags)

def wilson(k,n):
    z=1.959963984540054; p=k/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-radius,center+radius]

def experiment(repetitions=30,bootstrap=199,n=80,seed=9421):
    rng=np.random.default_rng(seed); results={}
    for planted in (False,True):
        detected=covered=0
        for rep in range(repetitions):
            initial=[0,.25,1.2]; final=[0,1.2,.25] if planted else initial
            data=[{},{}]; population=[]
            groups={t:np.asarray([f"{t}:{i}" for i in range(n)]) for t in ("a","b","c")}
            for endpoint,angles in enumerate((initial,final)):
                bases=np.asarray([[np.cos(a),np.sin(a),0,0,0,0] for a in angles])
                _,values=pair_vector(np.square(bases@bases.T)); _,d=differences(values)
                population.append(d)
                for task,u in zip(groups,bases):
                    if endpoint==0:
                        pass
                    # Shared row IDs make bootstrap draws paired. The synthetic model's
                    # input can include every latent random coordinate used here.
                    x=rng.normal(size=(n,1))*u*4+rng.normal(size=(n,6))*.3
                    data[endpoint][task]=x
            point=[]
            for endpoint in (0,1):
                _,a,_=affinity_matrix(data[endpoint],1); _,v=pair_vector(a); _,d=differences(v); point.append(d)
            point=np.asarray(point); boot=[]
            for b in range(bootstrap):
                indices=joint_group_bootstrap_indices(groups,rng); sample=[]
                for endpoint in (0,1):
                    _,a,_=affinity_matrix({t:data[endpoint][t][indices[t]] for t in groups},1)
                    _,v=pair_vector(a); _,d=differences(v); sample.append(d)
                boot.append(sample)
            lo,hi,_=simultaneous_intervals(point,np.asarray(boot))
            covered+=int(np.all((np.asarray(population)>=lo)&(np.asarray(population)<=hi)))
            detected+=int(inversion_flags(lo,hi,.01).any())
        results["planted" if planted else "stable_population"]={
            "detections":detected,"repetitions":repetitions,"detection_rate":detected/repetitions,
            "detection_rate_wilson_95_interval":wilson(detected,repetitions),
            "simultaneous_population_coverage":covered/repetitions,
            "coverage_wilson_95_interval":wilson(covered,repetitions)}
    return {"created_at":now(),"seed":seed,"bootstrap":bootstrap,"n":n,"results":results,
        "scope":"Synthetic Gaussian instrument diagnostic only; not real-model H2 evidence, "
                "a power justification for transformers, or a coverage guarantee"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repetitions",type=int,default=100); p.add_argument("--bootstrap",type=int,default=499)
    p.add_argument("--local-fixture",action="store_true"); p.add_argument("--output",default="reports/instrument_validation.json")
    a=p.parse_args()
    if a.repetitions<2 or a.bootstrap<20: raise ValueError("At least 2 repetitions and 20 resamples required")
    if a.local_fixture:
        if a.repetitions>50 or a.bootstrap>199: raise ValueError("Use CPU PBS for larger diagnostics")
    else: require_pbs("cpuq")
    out=(ROOT/a.output).resolve()
    if not out.is_relative_to(ROOT):raise ValueError("Output must remain within the project")
    result=experiment(a.repetitions,a.bootstrap)
    write_json(out,result)
    print(result)

if __name__=="__main__":main()
