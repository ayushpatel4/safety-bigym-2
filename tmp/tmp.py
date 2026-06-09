import pandas as pd, glob, os
print("EE-retract sweep (cbf_mode=ee, coworker_train noisy):")
for f in sorted(glob.glob("results/e4_1/hybrid_cbf_ee_d0p*.csv")):
    r = pd.read_csv(f).iloc[-1]
    print(" ", os.path.basename(f),
        "succ=%.2f prox=%.3f interv=%.1f%% eplen=%.0f" %
        (r.success_rate, r.ep_proximity_violation_rate,
            r.filter_intervention_rate*100, r.mean_episode_length))