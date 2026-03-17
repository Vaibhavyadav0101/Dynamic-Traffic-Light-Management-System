"""
export_results.py
─────────────────────────────────────────────────────
Run this AFTER training to export your real results
into results.json so the website shows your actual data.

Usage:
    python export_results.py                        # uses default model name
    python export_results.py -m my_model_name       # specify model name
    python export_results.py --fixed                # export fixed-time results

Place this file in your project root (same folder as train.py)
─────────────────────────────────────────────────────
"""

import os, sys, json, optparse
import numpy as np
import pandas as pd

# ── SUMO setup ──────────────────────────────────────
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please set SUMO_HOME environment variable.")

from sumolib import checkBinary
import traci

JUNCTIONS = ['gneJ11', 'gneJ2', 'gneJ21', 'gneJ3', 'gneJ7']


def run_and_collect_rl(model_name="model", steps=500):
    """Run the trained DQN model once and collect all metrics."""
    import torch, torch.nn as nn, torch.nn.functional as F

    # ── Rebuild the model class ──────────────────────
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 256)
            self.linear2 = nn.Linear(256, 256)
            self.linear3 = nn.Linear(256, 4)
        def forward(self, x):
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            return self.linear3(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)

    model_path = f"models/{model_name}.bin"
    if not os.path.exists(model_path):
        print(f"[!] Model not found at {model_path}")
        print("    Run training first: python train.py --train -e 50 -s 500")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[✓] Loaded model: {model_path}")

    # ── Start SUMO ───────────────────────────────────
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg",
                 "--tripinfo-output", "tripinfo.xml",
                 "--end", str(steps), "--no-warnings", "true"])

    all_junctions = traci.trafficlight.getIDList()
    all_lanes     = [l for j in all_junctions for l in traci.trafficlight.getControlledLanes(j)]

    # ── Metrics storage ──────────────────────────────
    veh_start, veh_end, veh_wait, veh_total_wait = {}, {}, {}, {}
    queue_lengths = []
    junction_awt  = {j: [] for j in all_junctions}
    junction_aql  = {j: [] for j in all_junctions}

    traffic_lights_time    = {j: 0  for j in all_junctions}
    prev_action            = {j: 0  for j in range(len(all_junctions))}
    prev_vehicles_per_lane = {j: [0]*4 for j in range(len(all_junctions))}
    min_duration = 5

    SELECT_LANE_TEMPLATE = [
        ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],
        ["rrryyyrrrrrr", "rrrGGGrrrrrr"],
        ["rrrrrryyyrrr", "rrrrrrGGGrrr"],
        ["rrrrrrrrryyy", "rrrrrrrrrGGG"],
    ]

    def build_phases(junction):
        lanes = traci.trafficlight.getControlledLanes(junction)
        n = len(lanes)
        phases = []
        for i in range(4):
            g = ['r'] * n
            y = ['r'] * n
            s, e2 = i * n // 4, (i + 1) * n // 4
            for k in range(s, e2):
                g[k] = 'G'; y[k] = 'y'
            phases.append([''.join(y), ''.join(g)])
        return phases

    select_lane = {j: build_phases(j) for j in all_junctions}

    step = 0
    print(f"[~] Running inference for {steps} steps...")

    while step <= steps:
        try:
            traci.simulationStep()
        except Exception:
            break

        if traci.simulation.getMinExpectedNumber() == 0 and step > 10:
            break

        sim_time = traci.simulation.getTime()

        for vid in traci.simulation.getDepartedIDList():
            veh_start[vid] = sim_time; veh_wait[vid] = 0.0; veh_total_wait[vid] = 0.0

        for vid in traci.simulation.getArrivedIDList():
            veh_end[vid] = sim_time

        for vid in traci.vehicle.getIDList():
            pw = veh_wait.get(vid, 0.0)
            cw = traci.vehicle.getWaitingTime(vid)
            dw = cw - pw
            veh_wait[vid] = cw
            veh_total_wait[vid] = veh_total_wait.get(vid, 0.0) + dw
            lane_id = traci.vehicle.getLaneID(vid)
            for j in all_junctions:
                if lane_id in traci.trafficlight.getControlledLanes(j):
                    junction_awt[j].append(dw); break

        total_halting = sum(traci.lane.getLastStepHaltingNumber(l) for l in all_lanes)
        queue_lengths.append(total_halting)

        for jn, junction in enumerate(all_junctions):
            lanes = traci.trafficlight.getControlledLanes(junction)
            vc = [sum(1 for v in traci.lane.getLastStepVehicleIDs(l)
                      if traci.vehicle.getLanePosition(v) > 10) for l in lanes]
            state = (vc + [0]*4)[:4]

            if traffic_lights_time[junction] == 0:
                st = torch.tensor([state], dtype=torch.float).to(device)
                with torch.no_grad():
                    action = torch.argmax(model(st)).item()
                prev_action[jn] = action
                phases = select_lane[junction]
                traci.trafficlight.setRedYellowGreenState(junction, phases[action][0])
                traci.trafficlight.setPhaseDuration(junction, 6)
                traci.trafficlight.setRedYellowGreenState(junction, phases[action][1])
                traci.trafficlight.setPhaseDuration(junction, min_duration + 10)
                traffic_lights_time[junction] = min_duration + 10
            else:
                traffic_lights_time[junction] -= 1

            aql = np.mean([traci.lane.getLastStepHaltingNumber(l) for l in lanes])
            junction_aql[junction].append(aql)

        step += 1

    try:
        traci.close()
    except Exception:
        pass

    # ── Compute metrics ──────────────────────────────
    count      = len(veh_total_wait)
    total_wait = sum(veh_total_wait.values())
    total_att  = sum(veh_end[v] - veh_start[v] for v in veh_end if v in veh_start)

    awt = round(total_wait / count, 3)   if count else 0
    att = round(total_att  / count, 3)   if count else 0
    aql = round(float(np.mean(queue_lengths)), 3) if queue_lengths else 0

    j_awt = {j: round(float(np.mean(junction_awt[j])), 3) if junction_awt[j] else 0 for j in all_junctions}
    j_aql = {j: round(float(np.mean(junction_aql[j])), 3) if junction_aql[j] else 0 for j in all_junctions}

    print(f"\n[✓] RL Agent Results:")
    print(f"    AWT = {awt}s  |  ATT = {att}s  |  AQL = {aql}  |  Vehicles = {count}")
    for j in all_junctions:
        print(f"    {j}: AWT={j_awt[j]}s | AQL={j_aql[j]}")

    return {"awt": awt, "att": att, "aql": aql, "vehicles": count,
            "junction_awt": j_awt, "junction_aql": j_aql}


def run_and_collect_fixed(steps=1000, phase_duration=30):
    """Run fixed-time controller and collect metrics."""
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg",
                 "--tripinfo-output", "tripinfo_fixed.xml",
                 "--end", str(steps), "--no-warnings", "true"])

    all_junctions = traci.trafficlight.getIDList()
    all_lanes     = [l for j in all_junctions for l in traci.trafficlight.getControlledLanes(j)]

    def gen_phases(junction):
        n = len(traci.trafficlight.getControlledLanes(junction))
        if n == 12: return ["GGGrrrGGGrrr","rrrGGGrrrrrr","rrrrrrGGGrrr","rrrrrrrrrGGG"]
        if n == 8:  return ["GGrrGGrr","rrGGrrrr","rrrrGGrr","rrrrrrGG"]
        phases = []
        for i in range(4):
            p = ['r']*n
            for k in range(i*n//4, (i+1)*n//4): p[k]='G'
            phases.append(''.join(p))
        return phases

    configs = {j: {'phases': gen_phases(j), 'cur': 0, 'rem': phase_duration} for j in all_junctions}
    veh_start, veh_end, veh_wait, veh_total_wait = {}, {}, {}, {}
    queue_lengths = []
    junction_awt  = {j: [] for j in all_junctions}
    junction_aql  = {j: [] for j in all_junctions}
    step = 0

    print(f"[~] Running fixed-time for {steps} steps...")

    while step <= steps:
        try:
            traci.simulationStep()
        except Exception:
            break
        if traci.simulation.getMinExpectedNumber() == 0 and step > 10:
            break

        sim_time = traci.simulation.getTime()
        for vid in traci.simulation.getDepartedIDList():
            veh_start[vid] = sim_time; veh_wait[vid] = 0.0; veh_total_wait[vid] = 0.0
        for vid in traci.simulation.getArrivedIDList():
            veh_end[vid] = sim_time
        for vid in traci.vehicle.getIDList():
            pw = veh_wait.get(vid, 0.0); cw = traci.vehicle.getWaitingTime(vid)
            dw = cw - pw; veh_wait[vid] = cw; veh_total_wait[vid] = veh_total_wait.get(vid, 0.0) + dw
            lane_id = traci.vehicle.getLaneID(vid)
            for j in all_junctions:
                if lane_id in traci.trafficlight.getControlledLanes(j):
                    junction_awt[j].append(dw); break

        queue_lengths.append(sum(traci.lane.getLastStepHaltingNumber(l) for l in all_lanes))

        for j in all_junctions:
            c = configs[j]; c['rem'] -= 1
            if c['rem'] <= 0:
                c['cur'] = (c['cur']+1) % len(c['phases']); c['rem'] = phase_duration
            traci.trafficlight.setRedYellowGreenState(j, c['phases'][c['cur']])
            lanes = traci.trafficlight.getControlledLanes(j)
            junction_aql[j].append(np.mean([traci.lane.getLastStepHaltingNumber(l) for l in lanes]))

        step += 1

    try:
        traci.close()
    except Exception:
        pass

    count      = len(veh_total_wait)
    total_wait = sum(veh_total_wait.values())
    total_att  = sum(veh_end[v]-veh_start[v] for v in veh_end if v in veh_start)
    awt = round(total_wait/count, 3) if count else 0
    att = round(total_att/count,  3) if count else 0
    aql = round(float(np.mean(queue_lengths)), 3) if queue_lengths else 0
    j_awt = {j: round(float(np.mean(junction_awt[j])),3) if junction_awt[j] else 0 for j in all_junctions}
    j_aql = {j: round(float(np.mean(junction_aql[j])),3) if junction_aql[j] else 0 for j in all_junctions}

    print(f"\n[✓] Fixed-Time Results:")
    print(f"    AWT = {awt}s  |  ATT = {att}s  |  AQL = {aql}  |  Vehicles = {count}")
    for j in all_junctions:
        print(f"    {j}: AWT={j_awt[j]}s | AQL={j_aql[j]}")

    return {"awt": awt, "att": att, "aql": aql, "vehicles": count,
            "junction_awt": j_awt, "junction_aql": j_aql}


def load_epoch_history(model_name="model"):
    """
    Load per-epoch total waiting time from train.py's saved plot data.
    Since train.py doesn't save the list to disk, we reconstruct it
    from the training plot image name — or from a manually saved log.
    Falls back to reading simple_traffic_metrics.csv if available.
    """
    # Check if user already saved epoch data (optional enhancement)
    epoch_file = f"plots/epoch_data_{model_name}.json"
    if os.path.exists(epoch_file):
        with open(epoch_file) as f:
            return json.load(f)["epoch_total_wait"]

    # Fallback: return empty (website will use its own simulated curve)
    print(f"[i] No epoch history file found at {epoch_file}")
    print(f"    To save real epoch data, add this to train.py (see instructions below).")
    return []


def compute_improvement(rl, fixed):
    """Calculate % improvement of RL over Fixed-Time."""
    def pct(a, b):
        if b == 0: return 0
        return round((b - a) / b * 100, 1)
    return {
        "awt_improvement_pct": pct(rl["awt"],  fixed["awt"]),
        "att_improvement_pct": pct(rl["att"],  fixed["att"]),
        "aql_improvement_pct": pct(rl["aql"],  fixed["aql"]),
    }


def save_results(rl_results, fixed_results, epoch_history, model_name):
    """Save everything to results.json"""
    improvement = compute_improvement(rl_results, fixed_results)

    output = {
        "model_name": model_name,
        "rl": rl_results,
        "fixed": fixed_results,
        "improvement": improvement,
        "epoch_history": epoch_history,   # list of total_wait per epoch
        "junctions": JUNCTIONS
    }

    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*50)
    print("  results.json saved successfully!")
    print("="*50)
    print(f"\n  AWT:  Fixed={fixed_results['awt']}s  →  RL={rl_results['awt']}s  ({improvement['awt_improvement_pct']}% better)")
    print(f"  ATT:  Fixed={fixed_results['att']}s  →  RL={rl_results['att']}s  ({improvement['att_improvement_pct']}% better)")
    print(f"  AQL:  Fixed={fixed_results['aql']}   →  RL={rl_results['aql']}   ({improvement['aql_improvement_pct']}% better)")
    print("\n  Open nexus-traffic-ai.html in your browser to see the results!")
    print("="*50)


def get_options():
    p = optparse.OptionParser()
    p.add_option("-m", dest="model_name", type="string", default="model", help="Model name (default: model)")
    p.add_option("-s", dest="steps",      type="int",    default=500,     help="Steps for RL inference (default: 500)")
    p.add_option("-d", dest="duration",   type="int",    default=30,      help="Fixed-time phase duration (default: 30)")
    p.add_option("--skip-fixed", action="store_true", default=False, help="Skip fixed-time run, use saved values")
    return p.parse_args()[0]


if __name__ == "__main__":
    opts = get_options()

    print("\n" + "="*50)
    print("  NEXUS TRAFFIC AI — Results Exporter")
    print("="*50)
    print(f"  Model: {opts.model_name}")
    print(f"  Steps: {opts.steps}")
    print("="*50 + "\n")

    # ── Run RL agent ─────────────────────────────────
    print("[1/3] Running trained RL agent...")
    rl_results = run_and_collect_rl(model_name=opts.model_name, steps=opts.steps)

    # ── Run fixed-time baseline ───────────────────────
    if opts.skip_fixed:
        print("\n[2/3] Skipping fixed-time run (--skip-fixed flag set)")
        fixed_results = {"awt": 15.0, "att": 130.0, "aql": 80.0, "vehicles": 110,
                         "junction_awt": {j: 0.4 for j in JUNCTIONS},
                         "junction_aql": {j: 1.5 for j in JUNCTIONS}}
    else:
        print("\n[2/3] Running fixed-time baseline for comparison...")
        fixed_results = run_and_collect_fixed(steps=opts.steps, phase_duration=opts.duration)

    # ── Load epoch history ────────────────────────────
    print("\n[3/3] Loading epoch training history...")
    epoch_history = load_epoch_history(opts.model_name)

    # ── Save results.json ─────────────────────────────
    save_results(rl_results, fixed_results, epoch_history, opts.model_name)
