from __future__ import absolute_import, print_function

import os
import sys
import optparse
import numpy as np
import pandas as pd

# SUMO tools
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci


def generate_standard_phases(junction, num_lanes):
    """Generate 4-phase traffic light pattern matching actual lane count."""
    if num_lanes == 12:
        return ["GGGrrrGGGrrr", "rrrGGGrrrrrr", "rrrrrrGGGrrr", "rrrrrrrrrGGG"]
    elif num_lanes == 8:
        return ["GGrrGGrr", "rrGGrrrr", "rrrrGGrr", "rrrrrrGG"]
    else:
        # Auto-generate phases for any lane count
        phases = []
        for i in range(4):
            phase = ['r'] * num_lanes
            start = i * num_lanes // 4
            end   = (i + 1) * num_lanes // 4
            for j in range(start, end):
                phase[j] = 'G'
            phases.append(''.join(phase))
        return phases


def run_fixed_time_with_simple_metrics(steps=1000, phase_duration=30):
    """Run fixed-time traffic signal control and collect AWT, ATT, AQL metrics."""

    sumo_binary = checkBinary("sumo-gui")

    # ── FIX: pass --end to override the config's <end value='200'/> ──
    traci.start([
        sumo_binary,
        "-c", "configuration.sumocfg",
        "--tripinfo-output", "tripinfo.xml",
        "--waiting-time-memory", "1000",
        "--end", str(steps),          # <-- lets simulation run for 'steps' seconds
        "--no-warnings", "true",
    ])

    all_junctions = traci.trafficlight.getIDList()
    all_lanes     = [l for j in all_junctions
                     for l in traci.trafficlight.getControlledLanes(j)]

    # Build phase configs per junction
    junction_configs = {}
    for junction in all_junctions:
        num_lanes = len(traci.trafficlight.getControlledLanes(junction))
        junction_configs[junction] = {
            'phases':         generate_standard_phases(junction, num_lanes),
            'current_phase':  0,
            'remaining_time': phase_duration,
        }

    # Metrics
    step = 0
    total_time = 0
    veh_start_time, veh_end_time         = {}, {}
    veh_wait_time,  veh_total_wait_time  = {}, {}
    queue_lengths                         = []
    junction_vehicle_wait  = {j: [] for j in all_junctions}
    junction_queue_history = {j: [] for j in all_junctions}

    print(f"Starting simulation with {len(all_junctions)} junctions")
    print(f"Phase duration : {phase_duration} steps")
    print(f"Total steps    : {steps}")

    while step <= steps:

        # ── FIX: stop cleanly when SUMO finishes all vehicles early ──
        try:
            traci.simulationStep()
        except traci.exceptions.FatalTraCIError:
            print(f"\nSUMO closed at step {step} (all vehicles completed).")
            break

        sim_time = traci.simulation.getTime()

        # Check if simulation is truly done
        if traci.simulation.getMinExpectedNumber() == 0 and step > 10:
            print(f"\nAll vehicles completed at step {step}.")
            break

        # Track departures
        for vid in traci.simulation.getDepartedIDList():
            veh_start_time[vid]      = sim_time
            veh_wait_time[vid]       = 0.0
            veh_total_wait_time[vid] = 0.0

        # Track arrivals
        for vid in traci.simulation.getArrivedIDList():
            veh_end_time[vid] = sim_time

        # Update per-vehicle waiting time
        for vid in traci.vehicle.getIDList():
            prev_wait    = veh_wait_time.get(vid, 0.0)
            current_wait = traci.vehicle.getWaitingTime(vid)
            delta_wait   = current_wait - prev_wait
            veh_wait_time[vid]       = current_wait
            veh_total_wait_time[vid] = veh_total_wait_time.get(vid, 0.0) + delta_wait

            lane_id = traci.vehicle.getLaneID(vid)
            for junction in all_junctions:
                if lane_id in traci.trafficlight.getControlledLanes(junction):
                    junction_vehicle_wait[junction].append(delta_wait)
                    break

        # Overall queue
        total_halting = sum(traci.lane.getLastStepHaltingNumber(l) for l in all_lanes)
        queue_lengths.append(total_halting)
        total_time += total_halting

        # Update traffic lights
        for junction in all_junctions:
            config = junction_configs[junction]
            config['remaining_time'] -= 1

            if config['remaining_time'] <= 0:
                config['current_phase'] = (config['current_phase'] + 1) % len(config['phases'])
                config['remaining_time'] = phase_duration

            traci.trafficlight.setRedYellowGreenState(
                junction, config['phases'][config['current_phase']]
            )

            lanes     = traci.trafficlight.getControlledLanes(junction)
            avg_queue = np.mean([traci.lane.getLastStepHaltingNumber(l) for l in lanes])
            junction_queue_history[junction].append(avg_queue)

        step += 1

    # ── Metrics ──────────────────────────────────────────────────────
    count            = len(veh_total_wait_time)
    total_wait       = sum(veh_total_wait_time.values())
    total_turnaround = sum(
        veh_end_time[v] - veh_start_time[v]
        for v in veh_end_time if v in veh_start_time
    )

    avg_waiting_time  = total_wait / count if count else 0
    avg_turnaround    = total_turnaround / count if count else 0
    avg_queue_length  = np.mean(queue_lengths) if queue_lengths else 0

    print("\n---- PER-JUNCTION PERFORMANCE ----")
    for junction in all_junctions:
        awt = np.mean(junction_vehicle_wait[junction])  if junction_vehicle_wait[junction]  else 0
        aql = np.mean(junction_queue_history[junction]) if junction_queue_history[junction] else 0
        print(f"  Junction {junction}: AWT={awt:.2f}s | AQL={aql:.2f}")

    print("\n---- OVERALL PERFORMANCE ----")
    print(f"  Average Waiting Time  (AWT) : {avg_waiting_time:.2f} s")
    print(f"  Average Turnaround Time (ATT): {avg_turnaround:.2f} s")
    print(f"  Average Queue Length  (AQL) : {avg_queue_length:.2f} vehicles")
    print(f"  Total waiting time          : {total_time:.2f}")
    print(f"  Vehicles tracked            : {count}")
    print("-----------------------------------\n")

    save_simple_metrics(veh_start_time, veh_end_time, veh_total_wait_time)

    try:
        traci.close()
    except Exception:
        pass


def save_simple_metrics(veh_start_time, veh_end_time, veh_total_wait_time):
    """Save per-vehicle metrics to CSV."""
    rows = []
    for vid, wait_time in veh_total_wait_time.items():
        start = veh_start_time.get(vid, 0)
        end   = veh_end_time.get(vid, 0)
        rows.append({
            'vehicle_id':   vid,
            'waiting_time': round(wait_time, 3),
            'travel_time':  round(end - start, 3) if end > 0 else 0,
            'completed':    vid in veh_end_time,
        })
    df = pd.DataFrame(rows)
    df.to_csv('simple_traffic_metrics.csv', index=False)
    print("  Metrics saved → simple_traffic_metrics.csv")


def get_options():
    p = optparse.OptionParser()
    p.add_option("-s", dest='steps',    type='int', default=1000, help="Simulation steps")
    p.add_option("-d", dest='duration', type='int', default=30,   help="Phase duration")
    return p.parse_args()[0]


if __name__ == "__main__":
    options = get_options()
    run_fixed_time_with_simple_metrics(steps=options.steps, phase_duration=options.duration)