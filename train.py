from __future__ import absolute_import, print_function

import os
import sys
import time
import optparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt

# SUMO tools
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'.\n"
             "  Linux/Mac: export SUMO_HOME=/usr/share/sumo\n"
             "  Windows:   set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo")

from sumolib import checkBinary
import traci


# ─────────────────────────────────────────────
#  Neural Network (DQN)
# ─────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.linear1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.linear3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


# ─────────────────────────────────────────────
#  RL Agent
# ─────────────────────────────────────────────
class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, fc1_dims, fc2_dims,
                 batch_size, n_actions, junctions,
                 max_memory_size=100000, epsilon_dec=5e-4, epsilon_end=0.05):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = list(range(n_actions))
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.iter_cntr = 0

        self.Q_eval = Model(lr, input_dims, fc1_dims, fc2_dims, n_actions)

        self.memory = {}
        for j in junctions:
            self.memory[j] = {
                "state_memory":     np.zeros((max_memory_size, input_dims), dtype=np.float32),
                "new_state_memory": np.zeros((max_memory_size, input_dims), dtype=np.float32),
                "reward_memory":    np.zeros(max_memory_size, dtype=np.float32),
                "action_memory":    np.zeros(max_memory_size, dtype=np.int32),
                "terminal_memory":  np.zeros(max_memory_size, dtype=bool),
                "mem_cntr": 0,
            }

    def store_transition(self, state, state_, action, reward, done, junction):
        idx = self.memory[junction]["mem_cntr"] % self.max_mem
        self.memory[junction]["state_memory"][idx] = state
        self.memory[junction]["new_state_memory"][idx] = state_
        self.memory[junction]["reward_memory"][idx] = reward
        self.memory[junction]["action_memory"][idx] = action
        self.memory[junction]["terminal_memory"][idx] = done
        self.memory[junction]["mem_cntr"] += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
            actions = self.Q_eval(state)
            return torch.argmax(actions).item()
        return np.random.choice(self.action_space)

    def reset(self, junction_numbers):
        for j in junction_numbers:
            self.memory[j]["mem_cntr"] = 0

    def save(self, model_name):
        os.makedirs("models", exist_ok=True)
        torch.save(self.Q_eval.state_dict(), f"models/{model_name}.bin")
        print(f"  Model saved → models/{model_name}.bin")

    def learn(self, junction):
        self.Q_eval.optimizer.zero_grad()
        batch = np.arange(self.memory[junction]["mem_cntr"], dtype=np.int32)

        state_batch     = torch.tensor(self.memory[junction]["state_memory"][batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.memory[junction]["new_state_memory"][batch]).to(self.Q_eval.device)
        reward_batch    = torch.tensor(self.memory[junction]["reward_memory"][batch]).to(self.Q_eval.device)
        terminal_batch  = torch.tensor(self.memory[junction]["terminal_memory"][batch]).to(self.Q_eval.device)
        action_batch    = self.memory[junction]["action_memory"][batch]

        q_eval = self.Q_eval(state_batch)[batch, action_batch]
        q_next = self.Q_eval(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def get_vehicle_numbers(lanes):
    counts = {}
    for lane in lanes:
        counts[lane] = sum(
            1 for vid in traci.lane.getLastStepVehicleIDs(lane)
            if traci.vehicle.getLanePosition(vid) > 10
        )
    return counts


def get_waiting_time(lanes):
    return sum(traci.lane.getWaitingTime(l) for l in lanes)


def phaseDuration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)


def build_select_lane(junction):
    """Build phase strings that match the exact number of controlled lanes."""
    lanes = traci.trafficlight.getControlledLanes(junction)
    n = len(lanes)
    phases = []
    for i in range(4):
        phase_list = ['r'] * n
        start = i * n // 4
        end   = (i + 1) * n // 4
        for j in range(start, end):
            phase_list[j] = 'G'
        yellow = phase_list.copy()
        for j in range(start, end):
            yellow[j] = 'y'
        phases.append([''.join(yellow), ''.join(phase_list)])
    return phases


# ─────────────────────────────────────────────
#  Main training / testing loop
# ─────────────────────────────────────────────
def run(train=True, model_name="model", epochs=50, steps=500):
    best_time = np.inf
    total_time_list = []

    # Discover junctions with a quick dry run
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg",
                 "--tripinfo-output", "tripinfo.xml"])
    all_junctions   = traci.trafficlight.getIDList()
    junction_numbers = list(range(len(all_junctions)))
    traci.close()

    print(f"Junctions found: {list(all_junctions)}")

    brain = Agent(
        gamma=0.99, epsilon=0.0 if not train else 1.0,
        lr=0.1, input_dims=4, fc1_dims=256, fc2_dims=256,
        batch_size=1024, n_actions=4, junctions=junction_numbers,
    )

    if not train:
        brain.Q_eval.load_state_dict(
            torch.load(f"models/{model_name}.bin", map_location=brain.Q_eval.device)
        )
        print(f"Loaded model: models/{model_name}.bin")

    for epoch in range(epochs):
        sumo_cmd = [
            checkBinary("sumo" if train else "sumo-gui"),
            "-c", "configuration.sumocfg",
            "--tripinfo-output", "tripinfo.xml"
        ]
        traci.start(sumo_cmd)
        print(f"\nEpoch {epoch+1}/{epochs}  (ε={brain.epsilon:.4f})")

        # Build phase strings per junction (avoids hardcoded lane count)
        select_lane = {j: build_select_lane(j) for j in all_junctions}

        step = 0
        total_time = 0
        min_duration = 5
        traffic_lights_time   = {j: 0   for j in all_junctions}
        prev_action           = {j: 0   for j in junction_numbers}
        prev_vehicles_per_lane = {j: [0]*4 for j in junction_numbers}
        all_lanes = [l for j in all_junctions for l in traci.trafficlight.getControlledLanes(j)]

        # Metrics
        veh_start_time, veh_end_time         = {}, {}
        veh_wait_time, veh_total_wait_time   = {}, {}
        queue_lengths                         = []
        junction_vehicle_wait  = {j: [] for j in all_junctions}
        junction_queue_history = {j: [] for j in all_junctions}

        while step <= steps:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            for vid in traci.simulation.getDepartedIDList():
                veh_start_time[vid]      = sim_time
                veh_wait_time[vid]       = 0.0
                veh_total_wait_time[vid] = 0.0

            for vid in traci.simulation.getArrivedIDList():
                veh_end_time[vid] = sim_time

            for vid in traci.vehicle.getIDList():
                prev_wait    = veh_wait_time.get(vid, 0.0)
                current_wait = traci.vehicle.getWaitingTime(vid)
                delta_wait   = current_wait - prev_wait
                veh_wait_time[vid]       = current_wait
                veh_total_wait_time[vid] = veh_total_wait_time.get(vid, 0.0) + delta_wait

                lane_id = traci.vehicle.getLaneID(vid)
                for j in all_junctions:
                    if lane_id in traci.trafficlight.getControlledLanes(j):
                        junction_vehicle_wait[j].append(delta_wait)
                        break

            total_halting = sum(traci.lane.getLastStepHaltingNumber(l) for l in all_lanes)
            queue_lengths.append(total_halting)

            for junction_number, junction in enumerate(all_junctions):
                lanes        = traci.trafficlight.getControlledLanes(junction)
                waiting_time = get_waiting_time(lanes)
                total_time  += waiting_time

                if traffic_lights_time[junction] == 0:
                    vehicles_per_lane = get_vehicle_numbers(lanes)
                    reward  = -1 * waiting_time
                    state_  = list(vehicles_per_lane.values())[:4]  # use first 4 lanes as state
                    state_  = (state_ + [0]*4)[:4]                   # pad / truncate to 4
                    state   = prev_vehicles_per_lane[junction_number]
                    prev_vehicles_per_lane[junction_number] = state_

                    brain.store_transition(state, state_, prev_action[junction_number],
                                           reward, (step == steps), junction_number)

                    lane_choice = brain.choose_action(state_)
                    prev_action[junction_number] = lane_choice

                    phases = select_lane[junction]
                    phaseDuration(junction, 6,                      phases[lane_choice][0])
                    phaseDuration(junction, min_duration + 10,      phases[lane_choice][1])

                    traffic_lights_time[junction] = min_duration + 10
                    if train:
                        brain.learn(junction_number)
                else:
                    traffic_lights_time[junction] -= 1

                avg_queue = np.mean([traci.lane.getLastStepHaltingNumber(l) for l in lanes])
                junction_queue_history[junction].append(avg_queue)

            step += 1

        # ── Metrics ──────────────────────────────────────────
        count = len(veh_total_wait_time)
        total_wait      = sum(veh_total_wait_time.values())
        total_turnaround = sum(
            veh_end_time[v] - veh_start_time[v]
            for v in veh_end_time if v in veh_start_time
        )
        avg_waiting_time  = total_wait / count if count else 0
        avg_turnaround    = total_turnaround / count if count else 0
        avg_queue_length  = np.mean(queue_lengths) if queue_lengths else 0

        print("\n---- PER-JUNCTION PERFORMANCE ----")
        for j in all_junctions:
            awt = np.mean(junction_vehicle_wait[j])  if junction_vehicle_wait[j]  else 0
            aql = np.mean(junction_queue_history[j]) if junction_queue_history[j] else 0
            print(f"  Junction {j}: AWT={awt:.2f}s | AQL={aql:.2f}")

        print("\n---- OVERALL PERFORMANCE ----")
        print(f"  AWT : {avg_waiting_time:.2f} s")
        print(f"  ATT : {avg_turnaround:.2f} s")
        print(f"  AQL : {avg_queue_length:.2f} vehicles")
        print(f"  Total waiting time: {total_time:.2f}")
        print("-----------------------------------")

        total_time_list.append(total_time)
        if total_time < best_time:
            best_time = total_time
            if train:
                brain.save(model_name)

        traci.close()
        sys.stdout.flush()

        if not train:
            break

    # ── Plot training curve ───────────────────────────────
    if train:
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(9, 5))
        plt.plot(range(1, len(total_time_list)+1), total_time_list, marker='o', linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Total Waiting Time")
        plt.title("DQN Training: Total Waiting Time vs Epoch")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plots/time_vs_epoch_{model_name}.png", dpi=150)
        print(f"\nTraining plot saved → plots/time_vs_epoch_{model_name}.png")

        # ── Save epoch data for website ──────────────────
        import json
        os.makedirs("plots", exist_ok=True)
        epoch_data = {"epoch_total_wait": total_time_list}
        with open(f"plots/epoch_data_{model_name}.json", "w") as f:
            json.dump(epoch_data, f)
        print(f"Epoch data saved → plots/epoch_data_{model_name}.json")


# ─────────────────────────────────────────────
#  CLI options
# ─────────────────────────────────────────────
def get_options():
    p = optparse.OptionParser()
    p.add_option("-m", dest="model_name", type="string", default="model",  help="Model name")
    p.add_option("--train", action="store_true", default=False,             help="Train mode")
    p.add_option("-e", dest="epochs", type="int",  default=50,              help="Epochs")
    p.add_option("-s", dest="steps",  type="int",  default=500,             help="Steps per epoch")
    return p.parse_args()[0]


if __name__ == "__main__":
    opts = get_options()
    run(train=opts.train, model_name=opts.model_name,
        epochs=opts.epochs, steps=opts.steps)
