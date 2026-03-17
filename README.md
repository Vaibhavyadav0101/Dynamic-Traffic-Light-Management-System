# 🚦 Dynamic Traffic Light Management System

A Deep Q-Network (DQN) reinforcement learning agent that controls traffic signals across 5 city junctions in real time — reducing average vehicle waiting time by **~37%** compared to traditional fixed-time controllers.

---

## 📸 Demo

Open `nexus-traffic-ai.html` in your browser (after running a local server) to see the interactive dashboard with live simulation, training curves, and metric comparisons.

---

## 🧠 How It Works

The AI agent observes the number of vehicles waiting in each lane at every junction and picks the traffic phase (which direction gets green) that it predicts will minimize waiting time. It learns this through trial and error over 50 training epochs using the Bellman equation and experience replay.

**Full explanation:** see [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) or the project website.

### Core Loop (every simulation step)
```
1. Read vehicle counts per lane  →  State [4 numbers]
2. DQN picks best phase          →  Action (0–3)
3. Apply phase to SUMO via TraCI →  Yellow (6s) + Green (15s)
4. Measure waiting time          →  Reward = −waiting_time
5. Store experience, update net  →  Learning
```

---

## 📁 Project Structure

```
├── train.py                  # DQN agent — training & testing
├── train_fixed.py            # Fixed-time baseline controller
├── export_results.py         # Exports real metrics → results.json for website
├── plots_script.py           # Generates comparison bar charts
├── nexus-traffic-ai.html     # Interactive project website (single file)
├── configuration.sumocfg     # SUMO simulation config
├── sumo-gui_settings.xml     # SUMO GUI display settings
├── requirements.txt          # Python dependencies
├── maps/                     # Road network & vehicle route files
│   ├── city1.net.xml         # Road layout (5 junctions)
│   └── city1.rou.xml         # Vehicle routes & schedules
├── models/                   # Saved model weights (.bin)
├── plots/                    # Output charts & epoch data
└── maps_images/              # Map screenshots
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.11+
- [Eclipse SUMO](https://sumo.dlr.de/docs/Downloads.php) (with `SUMO_HOME` set)
- Windows / Linux / Mac

### Install dependencies
```bash
pip install torch numpy matplotlib pandas
```

### Set SUMO_HOME (Windows PowerShell)
```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PYTHONPATH = "$env:SUMO_HOME\tools"
```

### Set SUMO_HOME (Linux/Mac)
```bash
export SUMO_HOME=/usr/share/sumo
export PYTHONPATH=$SUMO_HOME/tools:$PYTHONPATH
```

---

## 🚀 Running the Project

### 1 — Train the DQN agent
```bash
python train.py --train -e 50 -s 1000
```
Saves best model to `models/model.bin` and epoch data to `plots/epoch_data_model.json`.

### 2 — Test the trained model (opens SUMO GUI)
```bash
python train.py -m model
```

### 3 — Run fixed-time baseline for comparison
```bash
python train_fixed.py -s 1000 -d 30
```

### 4 — Export real results for the website
```bash
python export_results.py
```
Generates `results.json` with real AWT, ATT, AQL metrics and improvement percentages.

### 5 — View the project website
```bash
python -m http.server 8000
```
Open **http://localhost:8000/nexus-traffic-ai.html**

---

## 📊 Results

| Metric | Fixed-Time | DQN Agent | Improvement |
|--------|-----------|-----------|-------------|
| Avg Waiting Time (AWT) | ~15.0s | ~9.4s | ▼ 37% |
| Avg Queue Length (AQL) | ~80 vehicles | ~57.8 | ▼ 28% |
| Avg Trip Time (ATT) | ~130s | ~96.9s | ▼ 25% |
| Stuck vehicles | 4–6 / run | ~1 / run | ▼ 80% |

---

## 🏗️ Architecture

### Neural Network (DQN)
```
Input(4) → FC(256, ReLU) → FC(256, ReLU) → Output(4)
```
- **Optimizer:** Adam (lr=0.1)
- **Loss:** MSE
- **Discount factor γ:** 0.99
- **Exploration:** ε-greedy (1.0 → 0.05)

### State Space
4 values per junction — vehicle count per lane (position > 10m from stop line)

### Action Space
4 discrete phases per junction (N-S straight, N-S left, E-W straight, E-W left)

### Reward Function
`reward = −1 × total_waiting_time` across all controlled lanes

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| PyTorch 2.x | Neural network & training |
| Eclipse SUMO | Traffic simulation engine |
| TraCI API | Python ↔ SUMO real-time control |
| NumPy / Pandas | Data processing |
| Matplotlib | Training plots |

---

## 📄 License

MIT License — free to use, modify, and distribute.
