"""
docker_run.py
─────────────────────────────────────────────────────
Interactive menu for running the Traffic AI project.
Your friend just runs:  docker compose up
And this menu appears automatically.
─────────────────────────────────────────────────────
"""

import os
import sys
import subprocess
import time

# ── Colors for terminal output ────────────────────────
class C:
    NEON  = '\033[96m'
    GREEN = '\033[92m'
    WARN  = '\033[93m'
    RED   = '\033[91m'
    DIM   = '\033[90m'
    BOLD  = '\033[1m'
    END   = '\033[0m'

def banner():
    print(f"""
{C.NEON}{C.BOLD}
╔══════════════════════════════════════════════════════╗
║         NEXUS TRAFFIC AI — Control Panel             ║
║      Dynamic Traffic Light Management System         ║
╚══════════════════════════════════════════════════════╝
{C.END}
{C.DIM}  Junctions: 5  |  Algorithm: Deep Q-Network  |  SUMO: headless{C.END}
""")

def menu():
    print(f"{C.NEON}  What do you want to do?{C.END}\n")
    print(f"  {C.BOLD}[1]{C.END}  Train the AI agent          {C.DIM}(~15 mins, 50 epochs){C.END}")
    print(f"  {C.BOLD}[2]{C.END}  Run fixed-time baseline     {C.DIM}(comparison controller){C.END}")
    print(f"  {C.BOLD}[3]{C.END}  Export results for website  {C.DIM}(generates results.json){C.END}")
    print(f"  {C.BOLD}[4]{C.END}  Generate plots              {C.DIM}(bar charts → plots/){C.END}")
    print(f"  {C.BOLD}[5]{C.END}  Start website server        {C.DIM}(open localhost:8000){C.END}")
    print(f"  {C.BOLD}[6]{C.END}  Run full pipeline           {C.DIM}(train → export → serve){C.END}")
    print(f"  {C.BOLD}[q]{C.END}  Quit\n")

def run(cmd, desc):
    print(f"\n{C.NEON}▶ {desc}{C.END}")
    print(f"{C.DIM}  $ {cmd}{C.END}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        print(f"\n{C.GREEN}  ✓ Done!{C.END}\n")
    else:
        print(f"\n{C.RED}  ✗ Error (code {result.returncode}){C.END}\n")
    return result.returncode

def check_model_exists():
    if not os.path.exists("models/model.bin"):
        print(f"\n{C.WARN}  ⚠ No trained model found at models/model.bin{C.END}")
        print(f"  {C.DIM}Run option [1] to train first.{C.END}\n")
        return False
    return True

def check_maps():
    if not os.path.exists("maps/city1.net.xml"):
        print(f"\n{C.RED}  ✗ Map files not found in maps/")
        print(f"  Make sure city1.net.xml and city1.rou.xml are in the maps/ folder.{C.END}\n")
        return False
    return True

def serve_website():
    print(f"\n{C.NEON}▶ Starting website server...{C.END}")
    print(f"\n{C.GREEN}  ✓ Website running!{C.END}")
    print(f"\n  {C.BOLD}Open your browser and go to:{C.END}")
    print(f"  {C.NEON}  http://localhost:8000/nexus-traffic-ai.html{C.END}\n")
    print(f"  {C.DIM}Press Ctrl+C to stop the server.{C.END}\n")
    os.system("python3 -m http.server 8000")

def run_full_pipeline():
    print(f"\n{C.NEON}{C.BOLD}Running full pipeline...{C.END}")
    print(f"{C.DIM}  Step 1: Train  →  Step 2: Fixed-Time  →  Step 3: Export  →  Step 4: Serve{C.END}\n")

    if not check_maps():
        return

    steps = [
        ("python3 train.py --train -e 50 -s 1000",  "Training DQN agent (50 epochs)..."),
        ("python3 train_fixed.py -s 1000 -d 30",    "Running fixed-time baseline..."),
        ("python3 export_results.py",                "Exporting results to results.json..."),
        ("python3 plots_script.py",                  "Generating plots..."),
    ]

    for cmd, desc in steps:
        code = run(cmd, desc)
        if code != 0:
            print(f"{C.RED}  Pipeline stopped due to error. Fix the issue and retry.{C.END}\n")
            return

    serve_website()

def main():
    banner()

    if not check_maps():
        print(f"{C.RED}  Cannot continue without map files. Exiting.{C.END}")
        sys.exit(1)

    while True:
        menu()
        try:
            choice = input(f"  {C.BOLD}Enter choice: {C.END}").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{C.DIM}  Goodbye!{C.END}\n")
            break

        if choice == '1':
            epochs = input(f"  Epochs? {C.DIM}(default 50){C.END}: ").strip() or "50"
            steps  = input(f"  Steps?  {C.DIM}(default 1000){C.END}: ").strip() or "1000"
            run(f"python3 train.py --train -e {epochs} -s {steps}", "Training DQN agent...")

        elif choice == '2':
            run("python3 train_fixed.py -s 1000 -d 30", "Running fixed-time baseline...")

        elif choice == '3':
            if check_model_exists():
                run("python3 export_results.py", "Exporting results...")

        elif choice == '4':
            run("python3 plots_script.py", "Generating plots...")

        elif choice == '5':
            serve_website()

        elif choice == '6':
            run_full_pipeline()

        elif choice in ('q', 'quit', 'exit'):
            print(f"\n{C.DIM}  Goodbye!{C.END}\n")
            break

        else:
            print(f"\n{C.WARN}  Invalid choice. Enter 1–6 or q.{C.END}\n")

if __name__ == "__main__":
    main()
