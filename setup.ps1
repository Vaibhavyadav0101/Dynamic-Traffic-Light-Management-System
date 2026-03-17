# ============================================================
# SUMO Traffic RL Project - Windows Setup Script
# Run this from your project root folder in PowerShell:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#   .\setup.ps1
# ============================================================

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " SUMO Traffic RL Project - Windows Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# ---------- Step 1: Install SUMO ----------
Write-Host ""
Write-Host "[1/4] Checking SUMO..." -ForegroundColor Yellow

$sumoDefault = "C:\Program Files (x86)\Eclipse\Sumo"
$sumoAlt     = "C:\Sumo"

if (Test-Path "$sumoDefault\bin\sumo.exe") {
    $SUMO_HOME = $sumoDefault
} elseif (Test-Path "$sumoAlt\bin\sumo.exe") {
    $SUMO_HOME = $sumoAlt
} else {
    Write-Host ""
    Write-Host "  SUMO not found! Please install it manually:" -ForegroundColor Red
    Write-Host "  1. Go to: https://sumo.dlr.de/docs/Downloads.php" -ForegroundColor White
    Write-Host "  2. Download the Windows installer (.msi)" -ForegroundColor White
    Write-Host "  3. Install it (default path: C:\Program Files (x86)\Eclipse\Sumo)" -ForegroundColor White
    Write-Host "  4. Re-run this script after installation" -ForegroundColor White
    Write-Host ""
    # Try winget as a fallback
    Write-Host "  Attempting install via winget..." -ForegroundColor Yellow
    winget install --id Eclipse.SUMO -e --silent
    if (Test-Path "$sumoDefault\bin\sumo.exe") {
        $SUMO_HOME = $sumoDefault
        Write-Host "  SUMO installed via winget!" -ForegroundColor Green
    } else {
        Write-Host "  winget install failed. Please install SUMO manually from the link above." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host "  SUMO found at: $SUMO_HOME" -ForegroundColor Green

# Set SUMO_HOME permanently for the current user
[System.Environment]::SetEnvironmentVariable("SUMO_HOME", $SUMO_HOME, "User")
$env:SUMO_HOME = $SUMO_HOME

# Add SUMO bin and tools to PATH
$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
$sumo_bin    = "$SUMO_HOME\bin"
$sumo_tools  = "$SUMO_HOME\tools"

if ($currentPath -notlike "*$sumo_bin*") {
    [System.Environment]::SetEnvironmentVariable("PATH", "$currentPath;$sumo_bin;$sumo_tools", "User")
    $env:PATH += ";$sumo_bin;$sumo_tools"
    Write-Host "  Added SUMO to PATH" -ForegroundColor Green
} else {
    Write-Host "  SUMO already in PATH" -ForegroundColor Green
}

# Set PYTHONPATH so 'import traci' works
$currentPP = [System.Environment]::GetEnvironmentVariable("PYTHONPATH", "User")
if ($null -eq $currentPP -or $currentPP -notlike "*$sumo_tools*") {
    [System.Environment]::SetEnvironmentVariable("PYTHONPATH", "$sumo_tools;$currentPP", "User")
    $env:PYTHONPATH = "$sumo_tools;$currentPP"
    Write-Host "  SUMO tools added to PYTHONPATH" -ForegroundColor Green
}

# ---------- Step 2: Install Python packages ----------
Write-Host ""
Write-Host "[2/4] Installing Python packages..." -ForegroundColor Yellow

pip install torch numpy matplotlib pandas
if ($LASTEXITCODE -ne 0) {
    Write-Host "  pip failed. Make sure Python is installed and in PATH." -ForegroundColor Red
    Write-Host "  Download Python from: https://www.python.org/downloads/" -ForegroundColor White
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "  Python packages installed" -ForegroundColor Green

# ---------- Step 3: Create folder structure ----------
Write-Host ""
Write-Host "[3/4] Creating project folders..." -ForegroundColor Yellow

$folders = @("maps", "maps_images", "models", "plots")
foreach ($folder in $folders) {
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
        Write-Host "  Created: $folder\" -ForegroundColor Green
    } else {
        Write-Host "  Already exists: $folder\" -ForegroundColor Gray
    }
}

# ---------- Step 4: Verify ----------
Write-Host ""
Write-Host "[4/4] Verifying installation..." -ForegroundColor Yellow

try {
    $v = & "$SUMO_HOME\bin\sumo.exe" --version 2>&1 | Select-Object -First 1
    Write-Host "  sumo    : OK  ($v)" -ForegroundColor Green
} catch {
    Write-Host "  sumo    : FAILED" -ForegroundColor Red
}

$checks = @("traci", "torch", "numpy", "matplotlib", "pandas")
foreach ($pkg in $checks) {
    $result = python -c "import $pkg; print($pkg.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  $pkg : OK  (v$result)" -ForegroundColor Green
    } else {
        Write-Host "  $pkg : FAILED - $result" -ForegroundColor Red
    }
}

# ---------- Done ----------
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host " IMPORTANT: Close and reopen PowerShell" -ForegroundColor Yellow
Write-Host " so the new environment variables take effect." -ForegroundColor Yellow
Write-Host ""
Write-Host " Then run your project:" -ForegroundColor White
Write-Host ""
Write-Host "   # Train the RL agent:" -ForegroundColor Gray
Write-Host "   python train.py --train -e 50 -s 500" -ForegroundColor White
Write-Host ""
Write-Host "   # Test the trained model (opens sumo-gui):" -ForegroundColor Gray
Write-Host "   python train.py -m model" -ForegroundColor White
Write-Host ""
Write-Host "   # Run fixed-time baseline:" -ForegroundColor Gray
Write-Host "   python train_fixed.py -s 1000 -d 30" -ForegroundColor White
Write-Host ""
Write-Host "   # Generate plots:" -ForegroundColor Gray
Write-Host "   python plots_script.py" -ForegroundColor White
Write-Host "=========================================" -ForegroundColor Cyan
