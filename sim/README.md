# Windows AutoDRIVE RoboRacer Visual Simulator

This folder is for the Windows-native AutoDRIVE RoboRacer Simulator only. Keep ROS 2 Humble, Docker, colcon, Python RL, training, and policy serving in WSL.

## Installed Asset

- Official source: AutoDRIVE RoboRacer Sim Racing League ICRA 2026 release
- Competition page: <https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-icra-2026/>
- Release page: <https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing/releases/tag/2026-icra>
- Direct asset URL: <https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing/releases/download/2026-icra/autodrive_simulator_practice_windows.zip>
- Technical guide: <https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/>
- Release tag: `2026-icra`
- Asset: `autodrive_simulator_practice_windows.zip`
- Downloaded to: `sim/downloads/autodrive_simulator_practice_windows.zip`
- Extracted app: `sim/windows_practice/autodrive_simulator/AutoDRIVE Simulator.exe`
- SHA256: `2D3E5CC7073009D8F28F41B2266C129DD64210688C70228A6811B844EE48A751`

The downloaded and extracted simulator binaries are ignored by git. The small runbook and scripts in this folder are intended to be kept.

## Normal Workflow

Use two WSL terminals and one Windows PowerShell terminal.

### 1. Start the WSL API/devkit

In WSL:

```bash
cd /home/ppwhi/F1Tenth/autodrive_simulator/autodrive-roboracer-rl
make build
make api
```

Leave this terminal running. The API container is the server; the Windows simulator is the client.

For a saved SB3 policy, use this shape instead:

```bash
cd /home/ppwhi/F1Tenth/autodrive_simulator/autodrive-roboracer-rl
RUN_MODE=policy POLICY_TYPE=sb3 MODEL_PATH=/home/autodrive_devkit/runs/<path-to-model> make api
```

### 2. Find the WSL IP

Current simulator bridge endpoint:

```text
IP Address: 172.19.30.197
Port:       4567
```

In WSL:

```bash
hostname -I | awk '{print $1}'
```

Or from Windows PowerShell. Use the `.cmd` wrapper if your PowerShell execution policy blocks unsigned `.ps1` scripts:

```powershell
cd C:\RewardFunction
$ip = .\sim\scripts\get-wsl-ip.cmd
$ip
```

Equivalent one-off PowerShell form:

```powershell
$ip = powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\sim\scripts\get-wsl-ip.ps1
```

WSL2 IP addresses can change after WSL restarts, so refresh this before each simulator session.

### 3. Check the bridge port from Windows

After `make api` is running in WSL, run this in Windows PowerShell:

```powershell
cd C:\RewardFunction
.\sim\scripts\test-api-port.cmd -WslIp 172.19.30.197 -Port 4567
```

Expected result: `OK: port 4567 is reachable`.

### 4. Launch the Windows simulator

In Windows PowerShell:

```powershell
cd C:\RewardFunction
.\sim\scripts\launch-practice-sim.cmd
```

If Windows Security asks about network access for AutoDRIVE Simulator, allow Private networks.

### 5. Configure the simulator UI

In the simulator menu panel:

1. Set `IP Address` to `172.19.30.197`.
2. Set `Port Number` to `4567`.
3. Click the connection button to enable the simulator-devkit bridge.
4. Confirm the bridge status changes to `Connected`.
5. Click the driving mode button until the mode is `Autonomous`.
6. Optionally set graphics quality to `High` or `Ultra` for visual checking.
7. Reset the scene if the vehicle was already in a bad state.

The bridge connection button is the simulator side of "connect". The API/devkit must already be running in WSL before clicking it.

This ICRA 2026 RoboRacer simulator build does not expose a runtime track selector in the UI. Track/environment selection is tied to the downloaded simulator build, for example `explore`, `practice`, or `compete`, rather than a dropdown inside the running simulator. If a coordination file says `Berlin Track`, that is a readiness label for the WSL automation unless you also have a separate simulator build from the organizers that contains Berlin.

## Verify from WSL

Open a second WSL terminal while `make api` is still running:

```bash
cd /home/ppwhi/F1Tenth/autodrive_simulator/autodrive-roboracer-rl
make smoke
```

For the exact ROS checks:

```bash
docker exec -it autodrive_roboracer_api bash
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
ros2 topic list
ros2 run autodrive_rl_stack topic_probe
```

You should see `/autodrive/...` topics from the simulator and the policy command topics, including throttle and steering command topics.

## Troubleshooting

- If the simulator stays disconnected, re-run WSL IP discovery and update the simulator IP field.
- If PowerShell says a `.ps1` file is not digitally signed, use the matching `.cmd` wrapper in `sim\scripts`; it runs the same script with process-local execution policy bypass.
- If `test-api-port.ps1` cannot reach port `4567`, confirm `make api` is still running and the Docker container name is `autodrive_roboracer_api`.
- If the simulator connects but the car does not move, confirm the simulator vehicle mode is `Autonomous`, not `Manual`.
- Do not debug the pink WSL Docker Unity screen for this workflow. The Windows-native simulator is the visual renderer.
- Use Windows graphics quality for visual inspection only; keep training and policy serving inside WSL.

## Fully Headless WSL Workflow

Use this when you do not need the Windows visual simulator. The simulator and API both run in WSL/Docker, with Unity rendering disabled.

Start clean:

```bash
cd /home/ppwhi/F1Tenth/autodrive_simulator/autodrive-roboracer-rl
make stop
```

Start the official simulator headlessly. The explicit `-ip` and `-port` arguments matter:

```bash
SIM_LAUNCH_ARGS='-batchmode -nographics -ip 127.0.0.1 -port 4567' make sim-official-headless
```

Start the API/policy stack headlessly:

```bash
RUN_MODE=policy POLICY_TYPE=gap_follower AUTODRIVE_BRINGUP=headless make api
```

For a saved SB3 model:

```bash
RUN_MODE=policy \
POLICY_TYPE=sb3 \
MODEL_PATH=/home/autodrive_devkit/runs/<path-to-model> \
AUTODRIVE_BRINGUP=headless \
make api
```

Verify LiDAR:

```bash
make smoke
docker exec autodrive_roboracer_api bash -lc 'source /opt/ros/humble/setup.bash && source /home/autodrive_devkit/install/setup.bash && timeout 8s ros2 topic hz /autodrive/roboracer_1/lidar'
```

The current shared status file for WSL-side orchestration is:

```bash
/mnt/c/RewardFunction/sim/status.json
```
