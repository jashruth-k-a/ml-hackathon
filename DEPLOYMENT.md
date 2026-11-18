# Run PathSnap on two laptops

Extract the ZIP and open PowerShell in the pathsnap folder. Both laptops need Python 3.11. On Laptop 1 create the environment and install requirements.txt. Keep the organiser's input folders beside api_server.py on Laptop 1. The ZIP includes the fitted model and decoder settings. The UI laptop needs only ui_server.py, interface.html, FastAPI and Uvicorn.

Laptop 1, development rehearsal:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:PATHSNAP_MODE = "student"
.\.venv\Scripts\python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --access-log
```

Laptop 2, on the same network (replace LAPTOP1_IP with its actual LAN IPv4 address):

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install fastapi==0.163.2 uvicorn==0.54.0
$env:PATHSNAP_API = "http://LAPTOP1_IP:8000"
.\.venv\Scripts\python -m uvicorn ui_server:app --host 127.0.0.1 --port 8080
```

Open http://127.0.0.1:8080 on Laptop 2. Select a trip and reconstruct it. The first laptop should show the HTTP POST in its terminal. If the UI cannot list trips, check Laptop 1's address, port and private-network firewall access. At final evaluation only, restart the API with PATHSNAP_MODE set to evaluation and the evaluator_test folder on Laptop 1. The API lists actual directories and runs inference per request; it reads the route for scoring only after the decoder returns. The response contains GPS, predicted arcs, geometry, and aggregate scores.

The supplied browser evidence uses ten student trips and two processes on one host. Capture the actual receiving laptop's terminal and the UI during the LAN demonstration. To repeat the automated Windows browser test, install the Node playwright package, use an installed Chrome browser, and run node verify_ui.cjs. Start from the repository root with the project environment named .venv.
