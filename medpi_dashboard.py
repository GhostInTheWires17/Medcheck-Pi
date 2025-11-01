r"""
Standalone Medcheck Pi Dashboard (external controller)

This small Flask app serves a page with two big buttons (O and C). When a button
is clicked the server forwards the choice to the Medcheck Pi instance's
`/set` endpoint (the same endpoint used by the in-process dashboard in
`medpi.py`).

Usage:

1. Install dependencies (on the machine that will host the dashboard):

  py -3.11 -m pip install flask requests

2. Configure the Medcheck Pi host (where medpi.py is running). By default
  the dashboard will forward to http://localhost:5000; set the
  environment variable MEDPI_HOST to override.

  In PowerShell:

  $env:MEDPI_HOST = "192.168.1.50:5000"

3. Run the dashboard server:

  py -3.11 "c:\\Users\\Sai Kumar Santhosh\\Downloads\\medpi_dashboard.py"

4. Open a browser to http://localhost:8000 and click O or C. The dashboard will
  forward the selection to the Medcheck Pi and show the response.

Notes:
- This is intentionally minimal: the dashboard forwards server-side so you
  don't run into CORS issues in the browser. It works even if the Medcheck Pi
  device is on a different host.
- If you already run the embedded dashboard inside `medpi.py`, you don't need
  this external dashboard — but an external dashboard lets you control the PI
  from another machine without opening ports on the PI for the web UI.

"""

from flask import Flask, render_template_string, request, redirect, url_for, flash
import os
import requests

app = Flask(__name__)
app.secret_key = os.environ.get('DASHBOARD_SECRET', 'medpi-dashboard-secret')

MEDPI_HOST = os.environ.get('MEDPI_HOST', 'localhost:5000')
MEDPI_SET_URL = f"http://{MEDPI_HOST}/set"

HTML = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Medcheck Pi Dashboard</title>
  <style>
    body { display:flex; height:100vh; margin:0; align-items:center; justify-content:space-around; background:#111; }
    button.big { width:40vw; height:70vh; font-size:10vw; background:#000; color:#fff; border-radius:12px; border:3px solid #444; }
    .msg { position:fixed; top:8px; left:8px; padding:8px 12px; background:#222; color:#fff; border-radius:6px; border:1px solid #333; }
  </style>
</head>
<body>
  {% if msg %}<div class="msg">{{ msg }}</div>{% endif %}
  <form method="post" action="/press">
    <input type="hidden" name="val" value="o">
    <button class="big" type="submit">O</button>
  </form>
  <form method="post" action="/press">
    <input type="hidden" name="val" value="c">
    <button class="big" type="submit">C</button>
  </form>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    msg = request.args.get('msg')
    return render_template_string(HTML, msg=msg)

@app.route('/press', methods=['POST'])
def press():
    val = request.form.get('val')
    if val not in ('o', 'c'):
        return redirect(url_for('index', msg='Invalid value'))

    try:
        # Forward to Medcheck Pi set endpoint
        resp = requests.post(MEDPI_SET_URL, json={'val': val}, timeout=5)
        if resp.status_code == 200:
            return redirect(url_for('index', msg=f"Sent '{val}' to Medcheck Pi"))
        else:
            return redirect(url_for('index', msg=f"Medcheck Pi responded {resp.status_code}"))
    except Exception as e:
        return redirect(url_for('index', msg=f"Error: {e}"))

if __name__ == '__main__':
    host = os.environ.get('DASHBOARD_HOST', '0.0.0.0')
    port = int(os.environ.get('DASHBOARD_PORT', 8000))
    print(f"Starting Medcheck Pi Dashboard. Forwarding to {MEDPI_SET_URL}")
    app.run(host=host, port=port, debug=False)
