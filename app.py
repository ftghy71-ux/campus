# gevent monkey-patching must be first — makes threading/queue/time cooperative
import os
import sys

_gevent_mode = str(os.environ.get('USE_GEVENT_MONKEY', 'auto')).strip().lower()
_try_gevent = (
    _gevent_mode in ('1', 'true', 'yes', 'on')
    or (_gevent_mode in ('', 'auto') and sys.version_info < (3, 14))
)
if _try_gevent:
    try:
        from gevent import monkey
        monkey.patch_all()
    except BaseException as gevent_err:
        print(f"[startup] gevent disabled: {gevent_err!r}", file=sys.stderr)

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response, stream_with_context, has_request_context
from flask_cors import CORS
import json
import copy
import math
import shutil
import uuid
from werkzeug.security import generate_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix
from datetime import datetime, timezone
import threading
import queue
import time
from services.admin_service import (
    load_admins as service_load_admins,
    save_admins as service_save_admins,
    validate_login as service_validate_login,
    validate_pin as service_validate_pin,
    is_gold_username as service_is_gold_username,
    bootstrap_from_legacy as service_bootstrap_from_legacy,
)

def _load_local_env_file():
    """Load .env entries for local development without overriding real env vars."""
    try:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        if not os.path.isfile(env_path):
            return
        with open(env_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                if not key or key in os.environ:
                    continue
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                os.environ[key] = value
    except Exception:
        pass

_load_local_env_file()

# ---------- 1-D Kalman filter for GPS smoothing ----------
class GPSKalman:
    """Lightweight 1-D Kalman per axis. ~2 multiplications per update."""
    __slots__ = ('x', 'p', 'q', 'r')
    def __init__(self, process_noise=0.00001, measurement_noise=0.00005):
        self.x = None   # estimate
        self.p = 1.0     # error covariance
        self.q = process_noise
        self.r = measurement_noise
    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return measurement
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)
        return self.x

_kalman_filters = {}  # bus_id -> {'lat': GPSKalman, 'lng': GPSKalman}

def kalman_smooth(bus_id, lat, lng):
    if bus_id not in _kalman_filters:
        _kalman_filters[bus_id] = {'lat': GPSKalman(), 'lng': GPSKalman()}
    kf = _kalman_filters[bus_id]
    return kf['lat'].update(lat), kf['lng'].update(lng)

# ---------- server-side stop detection ----------
def _haversine_m(lat1, lng1, lat2, lng2):
    """Fast equirectangular distance in meters — accurate at campus scale."""
    d2r = math.pi / 180
    dlat = (lat2 - lat1) * d2r
    dlng = (lng2 - lng1) * d2r
    x = dlng * math.cos((lat1 + lat2) * 0.5 * d2r)
    return 6371000 * math.sqrt(dlat * dlat + x * x)

_AT_STOP_M = 80  # meters — "at stop" threshold
_bus_stop_state = {}  # bus_id -> { 'atStop': str|None, 'nearestStopIdx': int, 'direction': 'up'|'down'|None }

def detect_stop_info(bus_id, lat, lng, route_id):
    """Detect nearest stop, at-stop, direction. Returns dict to merge into broadcast."""
    result = {
        'atStop': None,
        'nearestStopIdx': None,
        'nearestStopName': None,
        'nextStopName': None,
        'direction': None,
        'terminalState': None,
        'routeStopCount': None
    }
    if not route_id:
        return result
    route = get_route_from_locations(route_id)
    if not route or not route.get('waypoints') or len(route['waypoints']) < 2:
        return result
    wps = route['waypoints']
    result['routeStopCount'] = len(wps)
    stops = route.get('stops', [])
    best_idx, best_d = 0, float('inf')
    for i, wp in enumerate(wps):
        d = _haversine_m(lat, lng, wp[0], wp[1])
        if d < best_d:
            best_d = d
            best_idx = i
    result['nearestStopIdx'] = best_idx
    stop_name = stops[best_idx] if best_idx < len(stops) and stops[best_idx] else f'Stop {best_idx + 1}'
    result['nearestStopName'] = stop_name
    # Determine next stop index and name
    next_idx = best_idx + 1 if best_idx + 1 < len(stops) else best_idx
    result['nextStopName'] = stops[next_idx] if next_idx < len(stops) and stops[next_idx] else f'Stop {next_idx + 1}' if next_idx != best_idx else None
    if best_d <= _AT_STOP_M:
        result['atStop'] = stop_name
    # Direction
    prev = _bus_stop_state.get(bus_id, {})
    prev_idx = prev.get('nearestStopIdx')
    direction = prev.get('direction')
    if prev_idx is not None and prev_idx != best_idx:
        direction = 'down' if best_idx > prev_idx else 'up'
    result['direction'] = direction
    # Terminal state mirrors student-side logic so server can enforce cleanup.
    if result['atStop']:
        last_idx = len(wps) - 1
        if (direction == 'down' and best_idx == last_idx) or (direction == 'up' and best_idx == 0):
            result['terminalState'] = 'at_destination'
        elif (direction == 'down' and best_idx == 0) or (direction == 'up' and best_idx == last_idx):
            result['terminalState'] = 'at_start'
    _bus_stop_state[bus_id] = {
        'nearestStopIdx': best_idx,
        'direction': direction,
        'atStop': result['atStop'],
        'nearestStopName': result['nearestStopName'],
        'nextStopName': result['nextStopName'],
        'terminalState': result['terminalState']
    }
    return result

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_requested_data_dir = str(os.environ.get('DATA_DIR', BASE_DIR) or BASE_DIR).strip()
if not os.path.isabs(_requested_data_dir):
    _requested_data_dir = os.path.join(BASE_DIR, _requested_data_dir)
try:
    os.makedirs(_requested_data_dir, exist_ok=True)
    DATA_DIR = _requested_data_dir
except Exception:
    DATA_DIR = BASE_DIR

app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret-change-this')

CORS(app, resources={r"/api/*": {"origins": "*"}})
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

RENDER_URL = os.environ.get('RENDER_EXTERNAL_URL', '')
ON_RENDER = bool(os.environ.get('RENDER') or RENDER_URL)
IS_HTTPS = RENDER_URL.startswith('https://')
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=True if ON_RENDER else False,
    PREFERRED_URL_SCHEME='https' if IS_HTTPS else 'http'
)

# ---------- file paths ----------
BUSES_FILE_NAME = 'buses_location.json'
LOCATIONS_FILE_NAME = 'locations.json'
CREDENTIALS_FILE_NAME = 'credentials.json'
AUDIT_FILE_NAME = 'admin_audit.json'

BUSES_FILE = os.path.join(DATA_DIR, BUSES_FILE_NAME)
LOCATIONS_FILE = os.path.join(DATA_DIR, LOCATIONS_FILE_NAME)
CREDENTIALS_FILE = os.path.join(DATA_DIR, CREDENTIALS_FILE_NAME)
AUDIT_FILE = os.path.join(DATA_DIR, AUDIT_FILE_NAME)

PIN_ADMIN_SIGNUP = 'admin_signup_pin'
PIN_GOLD_SIGNUP = 'gold_signup_pin'
PIN_ADMIN_LOGIN = 'admin_login_pin'
PIN_GOLD_LOGIN = 'gold_login_pin'
PIN_KEYS = (PIN_ADMIN_SIGNUP, PIN_GOLD_SIGNUP, PIN_ADMIN_LOGIN, PIN_GOLD_LOGIN)
DEFAULT_PINS = {
    PIN_ADMIN_SIGNUP: '456123',
    PIN_GOLD_SIGNUP: '456789',
    PIN_ADMIN_LOGIN: '456123',
    PIN_GOLD_LOGIN: '456789',
}
DEFAULT_UI_THEME = {
    'accent_color': '#8b64ff',
    'saturation': 120,
}
DEFAULT_ROUTE_SNAP_SETTINGS = {
    'enabled': True,
    'distance_m': 10,
    'show_range': False,
}
DEFAULT_LOCATIONS_PAYLOAD = {"hostels": [], "classes": [], "routes": []}
DEFAULT_CREDENTIALS_PAYLOAD = {"admins": [], "institute_name": "INSTITUTE"}
ADMIN_ROLE_STANDARD = 'admin'
ADMIN_ROLE_GOLD = 'gold'
GOOGLE_MAPS_API_KEY_FALLBACK = ''
_app_disk_io_lock = threading.Lock()
APP_DISK_READ_BYTES = 0
APP_DISK_WRITE_BYTES = 0
ROOT_TRACE_ENABLED = str(os.environ.get('ROOT_DEBUG_TRACE', '')).strip().lower() in ('1', 'true', 'yes')

_locations_cache_lock = threading.Lock()
_locations_cache_data = None
_locations_cache_mtime = None
_locations_cache_route_index = {}

_credentials_cache_lock = threading.Lock()
_credentials_cache_data = None
_credentials_cache_mtime = None

_metrics_lock = threading.Lock()
_app_ready_lock = threading.Lock()

_login_rate_lock = threading.Lock()
_login_rate_state = {}  # ip -> {'window_start': float, 'fail_count': int, 'blocked_until': float}
LOGIN_RATE_WINDOW_SEC = max(60, int(os.environ.get('LOGIN_RATE_WINDOW_SEC', '300')))
LOGIN_RATE_MAX_FAILURES = max(3, int(os.environ.get('LOGIN_RATE_MAX_FAILURES', '8')))
LOGIN_RATE_BLOCK_SEC = max(30, int(os.environ.get('LOGIN_RATE_BLOCK_SEC', '300')))

# ---------- simple JSON helpers ----------
def _get_file_mtime(path):
    try:
        return os.path.getmtime(path)
    except Exception:
        return None

def _normalize_locations_payload(raw):
    data = raw if isinstance(raw, dict) else {}
    hostels = data.get('hostels') if isinstance(data.get('hostels'), list) else []
    classes = data.get('classes') if isinstance(data.get('classes'), list) else []
    routes = data.get('routes') if isinstance(data.get('routes'), list) else []
    return {'hostels': hostels, 'classes': classes, 'routes': routes}

def _merge_locations_payload(base_payload, overlay_payload):
    """Merge overlay into base by stable ids without deleting existing records."""
    base = _normalize_locations_payload(base_payload)
    overlay = _normalize_locations_payload(overlay_payload)
    merged = {
        'hostels': list(base.get('hostels', [])),
        'classes': list(base.get('classes', [])),
        'routes': list(base.get('routes', [])),
    }

    def merge_list_by_id(target_list, incoming_list, prefix):
        seen = set()
        for item in target_list:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get('id') or '')
            if item_id:
                seen.add(item_id)
        fallback_counter = 0
        for item in incoming_list if isinstance(incoming_list, list) else []:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get('id') or '').strip()
            if not item_id:
                fallback_counter += 1
                item_id = f'{prefix}_legacy_{fallback_counter}'
                if item_id in seen:
                    continue
                cloned = dict(item)
                cloned['id'] = item_id
                target_list.append(cloned)
                seen.add(item_id)
                continue
            if item_id in seen:
                continue
            target_list.append(dict(item))
            seen.add(item_id)

    merge_list_by_id(merged['hostels'], overlay.get('hostels', []), 'hostel')
    merge_list_by_id(merged['classes'], overlay.get('classes', []), 'class')
    merge_list_by_id(merged['routes'], overlay.get('routes', []), 'route')
    return merged

def _build_route_index(routes):
    idx = {}
    for route in routes if isinstance(routes, list) else []:
        if not isinstance(route, dict):
            continue
        rid = route.get('id')
        if rid is None:
            continue
        key = str(rid)
        if key not in idx:
            idx[key] = route
    return idx

def load_json(path, default):
    global APP_DISK_READ_BYTES
    try:
        try:
            size_est = os.path.getsize(path)
            with _app_disk_io_lock:
                APP_DISK_READ_BYTES += max(0, int(size_est))
        except Exception:
            pass
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default

def _save_json_with_status(path, data):
    global APP_DISK_WRITE_BYTES
    try:
        body = json.dumps(data, indent=2)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(body)
        try:
            with _app_disk_io_lock:
                APP_DISK_WRITE_BYTES += len(body.encode('utf-8'))
        except Exception:
            pass
        return True
    except Exception:
        return False

def save_json(path, data):
    _save_json_with_status(path, data)

def _get_locations_cached_snapshot():
    global _locations_cache_data, _locations_cache_mtime, _locations_cache_route_index
    current_mtime = _get_file_mtime(LOCATIONS_FILE)
    with _locations_cache_lock:
        if _locations_cache_data is not None and _locations_cache_mtime == current_mtime:
            return _locations_cache_data, _locations_cache_route_index
    loaded = _normalize_locations_payload(load_json(LOCATIONS_FILE, copy.deepcopy(DEFAULT_LOCATIONS_PAYLOAD)))
    route_index = _build_route_index(loaded.get('routes', []))
    with _locations_cache_lock:
        _locations_cache_data = loaded
        _locations_cache_mtime = _get_file_mtime(LOCATIONS_FILE)
        _locations_cache_route_index = route_index
        return _locations_cache_data, _locations_cache_route_index

def get_locations_readonly():
    data, _ = _get_locations_cached_snapshot()
    return data

def get_locations_for_update():
    return copy.deepcopy(get_locations_readonly())

def get_route_from_locations(route_id):
    if route_id is None:
        return None
    _, route_index = _get_locations_cached_snapshot()
    return route_index.get(str(route_id))

def save_locations(data):
    global _locations_cache_data, _locations_cache_mtime, _locations_cache_route_index
    normalized = _normalize_locations_payload(data)
    ok = _save_json_with_status(LOCATIONS_FILE, normalized)
    if ok:
        cache_payload = copy.deepcopy(normalized)
        with _locations_cache_lock:
            _locations_cache_data = cache_payload
            _locations_cache_mtime = _get_file_mtime(LOCATIONS_FILE)
            _locations_cache_route_index = _build_route_index(cache_payload.get('routes', []))
    return ok

def _get_credentials_cached_payload():
    global _credentials_cache_data, _credentials_cache_mtime
    current_mtime = _get_file_mtime(CREDENTIALS_FILE)
    with _credentials_cache_lock:
        if _credentials_cache_data is not None and _credentials_cache_mtime == current_mtime:
            return copy.deepcopy(_credentials_cache_data)
    loaded = load_json(CREDENTIALS_FILE, copy.deepcopy(DEFAULT_CREDENTIALS_PAYLOAD))
    payload = loaded if isinstance(loaded, dict) else copy.deepcopy(DEFAULT_CREDENTIALS_PAYLOAD)
    with _credentials_cache_lock:
        _credentials_cache_data = copy.deepcopy(payload)
        _credentials_cache_mtime = _get_file_mtime(CREDENTIALS_FILE)
    return payload

def _update_credentials_cache(payload):
    global _credentials_cache_data, _credentials_cache_mtime
    if not isinstance(payload, dict):
        return
    with _credentials_cache_lock:
        _credentials_cache_data = copy.deepcopy(payload)
        _credentials_cache_mtime = _get_file_mtime(CREDENTIALS_FILE)

def ensure_files():
    if not os.path.exists(BUSES_FILE):
        legacy = os.path.join(BASE_DIR, BUSES_FILE_NAME)
        if os.path.abspath(legacy) != os.path.abspath(BUSES_FILE) and os.path.exists(legacy):
            try:
                shutil.copy2(legacy, BUSES_FILE)
            except Exception:
                pass
    if not os.path.exists(BUSES_FILE):
        save_json(BUSES_FILE, {})
    if not os.path.exists(LOCATIONS_FILE):
        legacy = os.path.join(BASE_DIR, LOCATIONS_FILE_NAME)
        if os.path.abspath(legacy) != os.path.abspath(LOCATIONS_FILE) and os.path.exists(legacy):
            try:
                shutil.copy2(legacy, LOCATIONS_FILE)
            except Exception:
                pass
    if not os.path.exists(LOCATIONS_FILE):
        save_locations(copy.deepcopy(DEFAULT_LOCATIONS_PAYLOAD))
    # If DATA_DIR is different from BASE_DIR, merge any committed repo locations
    # so deploys can pick up routes committed in git without wiping existing data.
    legacy_locations = os.path.join(BASE_DIR, LOCATIONS_FILE_NAME)
    if os.path.abspath(legacy_locations) != os.path.abspath(LOCATIONS_FILE) and os.path.exists(legacy_locations):
        active_payload = load_json(LOCATIONS_FILE, copy.deepcopy(DEFAULT_LOCATIONS_PAYLOAD))
        legacy_payload = load_json(legacy_locations, copy.deepcopy(DEFAULT_LOCATIONS_PAYLOAD))
        merged_payload = _merge_locations_payload(active_payload, legacy_payload)
        if json.dumps(_normalize_locations_payload(active_payload), sort_keys=True) != json.dumps(_normalize_locations_payload(merged_payload), sort_keys=True):
            save_locations(merged_payload)
    if not os.path.exists(CREDENTIALS_FILE):
        legacy = os.path.join(BASE_DIR, CREDENTIALS_FILE_NAME)
        if os.path.abspath(legacy) != os.path.abspath(CREDENTIALS_FILE) and os.path.exists(legacy):
            try:
                shutil.copy2(legacy, CREDENTIALS_FILE)
            except Exception:
                pass
    if not os.path.exists(CREDENTIALS_FILE):
        save_json(CREDENTIALS_FILE, copy.deepcopy(DEFAULT_CREDENTIALS_PAYLOAD))
        _update_credentials_cache(copy.deepcopy(DEFAULT_CREDENTIALS_PAYLOAD))
    if not os.path.exists(AUDIT_FILE):
        legacy = os.path.join(BASE_DIR, AUDIT_FILE_NAME)
        if os.path.abspath(legacy) != os.path.abspath(AUDIT_FILE) and os.path.exists(legacy):
            try:
                shutil.copy2(legacy, AUDIT_FILE)
            except Exception:
                pass
    if not os.path.exists(AUDIT_FILE):
        save_json(AUDIT_FILE, [])
    try:
        service_bootstrap_from_legacy(CREDENTIALS_FILE)
    except Exception:
        pass
    try:
        service_load_admins(force=True)
    except Exception:
        pass

def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

def _client_ip():
    if not has_request_context():
        return 'system'
    try:
        xff = request.headers.get('X-Forwarded-For', '')
        if xff:
            return xff.split(',')[0].strip()
    except Exception:
        pass
    try:
        return request.remote_addr or 'unknown'
    except Exception:
        return 'unknown'

def _prune_login_rate_state(now_ts=None):
    now = float(now_ts if now_ts is not None else time.time())
    window_cutoff = now - max(1, LOGIN_RATE_WINDOW_SEC)
    with _login_rate_lock:
        stale_keys = []
        for ip, state in _login_rate_state.items():
            blocked_until = float((state or {}).get('blocked_until') or 0.0)
            window_start = float((state or {}).get('window_start') or 0.0)
            fail_count = int((state or {}).get('fail_count') or 0)
            if blocked_until > now:
                continue
            if fail_count <= 0 and window_start <= window_cutoff:
                stale_keys.append(ip)
            elif window_start <= window_cutoff:
                stale_keys.append(ip)
        for ip in stale_keys:
            _login_rate_state.pop(ip, None)

def _is_login_rate_limited(ip, now_ts=None):
    now = float(now_ts if now_ts is not None else time.time())
    _prune_login_rate_state(now)
    with _login_rate_lock:
        state = _login_rate_state.get(str(ip or 'unknown'), {})
        blocked_until = float((state or {}).get('blocked_until') or 0.0)
        if blocked_until > now:
            return True, max(1, int(blocked_until - now))
    return False, 0

def _record_login_attempt(ip, success, now_ts=None):
    now = float(now_ts if now_ts is not None else time.time())
    key = str(ip or 'unknown')
    with _login_rate_lock:
        state = _login_rate_state.get(key)
        if not state:
            state = {'window_start': now, 'fail_count': 0, 'blocked_until': 0.0}
            _login_rate_state[key] = state
        if success:
            state['window_start'] = now
            state['fail_count'] = 0
            state['blocked_until'] = 0.0
            return
        if (now - float(state.get('window_start') or 0.0)) > LOGIN_RATE_WINDOW_SEC:
            state['window_start'] = now
            state['fail_count'] = 0
        state['fail_count'] = int(state.get('fail_count') or 0) + 1
        if state['fail_count'] >= LOGIN_RATE_MAX_FAILURES:
            state['blocked_until'] = now + LOGIN_RATE_BLOCK_SEC

_audit_lock = threading.Lock()
_audit_logs = []
AUDIT_RETENTION_SEC = 4 * 24 * 60 * 60
AUDIT_MAX_ITEMS = 2000

def prune_audit_logs(logs, now_epoch=None):
    if not isinstance(logs, list):
        return []
    now_ts = now_epoch if now_epoch is not None else time.time()
    cutoff = now_ts - AUDIT_RETENTION_SEC
    kept = []
    for entry in logs:
        if not isinstance(entry, dict):
            continue
        ts = parse_iso_timestamp(entry.get('ts'))
        if ts is None:
            continue
        if ts >= cutoff:
            kept.append(entry)
    if len(kept) > AUDIT_MAX_ITEMS:
        kept = kept[-AUDIT_MAX_ITEMS:]
    return kept

def record_audit(event, status='success', username=None, details=''):
    """Persist lightweight admin audit entries for observability in admin panel."""
    try:
        actor = username
        if not actor and has_request_context():
            actor = session.get('admin')
        ua = ''
        if has_request_context() and request and request.user_agent:
            ua = request.user_agent.string or ''
        entry = {
            'ts': _utc_now_iso(),
            'event': str(event or '').strip() or 'unknown',
            'status': str(status or 'success'),
            'username': actor or 'anonymous',
            'ip': _client_ip(),
            'details': str(details or '')[:240],
            'ua': ua[:180]
        }
        with _audit_lock:
            _audit_logs[:] = prune_audit_logs(_audit_logs)
            _audit_logs.append(entry)
            _audit_logs[:] = prune_audit_logs(_audit_logs)
            save_json(AUDIT_FILE, _audit_logs)
    except Exception:
        pass

def _audit_status_for_http_status(status_code):
    try:
        code = int(status_code or 0)
    except Exception:
        return 'failed'
    if code >= 500:
        return 'error'
    if code >= 400:
        return 'failed'
    return 'success'

def _should_capture_admin_activity():
    if not has_request_context():
        return False
    path = str(request.path or '')
    if not path or path == '/favicon.ico' or path.startswith('/static/'):
        return False
    if path.startswith('/admin'):
        return True
    return bool(session.get('admin'))

def _admin_activity_actor():
    if not has_request_context():
        return 'anonymous'
    actor = session.get('admin')
    if actor:
        return str(actor).strip() or 'anonymous'
    if request.path == '/admin/login' and request.method == 'POST':
        try:
            username = (request.form.get('username') or '').strip()
            if username:
                return username
        except Exception:
            pass
    return 'anonymous'

def _record_admin_request_activity(resp):
    if not _should_capture_admin_activity():
        return
    method = str(request.method or 'GET').upper()
    path = str(request.path or '/')
    status_code = int(getattr(resp, 'status_code', 0) or 0)
    query_text = ''
    try:
        if request.query_string:
            query_text = request.query_string.decode('utf-8', errors='ignore')
    except Exception:
        query_text = ''
    action = ''
    try:
        if request.path == '/admin/login' and request.method == 'POST':
            action = (request.form.get('action') or '').strip()
    except Exception:
        action = ''
    details = f'method={method} path={path} status={status_code}'
    if query_text:
        details += f' query={query_text[:80]}'
    if action:
        details += f' action={action}'
    record_audit(
        'admin_request',
        status=_audit_status_for_http_status(status_code),
        username=_admin_activity_actor(),
        details=details
    )

def is_admin_activity_log(entry):
    if not isinstance(entry, dict):
        return False
    event = str(entry.get('event') or '').strip().lower()
    username = str(entry.get('username') or '').strip().lower()
    details = str(entry.get('details') or '').strip().lower()
    if event.startswith('admin_'):
        return True
    if 'path=/admin' in details:
        return True
    if username and username not in ('system', 'anonymous'):
        return True
    return False

def get_process_rss_mb():
    rss_bytes = None
    try:
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if raw:
            # Linux reports KB, macOS reports bytes.
            rss_bytes = int(raw) if sys.platform == 'darwin' else int(raw) * 1024
    except Exception:
        pass
    if rss_bytes is None:
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        parts = line.split()
                        if len(parts) >= 2:
                            rss_bytes = int(parts[1]) * 1024  # kB -> bytes
                        break
        except Exception:
            pass
    if not rss_bytes:
        try:
            if sys.platform.startswith('win'):
                import ctypes
                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ('cb', ctypes.c_uint32),
                        ('PageFaultCount', ctypes.c_uint32),
                        ('PeakWorkingSetSize', ctypes.c_size_t),
                        ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t),
                        ('PeakPagefileUsage', ctypes.c_size_t),
                    ]
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                proc = ctypes.windll.kernel32.GetCurrentProcess()
                if ctypes.windll.psapi.GetProcessMemoryInfo(proc, ctypes.byref(counters), counters.cb):
                    rss_bytes = int(counters.WorkingSetSize)
        except Exception:
            pass
    if not rss_bytes:
        return None
    return round(rss_bytes / (1024 * 1024), 2)

def _read_int_file(path):
    try:
        with open(path, 'r') as f:
            raw = (f.read() or '').strip()
        if not raw or raw.lower() == 'max':
            return None
        return int(raw)
    except Exception:
        return None

def get_cgroup_memory_stats():
    """Best-effort cgroup memory stats (container-aware) for Linux."""
    # cgroup v2
    limit = _read_int_file('/sys/fs/cgroup/memory.max')
    used = _read_int_file('/sys/fs/cgroup/memory.current')
    if limit is not None and 0 < limit < (1 << 60):
        if used is not None:
            used = max(0, min(int(used), int(limit)))
        return {
            'source': 'cgroup_v2',
            'limit_bytes': int(limit),
            'used_bytes': int(used) if used is not None else None
        }
    # cgroup v1
    limit = _read_int_file('/sys/fs/cgroup/memory/memory.limit_in_bytes')
    used = _read_int_file('/sys/fs/cgroup/memory/memory.usage_in_bytes')
    if limit is not None and 0 < limit < (1 << 60):
        if used is not None:
            used = max(0, min(int(used), int(limit)))
        return {
            'source': 'cgroup_v1',
            'limit_bytes': int(limit),
            'used_bytes': int(used) if used is not None else None
        }
    return {
        'source': None,
        'limit_bytes': None,
        'used_bytes': None
    }

def get_system_memory_stats():
    total_mb = None
    available_mb = None
    source = None
    if sys.platform.startswith('win'):
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_uint32),
                    ('dwMemoryLoad', ctypes.c_uint32),
                    ('ullTotalPhys', ctypes.c_uint64),
                    ('ullAvailPhys', ctypes.c_uint64),
                    ('ullTotalPageFile', ctypes.c_uint64),
                    ('ullAvailPageFile', ctypes.c_uint64),
                    ('ullTotalVirtual', ctypes.c_uint64),
                    ('ullAvailVirtual', ctypes.c_uint64),
                    ('ullAvailExtendedVirtual', ctypes.c_uint64),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_mb = round(int(stat.ullTotalPhys) / (1024 * 1024), 2)
                available_mb = round(int(stat.ullAvailPhys) / (1024 * 1024), 2)
                source = 'windows_api'
        except Exception:
            pass
    else:
        cgroup = get_cgroup_memory_stats()
        if cgroup.get('limit_bytes') is not None:
            total_mb = round(int(cgroup['limit_bytes']) / (1024 * 1024), 2)
            if cgroup.get('used_bytes') is not None:
                used_mb = round(int(cgroup['used_bytes']) / (1024 * 1024), 2)
                available_mb = round(max(0.0, total_mb - used_mb), 2)
            source = cgroup.get('source') or source
        try:
            if total_mb is None or available_mb is None:
                mem_total_kb = None
                mem_available_kb = None
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            parts = line.split()
                            if len(parts) >= 2:
                                mem_total_kb = int(parts[1])
                        elif line.startswith('MemAvailable:'):
                            parts = line.split()
                            if len(parts) >= 2:
                                mem_available_kb = int(parts[1])
                        if mem_total_kb is not None and mem_available_kb is not None:
                            break
                if mem_total_kb is not None:
                    total_mb = round(mem_total_kb / 1024, 2)
                    source = source or 'proc_meminfo'
                if mem_available_kb is not None:
                    available_mb = round(mem_available_kb / 1024, 2)
        except Exception:
            pass
    if total_mb is None:
        try:
            page_size = os.sysconf('SC_PAGE_SIZE')
            page_count = os.sysconf('SC_PHYS_PAGES')
            if page_size and page_count:
                total_mb = round((page_size * page_count) / (1024 * 1024), 2)
                source = source or 'sysconf'
        except Exception:
            pass
    used_mb = None
    if total_mb is not None and available_mb is not None:
        used_mb = round(max(0.0, total_mb - available_mb), 2)
    return {
        'total_mb': total_mb,
        'available_mb': available_mb,
        'used_mb': used_mb,
        'source': source or 'unknown'
    }

_cpu_sample_lock = threading.Lock()
_cpu_last_wall = time.monotonic()
_cpu_last_proc = time.process_time()
_cpu_last_percent = 0.0

def get_process_cpu_percent():
    global _cpu_last_wall, _cpu_last_proc, _cpu_last_percent
    now_wall = time.monotonic()
    now_proc = time.process_time()
    with _cpu_sample_lock:
        dt_wall = now_wall - _cpu_last_wall
        dt_proc = now_proc - _cpu_last_proc
        _cpu_last_wall = now_wall
        _cpu_last_proc = now_proc
        if dt_wall <= 0:
            return round(_cpu_last_percent, 2)
        raw_percent = max(0.0, (dt_proc / dt_wall) * 100.0)
        cpu_cap = float(max(1, os.cpu_count() or 1) * 100)
        raw_percent = min(cpu_cap, raw_percent)
        _cpu_last_percent = (_cpu_last_percent * 0.6) + (raw_percent * 0.4)
        return round(_cpu_last_percent, 2)

def get_process_cpu_stats():
    return {
        'process_percent': get_process_cpu_percent(),
        'cores': max(1, os.cpu_count() or 1)
    }

_disk_sample_lock = threading.Lock()
_disk_last_ts = time.monotonic()
_disk_last_read_bytes = None
_disk_last_write_bytes = None
_disk_last_read_kbps = 0.0
_disk_last_write_kbps = 0.0

def _read_process_io_bytes():
    read_bytes = None
    write_bytes = None
    try:
        import psutil  # type: ignore[import-not-found]  # optional dependency
        proc = psutil.Process(os.getpid())
        io = proc.io_counters()
        read_bytes = int(getattr(io, 'read_bytes', 0))
        write_bytes = int(getattr(io, 'write_bytes', 0))
        if read_bytes is not None and write_bytes is not None:
            return {'read_bytes': read_bytes, 'write_bytes': write_bytes}
    except Exception:
        pass
    if sys.platform.startswith('win'):
        try:
            import ctypes
            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ('ReadOperationCount', ctypes.c_uint64),
                    ('WriteOperationCount', ctypes.c_uint64),
                    ('OtherOperationCount', ctypes.c_uint64),
                    ('ReadTransferCount', ctypes.c_uint64),
                    ('WriteTransferCount', ctypes.c_uint64),
                    ('OtherTransferCount', ctypes.c_uint64),
                ]
            counters = IO_COUNTERS()
            proc = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.kernel32.GetProcessIoCounters(proc, ctypes.byref(counters)):
                read_bytes = int(counters.ReadTransferCount)
                write_bytes = int(counters.WriteTransferCount)
        except Exception:
            pass
    else:
        try:
            with open('/proc/self/io', 'r') as f:
                for raw_line in f:
                    line = raw_line.strip().lower()
                    if line.startswith('read_bytes:'):
                        parts = line.split(':', 1)
                        read_bytes = int(parts[1].strip()) if len(parts) == 2 else read_bytes
                    elif line.startswith('write_bytes:'):
                        parts = line.split(':', 1)
                        write_bytes = int(parts[1].strip()) if len(parts) == 2 else write_bytes
        except Exception:
            pass
    if read_bytes is None or write_bytes is None:
        try:
            with _app_disk_io_lock:
                read_bytes = APP_DISK_READ_BYTES if read_bytes is None else read_bytes
                write_bytes = APP_DISK_WRITE_BYTES if write_bytes is None else write_bytes
        except Exception:
            pass
    return {'read_bytes': read_bytes, 'write_bytes': write_bytes}

def get_process_disk_io_stats():
    """Low-overhead process disk IO speed sample using cumulative byte counters."""
    global _disk_last_ts, _disk_last_read_bytes, _disk_last_write_bytes, _disk_last_read_kbps, _disk_last_write_kbps
    sample = _read_process_io_bytes()
    now = time.monotonic()
    current_read = sample.get('read_bytes')
    current_write = sample.get('write_bytes')
    with _disk_sample_lock:
        prev_ts = _disk_last_ts
        prev_read = _disk_last_read_bytes
        prev_write = _disk_last_write_bytes
        dt = max(0.0, now - prev_ts)
        _disk_last_ts = now

        if current_read is not None:
            _disk_last_read_bytes = int(current_read)
        if current_write is not None:
            _disk_last_write_bytes = int(current_write)

        if dt > 0 and prev_read is not None and current_read is not None:
            delta_read = max(0, int(current_read) - int(prev_read))
            raw_read_kbps = ((delta_read * 8.0) / 1000.0) / dt
            _disk_last_read_kbps = (_disk_last_read_kbps * 0.6) + (raw_read_kbps * 0.4)

        if dt > 0 and prev_write is not None and current_write is not None:
            delta_write = max(0, int(current_write) - int(prev_write))
            raw_write_kbps = ((delta_write * 8.0) / 1000.0) / dt
            _disk_last_write_kbps = (_disk_last_write_kbps * 0.6) + (raw_write_kbps * 0.4)

        return {
            'read_bytes_total': int(current_read) if current_read is not None else None,
            'write_bytes_total': int(current_write) if current_write is not None else None,
            'read_kbps': round(_disk_last_read_kbps, 2) if current_read is not None else None,
            'write_kbps': round(_disk_last_write_kbps, 2) if current_write is not None else None,
            'sample_window_sec': round(dt, 3) if dt > 0 else None
        }

def get_storage_stats(path=None):
    target = path or BASE_DIR
    try:
        usage = shutil.disk_usage(target)
        total_bytes = int(usage.total)
        used_bytes = int(usage.used)
        free_bytes = int(usage.free)
        used_percent = round((used_bytes / total_bytes) * 100.0, 2) if total_bytes > 0 else None
        return {
            'path': target,
            'total_bytes': total_bytes,
            'used_bytes': used_bytes,
            'free_bytes': free_bytes,
            'total_gb': round(total_bytes / (1024 ** 3), 2),
            'used_gb': round(used_bytes / (1024 ** 3), 2),
            'free_gb': round(free_bytes / (1024 ** 3), 2),
            'used_percent': used_percent
        }
    except Exception:
        return {
            'path': target,
            'total_bytes': None,
            'used_bytes': None,
            'free_bytes': None,
            'total_gb': None,
            'used_gb': None,
            'free_gb': None,
            'used_percent': None
        }

def parse_iso_timestamp(value):
    """Parse ISO timestamp into epoch seconds; returns None on invalid."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
    except Exception:
        return None

# ---------- in-memory caches ----------
_buses = {}
_buses_lock = threading.Lock()
_worker_started = False

APP_START_TS = time.time()
REQUESTS_TOTAL = 0
BANDWIDTH_IN_BYTES = 0
BANDWIDTH_OUT_BYTES = 0
INACTIVE_REMOVE_SEC = 30
DESTINATION_REMOVE_SEC = 5
_bus_destination_ts = {}  # bus_id -> monotonic timestamp when destination reached
_buses_dirty = False

def _init_app():
    global _buses, _worker_started, _audit_logs, _buses_dirty
    init_t0 = time.perf_counter()
    app.logger.info('_init_app start')
    step_t0 = time.perf_counter()
    ensure_files()
    app.logger.info('_init_app ensure_files done in %.1fms', (time.perf_counter() - step_t0) * 1000.0)
    step_t0 = time.perf_counter()
    raw = load_json(BUSES_FILE, {})
    app.logger.info('_init_app load buses json done in %.1fms', (time.perf_counter() - step_t0) * 1000.0)
    step_t0 = time.perf_counter()
    logs = load_json(AUDIT_FILE, [])
    app.logger.info('_init_app load audit json done in %.1fms', (time.perf_counter() - step_t0) * 1000.0)
    _audit_logs = prune_audit_logs(logs if isinstance(logs, list) else [])
    if _audit_logs != (logs if isinstance(logs, list) else []):
        save_json(AUDIT_FILE, _audit_logs)
    # Filter out stale buses on startup (older than INACTIVE_REMOVE_SEC)
    now = time.time()
    cleaned = {}
    for k, v in raw.items():
        try:
            ts = parse_iso_timestamp(v.get('lastUpdate'))
            if ts is not None and (now - ts) <= INACTIVE_REMOVE_SEC:
                cleaned[k] = v
        except Exception:
            pass  # drop invalid entries
    _buses = cleaned
    _buses_dirty = False
    if cleaned != raw:
        save_json(BUSES_FILE, cleaned)
    if not _worker_started:
        _worker_started = True
        threading.Thread(target=_sync_worker, daemon=True).start()
    app.logger.info('_init_app done in %.1fms', (time.perf_counter() - init_t0) * 1000.0)

@app.before_request
def _before():
    global REQUESTS_TOTAL, BANDWIDTH_IN_BYTES
    trace = None
    try:
        if ROOT_TRACE_ENABLED and request.path in ('/', '/student'):
            trace = uuid.uuid4().hex[:8]
            request.environ['root_trace_id'] = trace
            app.logger.info('[req:%s] before_request start path=%s', trace, request.path)
    except Exception:
        trace = None
    if not hasattr(app, '_ready'):
        with _app_ready_lock:
            if not hasattr(app, '_ready'):
                ready_t0 = time.perf_counter()
                if trace:
                    app.logger.info('[req:%s] app not ready, running _init_app', trace)
                _init_app()
                app._ready = True
                if trace:
                    app.logger.info('[req:%s] _init_app finished in %.1fms', trace, (time.perf_counter() - ready_t0) * 1000.0)
    with _metrics_lock:
        REQUESTS_TOTAL += 1
    try:
        in_len = request.content_length
        if in_len is None:
            in_len = int(request.headers.get('Content-Length', '0') or 0)
        if in_len and in_len > 0:
            with _metrics_lock:
                BANDWIDTH_IN_BYTES += int(in_len)
    except Exception:
        pass

@app.after_request
def _after(resp):
    resp.headers.setdefault('Permissions-Policy', 'gamepad=(self)')
    global BANDWIDTH_OUT_BYTES
    try:
        out_len = resp.calculate_content_length()
        if out_len is None and not resp.is_streamed:
            body = resp.get_data(as_text=False)
            out_len = len(body) if body is not None else 0
        if out_len and out_len > 0:
            with _metrics_lock:
                BANDWIDTH_OUT_BYTES += int(out_len)
    except Exception:
        pass
    try:
        _record_admin_request_activity(resp)
    except Exception:
        pass
    return resp

def _auto_cleanup_buses():
    global _buses_dirty
    now_epoch = time.time()
    now_mono = time.monotonic()
    removed = []
    with _buses_lock:
        for bus_id, data in list(_buses.items()):
            reason = None
            reached_ts = _bus_destination_ts.get(bus_id)
            if reached_ts is not None and (now_mono - reached_ts) >= DESTINATION_REMOVE_SEC:
                reason = 'destination_timeout'
            else:
                last_ts = parse_iso_timestamp((data or {}).get('lastUpdate'))
                if last_ts is None or (now_epoch - last_ts) >= INACTIVE_REMOVE_SEC:
                    reason = 'inactivity_timeout'
            if not reason:
                continue
            removed_data = _buses.pop(bus_id, None)
            if not removed_data:
                continue
            removed.append((str(bus_id), removed_data.get('routeId'), reason))
            _kalman_filters.pop(str(bus_id), None)
            _bus_stop_state.pop(str(bus_id), None)
            _bus_last_broadcast.pop(str(bus_id), None)
            _bus_destination_ts.pop(str(bus_id), None)
            _buses_dirty = True
        # Drop dangling destination timers.
        for bus_id in list(_bus_destination_ts.keys()):
            if bus_id not in _buses:
                _bus_destination_ts.pop(bus_id, None)
    for bus_id, route_id, reason in removed:
        remove_driver_presence_for_bus(bus_id)
        record_audit('bus_auto_remove', status='success', username='system', details=f'bus={bus_id} reason={reason}')
        try:
            broadcast({'type': 'bus_stop', 'bus': bus_id, 'routeId': route_id, 'reason': reason})
        except Exception:
            pass

def _sync_worker():
    global _buses_dirty
    while True:
        time.sleep(1)
        try:
            _auto_cleanup_buses()
            snap = None
            with _buses_lock:
                if _buses_dirty:
                    snap = {k: dict(v) for k, v in _buses.items()}
                    _buses_dirty = False
            if snap is not None:
                save_json(BUSES_FILE, snap)
        except Exception:
            pass

# ---------- SSE ----------
_subscribers_lock = threading.Lock()
_subscribers = {}   # routeId|"all" -> [queue, ...]
_active_admin_sessions = {}  # session_id -> {'username': str, 'last_seen': ts}
_active_admin_lock = threading.Lock()
ACTIVE_ADMIN_TTL_SEC = 60 * 60
_presence_lock = threading.Lock()
_student_presence = {}  # client_id -> last_seen epoch seconds
_driver_presence = {}   # driver_key -> last_seen epoch seconds
STUDENT_PRESENCE_TTL_SEC = max(20, int(os.environ.get('STUDENT_PRESENCE_TTL_SEC', '45')))
DRIVER_PRESENCE_TTL_SEC = max(20, int(os.environ.get('DRIVER_PRESENCE_TTL_SEC', '45')))
SSE_QUEUE_MAXSIZE = max(200, int(os.environ.get('SSE_QUEUE_MAXSIZE', '2000')))

def broadcast(payload):
    try:
        data = json.dumps(payload)
    except Exception:
        data = json.dumps({"error": "bad-payload"})
    # Extract routeId for targeted delivery
    route_id = None
    if isinstance(payload, dict):
        route_id = payload.get('routeId')
        if not route_id:
            bus_data = payload.get('data')
            if isinstance(bus_data, dict):
                route_id = bus_data.get('routeId')
    targets = []
    with _subscribers_lock:
        if route_id:
            # Targeted: send to "all" subscribers + matching route subscribers
            sent = set()
            for q in list(_subscribers.get('all', [])):
                qid = id(q)
                if qid in sent:
                    continue
                sent.add(qid)
                targets.append(q)
            for q in list(_subscribers.get(route_id, [])):
                qid = id(q)
                if qid in sent:
                    continue
                sent.add(qid)
                targets.append(q)
        else:
            # No route context (buses_clear, etc.) - send to all subscribers
            sent = set()
            for group in _subscribers.values():
                for q in list(group):
                    qid = id(q)
                    if qid in sent:
                        continue
                    sent.add(qid)
                    targets.append(q)
    for q in targets:
        try:
            q.put_nowait(data)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(data)
            except Exception:
                pass
        except Exception:
            pass

@app.route('/events')
def sse_events():
    if os.environ.get('DISABLE_SSE', '').lower() in ('1', 'true', 'yes'):
        return Response('SSE disabled', status=503, mimetype='text/plain')
    route_id = request.args.get('routeId') or 'all'
    def stream():
        q = queue.Queue(maxsize=SSE_QUEUE_MAXSIZE)
        hb = max(5, int(os.environ.get('SSE_HEARTBEAT_SEC', '20')))
        with _subscribers_lock:
            _subscribers.setdefault(route_id, []).append(q)
        yield 'event: ping\ndata: "connected"\n\n'
        try:
            while True:
                try:
                    msg = q.get(timeout=hb)
                    yield f'data: {msg}\n\n'
                except queue.Empty:
                    yield 'event: ping\ndata: {}\n\n'
        finally:
            with _subscribers_lock:
                try:
                    subs = _subscribers.get(route_id)
                    if subs:
                        subs.remove(q)
                        if not subs:
                            del _subscribers[route_id]
                except (ValueError, KeyError):
                    pass
    return Response(stream_with_context(stream()), mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache, no-transform',
                        'X-Accel-Buffering': 'no',
                        'Connection': 'keep-alive',
                        'Content-Type': 'text/event-stream; charset=utf-8',
                        'Transfer-Encoding': 'chunked',
                    })

# ---------- credentials helpers ----------
def load_credentials(persist_changes=False):
    t0 = time.perf_counter()
    default_creds = copy.deepcopy(DEFAULT_CREDENTIALS_PAYLOAD)
    creds = _get_credentials_cached_payload()
    if not isinstance(creds, dict):
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > 120:
            app.logger.warning('load_credentials non-dict fallback in %.1fms', elapsed_ms)
        return dict(default_creds)

    changed = False
    live_admins = service_load_admins()
    if creds.get('admins') != live_admins:
        creds['admins'] = live_admins
    if 'institute_name' not in creds:
        creds['institute_name'] = default_creds['institute_name']
        changed = True
    raw_pins = creds.get('pins')
    if not isinstance(raw_pins, dict):
        raw_pins = {}
        changed = True
    normalized_pins = {}
    for key, default_pin in DEFAULT_PINS.items():
        pin_val = str(raw_pins.get(key) or '').strip()
        if not (pin_val.isdigit() and len(pin_val) == 6):
            pin_val = default_pin
            changed = True
        normalized_pins[key] = pin_val
    if normalized_pins.get(PIN_ADMIN_SIGNUP) == normalized_pins.get(PIN_GOLD_SIGNUP):
        normalized_pins[PIN_GOLD_SIGNUP] = DEFAULT_PINS[PIN_GOLD_SIGNUP]
        changed = True
    if creds.get('pins') != normalized_pins:
        creds['pins'] = normalized_pins
        changed = True
    normalized_ui_theme = sanitize_ui_theme(creds.get('ui_theme'))
    if creds.get('ui_theme') != normalized_ui_theme:
        creds['ui_theme'] = normalized_ui_theme
        changed = True
    normalized_route_snap = sanitize_route_snap_settings(creds.get('route_snap_settings'))
    if creds.get('route_snap_settings') != normalized_route_snap:
        creds['route_snap_settings'] = normalized_route_snap
        changed = True

    if changed and persist_changes:
        save_t0 = time.perf_counter()
        save_credentials(creds)
        save_elapsed_ms = (time.perf_counter() - save_t0) * 1000.0
        if save_elapsed_ms > 120:
            app.logger.warning('load_credentials save_json slow: %.1fms', save_elapsed_ms)
    total_elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if total_elapsed_ms > 120:
        app.logger.warning('load_credentials total slow: %.1fms changed=%s', total_elapsed_ms, changed)
    return creds

def save_credentials(data):
    payload = copy.deepcopy(data if isinstance(data, dict) else {})
    admins_payload = payload.pop('admins', None)
    if isinstance(admins_payload, list):
        try:
            service_save_admins(admins_payload)
        except Exception:
            pass
    ok = _save_json_with_status(CREDENTIALS_FILE, payload)
    if ok:
        _update_credentials_cache(payload)
    return ok

def sanitize_ui_theme(raw_theme):
    theme = raw_theme if isinstance(raw_theme, dict) else {}

    accent_raw = str(theme.get('accent_color') or DEFAULT_UI_THEME['accent_color']).strip()
    if len(accent_raw) == 4 and accent_raw.startswith('#'):
        accent_raw = '#' + ''.join(ch * 2 for ch in accent_raw[1:])
    if not (len(accent_raw) == 7 and accent_raw.startswith('#')):
        accent_raw = DEFAULT_UI_THEME['accent_color']
    else:
        hex_part = accent_raw[1:]
        if not all(ch in '0123456789abcdefABCDEF' for ch in hex_part):
            accent_raw = DEFAULT_UI_THEME['accent_color']
        else:
            accent_raw = '#' + hex_part.lower()

    sat_default = int(DEFAULT_UI_THEME['saturation'])
    try:
        sat_val = int(float(theme.get('saturation', sat_default)))
    except (TypeError, ValueError):
        sat_val = sat_default
    sat_val = max(20, min(260, sat_val))

    return {
        'accent_color': accent_raw,
        'saturation': sat_val,
    }

def get_ui_theme(creds=None):
    creds_obj = creds if isinstance(creds, dict) else load_credentials()
    return sanitize_ui_theme((creds_obj or {}).get('ui_theme'))

def _to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ('1', 'true', 'yes', 'on', 'enabled'):
            return True
        if v in ('0', 'false', 'no', 'off', 'disabled'):
            return False
    return bool(default)

def sanitize_route_snap_settings(raw_settings):
    settings = raw_settings if isinstance(raw_settings, dict) else {}
    default_distance = int(DEFAULT_ROUTE_SNAP_SETTINGS['distance_m'])
    try:
        distance_val = int(float(settings.get('distance_m', default_distance)))
    except (TypeError, ValueError):
        distance_val = default_distance
    distance_val = max(1, min(1000, distance_val))
    return {
        'enabled': _to_bool(settings.get('enabled'), DEFAULT_ROUTE_SNAP_SETTINGS['enabled']),
        'distance_m': distance_val,
        'show_range': _to_bool(settings.get('show_range'), DEFAULT_ROUTE_SNAP_SETTINGS['show_range']),
    }

def get_route_snap_settings(creds=None):
    creds_obj = creds if isinstance(creds, dict) else load_credentials()
    return sanitize_route_snap_settings((creds_obj or {}).get('route_snap_settings'))

def sanitize_route_snap_override(raw_settings, default_settings=None):
    fallback = sanitize_route_snap_settings(default_settings or DEFAULT_ROUTE_SNAP_SETTINGS)
    settings = raw_settings if isinstance(raw_settings, dict) else {}
    override_global = _to_bool(settings.get('override_global'), False)
    try:
        distance_val = int(float(settings.get('distance_m', fallback['distance_m'])))
    except (TypeError, ValueError):
        distance_val = int(fallback['distance_m'])
    distance_val = max(1, min(1000, distance_val))
    return {
        'override_global': override_global,
        'enabled': _to_bool(settings.get('enabled'), fallback['enabled']),
        'distance_m': distance_val,
        'show_range': _to_bool(settings.get('show_range'), fallback['show_range']),
    }

def sanitize_waypoint_list(raw_waypoints):
    cleaned = []
    if not isinstance(raw_waypoints, list):
        return cleaned
    for wp in raw_waypoints:
        if not isinstance(wp, (list, tuple)) or len(wp) < 2:
            continue
        try:
            lat = float(wp[0])
            lng = float(wp[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(lat) and math.isfinite(lng)):
            continue
        cleaned.append([lat, lng])
    return cleaned

def sanitize_follow_road_segments(raw_segments, segment_count, default_enabled=False):
    try:
        count = max(0, int(segment_count))
    except (TypeError, ValueError):
        count = 0
    fallback = _to_bool(default_enabled, False)
    cleaned = []
    if isinstance(raw_segments, list):
        for idx in range(min(len(raw_segments), count)):
            cleaned.append(_to_bool(raw_segments[idx], fallback))
    while len(cleaned) < count:
        cleaned.append(fallback)
    return cleaned

def get_admin_role(admin_entry):
    if not isinstance(admin_entry, dict):
        return ADMIN_ROLE_STANDARD
    role_raw = str(admin_entry.get('role') or ADMIN_ROLE_STANDARD).strip().lower()
    return ADMIN_ROLE_GOLD if role_raw == ADMIN_ROLE_GOLD else ADMIN_ROLE_STANDARD

def get_pin_config(creds=None):
    creds_obj = creds if isinstance(creds, dict) else load_credentials()
    pins = creds_obj.get('pins') if isinstance(creds_obj, dict) else {}
    merged = {}
    for key, default_pin in DEFAULT_PINS.items():
        v = str((pins or {}).get(key) or '').strip()
        merged[key] = v if (v.isdigit() and len(v) == 6) else default_pin
    if merged.get(PIN_ADMIN_SIGNUP) == merged.get(PIN_GOLD_SIGNUP):
        merged[PIN_GOLD_SIGNUP] = DEFAULT_PINS[PIN_GOLD_SIGNUP]
    return merged

def role_from_signup_pin(pin_value, creds=None):
    pins = get_pin_config(creds)
    pin = str(pin_value or '').strip()
    if pin == pins.get(PIN_GOLD_SIGNUP):
        return ADMIN_ROLE_GOLD
    if pin == pins.get(PIN_ADMIN_SIGNUP):
        return ADMIN_ROLE_STANDARD
    return None

def required_login_pin_for_role(role, creds=None):
    pins = get_pin_config(creds)
    safe_role = ADMIN_ROLE_GOLD if str(role or '').strip().lower() == ADMIN_ROLE_GOLD else ADMIN_ROLE_STANDARD
    if safe_role == ADMIN_ROLE_GOLD:
        return pins.get(PIN_GOLD_LOGIN)
    return pins.get(PIN_ADMIN_LOGIN)

def get_admin_record(creds, username):
    if not isinstance(creds, dict):
        return None
    for admin in creds.get('admins', []):
        if str(admin.get('username') or '') == str(username or ''):
            return admin
    return None

def current_admin_record(creds=None):
    username = session.get('admin') if has_request_context() else None
    if not username:
        return None
    creds_obj = creds if isinstance(creds, dict) else load_credentials()
    return get_admin_record(creds_obj, username)

def current_admin_is_gold(creds=None):
    admin = current_admin_record(creds)
    return get_admin_role(admin) == ADMIN_ROLE_GOLD

def access_denied_error():
    return {'error': 'Access denied'}

# ---------- admin session tracking ----------
def _prune_active_admin_sessions(now_ts=None):
    now = now_ts if now_ts is not None else time.time()
    cutoff = now - ACTIVE_ADMIN_TTL_SEC
    with _active_admin_lock:
        stale = [sid for sid, entry in _active_admin_sessions.items() if (entry or {}).get('last_seen', 0) < cutoff]
        for sid in stale:
            _active_admin_sessions.pop(sid, None)
        return len(_active_admin_sessions)

def touch_active_admin_session(username=None):
    if not has_request_context():
        return None
    sess_id = session.get('admin_session_id')
    if not sess_id:
        sess_id = uuid.uuid4().hex
        session['admin_session_id'] = sess_id
    now = time.time()
    with _active_admin_lock:
        _active_admin_sessions[sess_id] = {
            'username': username or session.get('admin') or 'unknown',
            'last_seen': now
        }
    _prune_active_admin_sessions(now)
    return sess_id

def remove_active_admin_session():
    if not has_request_context():
        return
    sess_id = session.pop('admin_session_id', None)
    if not sess_id:
        return
    with _active_admin_lock:
        _active_admin_sessions.pop(sess_id, None)

def get_active_admin_count():
    return _prune_active_admin_sessions()

def _driver_bus_presence_key(bus_id):
    return f'bus:{str(bus_id)}'

def _normalize_presence_id(raw_value, prefix):
    raw = str(raw_value or '').strip()
    if not raw:
        return None
    safe = ''.join(ch if (ch.isalnum() or ch in ('-', '_', ':', '.')) else '_' for ch in raw)[:96]
    if not safe:
        return None
    if safe.startswith(f'{prefix}:'):
        return safe
    return f'{prefix}:{safe}'

def _prune_presence_table(table, ttl_sec, now_ts=None):
    now = now_ts if now_ts is not None else time.time()
    cutoff = now - max(1, int(ttl_sec))
    stale_ids = [cid for cid, seen in table.items() if (seen or 0) < cutoff]
    for cid in stale_ids:
        table.pop(cid, None)
    return len(table)

def touch_student_presence(client_id):
    cid = _normalize_presence_id(client_id, 'student')
    if not cid:
        return None
    now = time.time()
    with _presence_lock:
        _student_presence[cid] = now
        _prune_presence_table(_student_presence, STUDENT_PRESENCE_TTL_SEC, now)
    return cid

def remove_student_presence(client_id):
    cid = _normalize_presence_id(client_id, 'student')
    if not cid:
        return
    with _presence_lock:
        _student_presence.pop(cid, None)

def get_active_student_count():
    with _presence_lock:
        return _prune_presence_table(_student_presence, STUDENT_PRESENCE_TTL_SEC)

def touch_driver_presence(driver_key):
    key = _normalize_presence_id(driver_key, 'driver')
    if not key:
        return None
    now = time.time()
    with _presence_lock:
        _driver_presence[key] = now
        _prune_presence_table(_driver_presence, DRIVER_PRESENCE_TTL_SEC, now)
    return key

def remove_driver_presence(driver_key):
    key = _normalize_presence_id(driver_key, 'driver')
    if not key:
        return
    with _presence_lock:
        _driver_presence.pop(key, None)

def remove_driver_presence_for_bus(bus_id):
    remove_driver_presence(_driver_bus_presence_key(bus_id))

def get_active_driver_count():
    with _presence_lock:
        return _prune_presence_table(_driver_presence, DRIVER_PRESENCE_TTL_SEC)

# ---------- auth ----------
def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if 'admin' not in session or not session.get('admin_authenticated'):
            return redirect(url_for('admin_login'))
        creds = load_credentials()
        if not get_admin_record(creds, session.get('admin')):
            remove_active_admin_session()
            session.pop('admin', None)
            session.pop('admin_authenticated', None)
            return redirect(url_for('admin_login'))
        touch_active_admin_session(session.get('admin'))
        return fn(*args, **kwargs)
    return wrapper

# ---------- page routes ----------
@app.route('/health')
def health_check():
    return 'OK', 200

@app.route('/')
def test_root():
    plain_root = str(os.environ.get('ROOT_PLAIN_TEXT_DEBUG', '')).strip().lower() in ('1', 'true', 'yes')
    if plain_root:
        if ROOT_TRACE_ENABLED:
            app.logger.info('GET / -> plain text fallback route')
        return 'server running', 200
    return _render_student_view()

def _render_student_view():
    trace = request.environ.get('root_trace_id') or uuid.uuid4().hex[:8]
    if ROOT_TRACE_ENABLED:
        app.logger.info('[student_view:%s] start', trace)
    creds_t0 = time.perf_counter()
    creds = load_credentials(persist_changes=False)
    creds_ms = (time.perf_counter() - creds_t0) * 1000.0
    if ROOT_TRACE_ENABLED:
        app.logger.info('[student_view:%s] load_credentials done in %.1fms', trace, creds_ms)
    template_t0 = time.perf_counter()
    rendered = render_template('student.html', institute_name=creds.get('institute_name', 'INSTITUTE'))
    template_ms = (time.perf_counter() - template_t0) * 1000.0
    if ROOT_TRACE_ENABLED:
        app.logger.info('[student_view:%s] render_template done in %.1fms', trace, template_ms)
    return rendered

@app.route('/student')
def student_view():
    return _render_student_view()

@app.route('/driver')
def driver_view():
    creds = load_credentials(persist_changes=False)
    return render_template('driver.html', institute_name=creds.get('institute_name', 'INSTITUTE'))

@app.route('/simulator')
def simulator_view():
    creds = load_credentials(persist_changes=False)
    return render_template('simulator.html', institute_name=creds.get('institute_name', 'INSTITUTE'))

@app.route('/admin')
@login_required
def admin_view():
    creds = load_credentials()
    current_admin = current_admin_record(creds)
    role = get_admin_role(current_admin)
    google_maps_api_key = str(os.environ.get('GOOGLE_MAPS_API_KEY', GOOGLE_MAPS_API_KEY_FALLBACK) or '').strip()
    return render_template(
        'admin.html',
        institute_name=creds.get('institute_name', 'INSTITUTE'),
        admin_user=session.get('admin'),
        admin_role=role,
        is_gold_admin=(role == ADMIN_ROLE_GOLD),
        google_maps_api_key=google_maps_api_key,
    )

PENDING_ADMIN_PIN_KEY = 'pending_admin_pin'
PENDING_ADMIN_PIN_TTL_SEC = 300

def _clear_pending_admin_pin():
    session.pop(PENDING_ADMIN_PIN_KEY, None)

def _get_pending_admin_pin(creds=None):
    if not has_request_context():
        return None
    pending = session.get(PENDING_ADMIN_PIN_KEY)
    if not isinstance(pending, dict):
        return None
    username = str(pending.get('username') or '').strip()
    role = str(pending.get('role') or ADMIN_ROLE_STANDARD).strip().lower()
    try:
        created_at = int(pending.get('created_at') or 0)
    except (TypeError, ValueError):
        created_at = 0
    now_ts = int(time.time())
    if not username or created_at <= 0 or (now_ts - created_at) > PENDING_ADMIN_PIN_TTL_SEC:
        _clear_pending_admin_pin()
        return None
    creds_obj = creds if isinstance(creds, dict) else load_credentials()
    admin = get_admin_record(creds_obj, username)
    if not admin:
        _clear_pending_admin_pin()
        return None
    safe_role = get_admin_role(admin)
    if role not in (ADMIN_ROLE_STANDARD, ADMIN_ROLE_GOLD):
        role = safe_role
    return {
        'username': username,
        'role': safe_role,
        'created_at': created_at,
    }

def _render_admin_login_view(creds, error_text=None, require_pin=False, pending_username='', institute_name=None, clear_pin_on_error=False):
    safe_creds = creds if isinstance(creds, dict) else load_credentials()
    return render_template(
        'admin_login.html',
        credentials_exist=bool(safe_creds.get('admins')),
        institute_name=(institute_name or safe_creds.get('institute_name', 'INSTITUTE')),
        error_text=error_text,
        require_gold_pin=bool(require_pin),
        pending_gold_username=(pending_username or ''),
        clear_pin_on_error=bool(clear_pin_on_error),
    )

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    try:
        creds = load_credentials()
        now_ts = int(time.time())
        if request.method == 'GET':
            pending = _get_pending_admin_pin(creds)
            return _render_admin_login_view(
                creds,
                require_pin=bool(pending),
                pending_username=(pending.get('username') if pending else '')
            )

        data = request.form
        action = (data.get('action') or 'login').strip().lower()
        institute = data.get('institute_name', '').strip()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        login_ip = _client_ip()

        if action == 'signup':
            _clear_pending_admin_pin()
            pin = (data.get('signup_pin', '') or '').strip()
            signup_role = role_from_signup_pin(pin, creds)
            if not signup_role:
                record_audit('admin_signup', status='failed', username=username or 'anonymous', details='invalid_pin')
                return _render_admin_login_view(creds, error_text='Invalid signup pin.', institute_name=institute)
            if not username or not password:
                record_audit('admin_signup', status='failed', username=username or 'anonymous', details='missing_credentials')
                return _render_admin_login_view(creds, error_text='Provide username and password.', institute_name=institute)
            if service_is_gold_username(username):
                record_audit('admin_signup', status='failed', username=username, details='reserved_gold_username')
                return _render_admin_login_view(creds, error_text='Username is reserved.', institute_name=institute)

            admins = [
                a for a in (service_load_admins() or [])
                if not service_is_gold_username(a.get('username'))
            ]
            if any(str(a.get('username') or '') == username for a in admins):
                record_audit('admin_signup', status='failed', username=username, details='username_exists')
                return _render_admin_login_view(creds, error_text='Admin username already exists.', institute_name=institute)

            admins.append({
                'username': username,
                'display_name': username,
                'password_hash': generate_password_hash(password),
                'role': signup_role
            })
            service_save_admins(admins)
            creds['admins'] = service_load_admins()
            creds['institute_name'] = institute or creds.get('institute_name', 'INSTITUTE')
            save_credentials(creds)
            session['admin'] = username
            session['admin_authenticated'] = True
            _clear_pending_admin_pin()
            touch_active_admin_session(username)
            record_audit('admin_signup', status='success', username=username, details=f'signup_created role={signup_role}')
            return redirect(url_for('admin_view'))

        if action == 'verify_gold_pin':
            return admin_verify_pin()

        if action != 'login':
            record_audit('admin_login', status='failed', username=username or 'anonymous', details='invalid_action')
            return _render_admin_login_view(creds, error_text='Invalid action.', institute_name=institute)

        blocked, retry_after = _is_login_rate_limited(login_ip, now_ts)
        if blocked:
            record_audit('admin_login', status='failed', username=username or 'anonymous', details=f'rate_limited retry_after={retry_after}s')
            return _render_admin_login_view(
                creds,
                error_text=f'Too many login attempts. Try again in {retry_after}s.',
                institute_name=institute
            )

        login_result = service_validate_login(username, password)
        if not login_result.get('ok'):
            _record_login_attempt(login_ip, False, now_ts)
            record_audit('admin_login', status='failed', username=username or 'anonymous', details=login_result.get('error') or 'invalid_login')
            return _render_admin_login_view(
                creds,
                error_text=login_result.get('message') or 'Invalid username or password.',
                institute_name=institute
            )

        admin = login_result.get('admin') or {}
        role = get_admin_role(admin)
        session.pop('admin', None)
        session.pop('admin_authenticated', None)
        session[PENDING_ADMIN_PIN_KEY] = {
            'username': admin.get('username'),
            'role': role,
            'created_at': now_ts
        }
        record_audit('admin_login', status='pending', username=admin.get('username') or 'anonymous', details=f'password_ok role={role}')
        return _render_admin_login_view(
            creds,
            require_pin=True,
            pending_username=admin.get('username'),
            institute_name=institute or creds.get('institute_name', 'INSTITUTE')
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        record_audit('admin_login', status='error', username=(request.form.get('username', '').strip() if request and request.form else 'anonymous'), details='server_error')
        return f"Server Error: {str(e)}", 500

@app.route('/admin/login/pin', methods=['POST'])
def admin_verify_pin():
    creds = load_credentials()
    pending = _get_pending_admin_pin(creds)
    username = (pending or {}).get('username') or 'anonymous'
    login_ip = _client_ip()
    now_ts = int(time.time())

    if not pending:
        record_audit('admin_login', status='failed', username='anonymous', details='missing_or_expired_pin_session')
        if request.is_json:
            return jsonify({'status': 'error', 'error': 'Pin session expired. Login again.'}), 400
        return _render_admin_login_view(
            creds,
            error_text='PIN session expired. Enter username and password again.',
            require_pin=False,
            institute_name=creds.get('institute_name', 'INSTITUTE')
        )

    blocked, retry_after = _is_login_rate_limited(login_ip, now_ts)
    if blocked:
        record_audit('admin_login', status='failed', username=username, details=f'rate_limited_pin retry_after={retry_after}s')
        if request.is_json:
            return jsonify({'status': 'error', 'error': f'Too many attempts. Retry in {retry_after}s.'}), 429
        return _render_admin_login_view(
            creds,
            error_text=f'Too many attempts. Retry in {retry_after}s.',
            require_pin=True,
            pending_username=username,
            clear_pin_on_error=True
        )

    pin = ''
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        pin = str(payload.get('pin') or '').strip()
    else:
        pin = str(request.form.get('gold_login_pin') or '').strip()

    pin_result = service_validate_pin(pending.get('role'), pin, get_pin_config(creds))
    if not pin_result.get('ok'):
        _record_login_attempt(login_ip, False, now_ts)
        record_audit('admin_login', status='failed', username=username, details=pin_result.get('error') or 'invalid_pin')
        if request.is_json:
            return jsonify({'status': 'error', 'error': pin_result.get('message') or 'Invalid login pin.'}), 400
        return _render_admin_login_view(
            creds,
            error_text=pin_result.get('message') or 'Invalid login pin.',
            require_pin=True,
            pending_username=username,
            clear_pin_on_error=True
        )

    _record_login_attempt(login_ip, True, now_ts)
    _clear_pending_admin_pin()
    session['admin'] = username
    session['admin_authenticated'] = True
    touch_active_admin_session(username)
    record_audit('admin_login', status='success', username=username, details=f'login_ok role={pending.get("role")}')
    if request.is_json:
        return jsonify({'status': 'success', 'redirect': url_for('admin_view')})
    return redirect(url_for('admin_view'))

@app.route('/admin/logout')
def admin_logout():
    username = session.get('admin') or 'anonymous'
    remove_active_admin_session()
    session.pop('admin', None)
    session.pop('admin_authenticated', None)
    _clear_pending_admin_pin()
    record_audit('admin_logout', status='success', username=username, details='logout')
    return redirect(url_for('admin_login'))

# ---------- admin user management ----------
@app.route('/admin/users')
@login_required
def admin_users():
    creds = load_credentials()
    is_gold = current_admin_is_gold(creds)
    users = []
    for adm in creds.get('admins', []):
        entry = {
            'type': 'Admin',
            'username': adm.get('username', ''),
            'password': '************',
            'role': get_admin_role(adm) if is_gold else ADMIN_ROLE_STANDARD,
            'display_name': adm.get('display_name') or adm.get('username', ''),
        }
        users.append(entry)
    for s in creds.get('students', []):
        users.append({'type': 'Student', 'username': s.get('username', ''), 'password': '************'})
    return jsonify({'users': users, 'is_gold_admin': is_gold})

@app.route('/admin/admins', methods=['GET'])
@login_required
def list_admins():
    creds = load_credentials()
    is_gold = current_admin_is_gold(creds)
    admins_payload = []
    for admin in creds.get('admins', []):
        row = {
            'username': admin.get('username', ''),
            'display_name': admin.get('display_name') or admin.get('username', ''),
            'role': get_admin_role(admin) if is_gold else ADMIN_ROLE_STANDARD
        }
        admins_payload.append(row)
    return jsonify({
        'admins': admins_payload,
        'current_admin': session.get('admin'),
        'is_gold_admin': is_gold
    })

@app.route('/admin/admins', methods=['POST'])
@login_required
def add_admin():
    data = request.json or {}
    username = (data.get('username', '') or '').strip()
    display_name = (data.get('display_name', '') or '').strip()
    password = (data.get('password', '') or '').strip()
    pin = (data.get('pin', '') or '').strip()
    actor = session.get('admin') or 'anonymous'
    creds = load_credentials()
    if not current_admin_is_gold(creds):
        record_audit('admin_add', status='failed', username=actor, details=f'forbidden_non_gold target={username or "-"}')
        return jsonify(access_denied_error()), 403
    target_role = role_from_signup_pin(pin, creds)
    if not target_role:
        record_audit('admin_add', status='failed', username=session.get('admin') or 'anonymous', details=f'invalid_pin target={username or "-"}')
        return jsonify({'error': 'Invalid pin'}), 400
    if not username or not password:
        record_audit('admin_add', status='failed', username=session.get('admin') or 'anonymous', details='missing_credentials')
        return jsonify({'error': 'Provide username and password'}), 400
    if service_is_gold_username(username):
        record_audit('admin_add', status='failed', username=actor, details=f'reserved_gold_username target={username}')
        return jsonify({'error': 'Username is reserved'}), 400
    if any(a.get('username') == username for a in creds.get('admins', [])):
        record_audit('admin_add', status='failed', username=session.get('admin') or 'anonymous', details=f'username_exists target={username}')
        return jsonify({'error': 'Admin username already exists'}), 400
    admins = [
        a for a in creds.get('admins', [])
        if not service_is_gold_username(a.get('username'))
    ]
    admins.append({
        'username': username,
        'display_name': display_name or username,
        'password_hash': generate_password_hash(password),
        'role': target_role
    })
    service_save_admins(admins)
    record_audit('admin_add', status='success', username=session.get('admin') or 'anonymous', details=f'target={username} role={target_role}')
    return jsonify({'status': 'success', 'username': username, 'role': target_role})

@app.route('/admin/admins/<username>', methods=['DELETE'])
@login_required
def delete_admin(username):
    actor = session.get('admin') or 'anonymous'
    creds = load_credentials()
    if not current_admin_is_gold(creds):
        record_audit('admin_delete', status='failed', username=actor, details=f'forbidden_non_gold target={username}')
        return jsonify(access_denied_error()), 403

    if service_is_gold_username(username):
        record_audit('admin_delete', status='failed', username=actor, details=f'blocked_fixed_gold target={username}')
        return jsonify({'error': 'Gold admin is fixed and cannot be deleted'}), 400

    admins = creds.get('admins', [])
    target = next((a for a in admins if a.get('username') == username), None)
    if not target:
        record_audit('admin_delete', status='failed', username=actor, details=f'not_found target={username}')
        return jsonify({'error': 'Admin not found'}), 404

    if get_admin_role(target) == ADMIN_ROLE_GOLD:
        gold_admins = [a for a in admins if get_admin_role(a) == ADMIN_ROLE_GOLD]
        if len(gold_admins) <= 1:
            record_audit('admin_delete', status='failed', username=actor, details=f'blocked_last_gold target={username}')
            return jsonify({'error': 'Cannot delete the last gold admin'}), 400

    new = [
        a for a in admins
        if a.get('username') != username and not service_is_gold_username(a.get('username'))
    ]
    service_save_admins(new)
    if session.get('admin') == username:
        remove_active_admin_session()
        session.pop('admin', None)
        session.pop('admin_authenticated', None)
    record_audit('admin_delete', status='success', username=actor, details=f'target={username}')
    return jsonify({'status': 'success'})

@app.route('/admin/admins/<username>/password', methods=['POST'])
@login_required
def change_admin_password(username):
    data = request.json or {}
    new_pw = (data.get('password', '') or '').strip()
    pin = (data.get('pin', '') or '').strip()
    actor = session.get('admin') or 'anonymous'
    creds = load_credentials()
    if not current_admin_is_gold(creds):
        record_audit('admin_password_change', status='failed', username=actor, details=f'forbidden_non_gold target={username}')
        return jsonify(access_denied_error()), 403
    if service_is_gold_username(username):
        record_audit('admin_password_change', status='failed', username=actor, details=f'blocked_fixed_gold target={username}')
        return jsonify({'error': 'Gold admin password is managed via environment and cannot be edited here'}), 400
    if not new_pw:
        record_audit('admin_password_change', status='failed', username=session.get('admin') or 'anonymous', details=f'missing_password target={username}')
        return jsonify({'error': 'Provide new password'}), 400
    admin = next((a for a in creds.get('admins', []) if a.get('username') == username), None)
    if not admin:
        record_audit('admin_password_change', status='failed', username=session.get('admin') or 'anonymous', details=f'not_found target={username}')
        return jsonify({'error': 'Admin not found'}), 404

    required_pin = required_login_pin_for_role(get_admin_role(admin), creds)
    if pin != required_pin:
        record_audit('admin_password_change', status='failed', username=actor, details=f'invalid_pin target={username}')
        return jsonify({'error': 'Invalid pin'}), 400

    admin['password_hash'] = generate_password_hash(new_pw)
    admins = [
        a for a in creds.get('admins', [])
        if not service_is_gold_username(a.get('username'))
    ]
    service_save_admins(admins)
    record_audit('admin_password_change', status='success', username=session.get('admin') or 'anonymous', details=f'target={username}')
    return jsonify({'status': 'success'})

@app.route('/admin/pins', methods=['GET'])
@login_required
def get_admin_pins():
    creds = load_credentials()
    actor = session.get('admin') or 'anonymous'
    if not current_admin_is_gold(creds):
        record_audit('admin_pin_view', status='failed', username=actor, details='forbidden_non_gold')
        return jsonify(access_denied_error()), 403
    return jsonify({'pins': get_pin_config(creds)})

@app.route('/admin/pins', methods=['POST'])
@login_required
def update_admin_pins():
    creds = load_credentials()
    actor = session.get('admin') or 'anonymous'
    if not current_admin_is_gold(creds):
        record_audit('admin_pin_update', status='failed', username=actor, details='forbidden_non_gold')
        return jsonify(access_denied_error()), 403

    data = request.json or {}
    existing = get_pin_config(creds)
    updated = dict(existing)
    changed_keys = []
    for key in PIN_KEYS:
        if key not in data:
            continue
        pin_val = str(data.get(key) or '').strip()
        if not (pin_val.isdigit() and len(pin_val) == 6):
            record_audit('admin_pin_update', status='failed', username=actor, details=f'invalid_pin_format key={key}')
            return jsonify({'error': f'{key} must be a 6-digit numeric pin'}), 400
        if updated.get(key) != pin_val:
            updated[key] = pin_val
            changed_keys.append(key)

    if not changed_keys:
        return jsonify({'status': 'noop', 'pins': existing})

    if updated.get(PIN_ADMIN_SIGNUP) == updated.get(PIN_GOLD_SIGNUP):
        record_audit('admin_pin_update', status='failed', username=actor, details='signup_pin_collision')
        return jsonify({'error': 'Admin signup pin and gold signup pin must be different'}), 400

    creds['pins'] = updated
    save_credentials(creds)
    record_audit('admin_pin_update', status='success', username=actor, details=f'updated={",".join(changed_keys)}')
    return jsonify({'status': 'success', 'pins': updated, 'updated_keys': changed_keys})

@app.route('/api/ui-theme', methods=['GET'])
def get_ui_theme_settings():
    creds = load_credentials()
    return jsonify(get_ui_theme(creds))

@app.route('/admin/ui-theme', methods=['POST'])
@login_required
def update_ui_theme_settings():
    data = request.json or {}
    requested_theme = {
        'accent_color': data.get('accent_color'),
        'saturation': data.get('saturation'),
    }
    sanitized = sanitize_ui_theme(requested_theme)
    creds = load_credentials()
    previous = get_ui_theme(creds)
    if previous == sanitized:
        return jsonify({'status': 'noop', 'ui_theme': previous})
    creds['ui_theme'] = sanitized
    save_credentials(creds)
    record_audit(
        'admin_ui_theme_update',
        status='success',
        username=session.get('admin') or 'anonymous',
        details=f"accent={sanitized.get('accent_color')} saturation={sanitized.get('saturation')}"
    )
    return jsonify({'status': 'success', 'ui_theme': sanitized})

@app.route('/admin/route-snap-settings', methods=['POST'])
@login_required
def update_route_snap_settings():
    data = request.json or {}
    sanitized = sanitize_route_snap_settings(data)
    creds = load_credentials()
    previous = get_route_snap_settings(creds)
    if previous == sanitized:
        return jsonify({'status': 'noop', 'route_snap_settings': previous})
    creds['route_snap_settings'] = sanitized
    save_credentials(creds)
    record_audit(
        'admin_route_snap_settings_update',
        status='success',
        username=session.get('admin') or 'anonymous',
        details=f"enabled={sanitized.get('enabled')} distance_m={sanitized.get('distance_m')} show_range={sanitized.get('show_range')}"
    )
    return jsonify({'status': 'success', 'route_snap_settings': sanitized})

# ---------- metrics ----------
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    creds = load_credentials()
    return jsonify({'total_transports': int(creds.get('total_transports', 100))})

@app.route('/api/metrics', methods=['POST'])
@login_required
def update_metrics():
    data = request.json or {}
    try:
        total = int(data.get('total_transports'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid total_transports'}), 400
    if total < 0:
        return jsonify({'error': 'Provide non-negative total_transports'}), 400
    creds = load_credentials()
    creds['total_transports'] = total
    save_credentials(creds)
    return jsonify({'status': 'success', 'total_transports': total})

@app.route('/admin/performance', methods=['GET'])
@login_required
def admin_performance():
    locs = get_locations_readonly()
    creds = load_credentials()
    is_gold = current_admin_is_gold(creds)
    uptime_sec = int(time.time() - APP_START_TS)
    with _buses_lock:
        buses_count = len(_buses)
    with _subscribers_lock:
        sse_clients = sum(len(v) for v in _subscribers.values())
    active_students = get_active_student_count()
    # Keep compatibility with existing bus-based driver tracking while adding presence tracking.
    active_drivers = max(get_active_driver_count(), buses_count)
    with _audit_lock:
        pruned_logs = prune_audit_logs(_audit_logs)
        if len(pruned_logs) != len(_audit_logs):
            _audit_logs[:] = pruned_logs
            save_json(AUDIT_FILE, _audit_logs)
        all_logs = list(_audit_logs)
    memory_stats = get_system_memory_stats()
    cpu_stats = get_process_cpu_stats()
    storage_stats = get_storage_stats(BASE_DIR)
    disk_stats = get_process_disk_io_stats()
    process_rss_mb = get_process_rss_mb()
    website_used_mb = memory_stats.get('used_mb')
    if website_used_mb is None:
        website_used_mb = process_rss_mb
    if is_gold:
        success_events = [e for e in all_logs if e.get('event') in ('admin_login', 'admin_signup') and e.get('status') == 'success']
        failed_events = [e for e in all_logs if e.get('event') in ('admin_login', 'admin_signup') and e.get('status') != 'success']
        recent_audit = list(reversed(all_logs[-120:]))
        successful_logins = len(success_events)
        failed_logins = len(failed_events)
        last_success_login = success_events[-1]['ts'] if success_events else None
        last_failed_login = failed_events[-1]['ts'] if failed_events else None
    else:
        successful_logins = None
        failed_logins = None
        last_success_login = None
        last_failed_login = None
        recent_audit = []
    with _metrics_lock:
        requests_total = REQUESTS_TOTAL
        bandwidth_in_bytes = BANDWIDTH_IN_BYTES
        bandwidth_out_bytes = BANDWIDTH_OUT_BYTES
    uptime_for_rate = max(1, uptime_sec)
    return jsonify({
        'server_time': _utc_now_iso(),
        'uptime_sec': uptime_sec,
        'memory': {
            'process_rss_mb': process_rss_mb,
            'website_used_mb': website_used_mb,
            'system_total_mb': memory_stats.get('total_mb'),
            'system_used_mb': memory_stats.get('used_mb'),
            'system_available_mb': memory_stats.get('available_mb'),
            'source': memory_stats.get('source')
        },
        'storage': storage_stats,
        'cpu': cpu_stats,
        'disk': {
            'process_read_bytes_total': disk_stats.get('read_bytes_total'),
            'process_write_bytes_total': disk_stats.get('write_bytes_total'),
            'process_read_kbps': disk_stats.get('read_kbps'),
            'process_write_kbps': disk_stats.get('write_kbps'),
            'sample_window_sec': disk_stats.get('sample_window_sec')
        },
        'bandwidth': {
            'in_bytes': bandwidth_in_bytes,
            'out_bytes': bandwidth_out_bytes,
            'in_mb': round(bandwidth_in_bytes / (1024 * 1024), 2),
            'out_mb': round(bandwidth_out_bytes / (1024 * 1024), 2),
            'avg_in_kbps': round(((bandwidth_in_bytes * 8) / 1000) / uptime_for_rate, 2),
            'avg_out_kbps': round(((bandwidth_out_bytes * 8) / 1000) / uptime_for_rate, 2)
        },
        'requests_total': requests_total,
        'sse_clients': sse_clients,
        'buses_count': buses_count,
        'active_students': active_students,
        'active_drivers': active_drivers,
        'active_admins': get_active_admin_count(),
        'routes_count': len(locs.get('routes', [])),
        'hostels_count': len(locs.get('hostels', [])),
        'classes_count': len(locs.get('classes', [])),
        'admin': {
            'current_admin': session.get('admin'),
            'is_gold_admin': is_gold,
            'admins_count': len(creds.get('admins', [])),
            'successful_logins': successful_logins,
            'failed_logins': failed_logins,
            'last_success_login': last_success_login,
            'last_failed_login': last_failed_login
        },
        'audit_logs': recent_audit
    })

@app.route('/admin/performance/export', methods=['GET'])
@login_required
def admin_performance_export():
    creds = load_credentials()
    if not current_admin_is_gold(creds):
        record_audit('admin_performance_export', status='failed', username=session.get('admin') or 'anonymous', details='forbidden_non_gold')
        return jsonify(access_denied_error()), 403
    export_format = str(request.args.get('format', 'md') or 'md').strip().lower()
    if export_format not in ('md', 'txt'):
        return jsonify({'error': 'format must be one of: md, txt'}), 400
    with _audit_lock:
        _audit_logs[:] = prune_audit_logs(_audit_logs)
        logs = list(_audit_logs)
        save_json(AUDIT_FILE, _audit_logs)
    generated_ts = _utc_now_iso()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    if export_format == 'md':
        lines = [
            '# Admin Audit Log Export',
            '',
            f'Generated (UTC): {generated_ts}',
            f'Entries: {len(logs)}',
            '',
            '| Time (UTC) | User | Event | Status | IP | Details |',
            '|---|---|---|---|---|---|'
        ]
        for entry in logs:
            ts = str(entry.get('ts') or '--').replace('\n', ' ').replace('\r', ' ')
            user = str(entry.get('username') or '--').replace('\n', ' ').replace('\r', ' ')
            event = str(entry.get('event') or '--').replace('\n', ' ').replace('\r', ' ')
            status = str(entry.get('status') or '--').replace('\n', ' ').replace('\r', ' ')
            ip = str(entry.get('ip') or '--').replace('\n', ' ').replace('\r', ' ')
            details = str(entry.get('details') or '--').replace('\n', ' ').replace('\r', ' ')
            ts_md = ts.replace('|', '\\|')
            user_md = user.replace('|', '\\|')
            event_md = event.replace('|', '\\|')
            status_md = status.replace('|', '\\|')
            ip_md = ip.replace('|', '\\|')
            details_md = details.replace('|', '\\|')
            lines.append(
                f'| {ts_md} | {user_md} | '
                f'{event_md} | {status_md} | '
                f'{ip_md} | {details_md} |'
            )
        body = '\n'.join(lines) + '\n'
        mimetype = 'text/markdown'
        ext = 'md'
    else:
        lines = [
            'Admin Audit Log Export',
            f'Generated (UTC): {generated_ts}',
            f'Entries: {len(logs)}',
            ''
        ]
        for entry in logs:
            ts = str(entry.get('ts') or '--').replace('\n', ' ').replace('\r', ' ')
            user = str(entry.get('username') or '--').replace('\n', ' ').replace('\r', ' ')
            event = str(entry.get('event') or '--').replace('\n', ' ').replace('\r', ' ')
            status = str(entry.get('status') or '--').replace('\n', ' ').replace('\r', ' ')
            ip = str(entry.get('ip') or '--').replace('\n', ' ').replace('\r', ' ')
            details = str(entry.get('details') or '--').replace('\n', ' ').replace('\r', ' ')
            lines.append(f'{ts} | {user} | {event} | {status} | {ip} | {details}')
        body = '\n'.join(lines) + '\n'
        mimetype = 'text/plain'
        ext = 'txt'
    resp = Response(body, mimetype=mimetype)
    resp.headers['Content-Disposition'] = f'attachment; filename=\"admin-audit-{stamp}.{ext}\"'
    return resp

@app.route('/admin/console/activity', methods=['GET'])
@login_required
def admin_console_activity():
    creds = load_credentials()
    actor = session.get('admin') or 'anonymous'
    if not current_admin_is_gold(creds):
        record_audit('admin_console_activity', status='failed', username=actor, details='forbidden_non_gold')
        return jsonify(access_denied_error()), 403

    try:
        limit = int(request.args.get('limit', 250) or 250)
    except (TypeError, ValueError):
        limit = 250
    limit = max(20, min(limit, 800))

    with _audit_lock:
        pruned_logs = prune_audit_logs(_audit_logs)
        if len(pruned_logs) != len(_audit_logs):
            _audit_logs[:] = pruned_logs
            save_json(AUDIT_FILE, _audit_logs)
        all_logs = list(_audit_logs)

    admin_activity = [e for e in all_logs if is_admin_activity_log(e)]
    success_count = sum(1 for e in admin_activity if str(e.get('status') or '').lower() in ('success', 'ok'))
    failed_count = len(admin_activity) - success_count
    recent_logs = list(reversed(admin_activity[-limit:]))

    record_audit(
        'admin_console_activity',
        status='success',
        username=actor,
        details=f'limit={limit} returned={len(recent_logs)} total={len(admin_activity)}'
    )
    return jsonify({
        'total_activity': len(admin_activity),
        'returned': len(recent_logs),
        'success_count': success_count,
        'failed_count': failed_count,
        'activity_logs': recent_logs
    })

@app.route('/admin/console/activity/export', methods=['GET'])
@login_required
def admin_console_activity_export():
    creds = load_credentials()
    actor = session.get('admin') or 'anonymous'
    if not current_admin_is_gold(creds):
        record_audit('admin_console_activity_export', status='failed', username=actor, details='forbidden_non_gold')
        return jsonify(access_denied_error()), 403

    export_format = str(request.args.get('format', 'txt') or 'txt').strip().lower()
    if export_format not in ('txt', 'json'):
        return jsonify({'error': 'format must be one of: txt, json'}), 400

    with _audit_lock:
        pruned_logs = prune_audit_logs(_audit_logs)
        if len(pruned_logs) != len(_audit_logs):
            _audit_logs[:] = pruned_logs
            save_json(AUDIT_FILE, _audit_logs)
        all_logs = list(_audit_logs)

    admin_activity = [e for e in all_logs if is_admin_activity_log(e)]
    generated_ts = _utc_now_iso()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')

    if export_format == 'json':
        payload = {
            'generated_utc': generated_ts,
            'entries': len(admin_activity),
            'activity_logs': admin_activity
        }
        body = json.dumps(payload, indent=2)
        mimetype = 'application/json'
        ext = 'json'
    else:
        lines = [
            'Admin Console Activity Export',
            f'Generated (UTC): {generated_ts}',
            f'Entries: {len(admin_activity)}',
            ''
        ]
        for entry in admin_activity:
            ts = str(entry.get('ts') or '--').replace('\n', ' ').replace('\r', ' ')
            user = str(entry.get('username') or '--').replace('\n', ' ').replace('\r', ' ')
            event = str(entry.get('event') or '--').replace('\n', ' ').replace('\r', ' ')
            status = str(entry.get('status') or '--').replace('\n', ' ').replace('\r', ' ')
            ip = str(entry.get('ip') or '--').replace('\n', ' ').replace('\r', ' ')
            details = str(entry.get('details') or '--').replace('\n', ' ').replace('\r', ' ')
            lines.append(f'{ts} | {user} | {event} | {status} | {ip} | {details}')
        body = '\n'.join(lines) + '\n'
        mimetype = 'text/plain'
        ext = 'txt'

    record_audit(
        'admin_console_activity_export',
        status='success',
        username=actor,
        details=f'format={export_format} entries={len(admin_activity)}'
    )
    resp = Response(body, mimetype=mimetype)
    resp.headers['Content-Disposition'] = f'attachment; filename=\"admin-console-activity-{stamp}.{ext}\"'
    return resp

@app.route('/api/presence/student', methods=['POST'])
def update_student_presence():
    payload = request.get_json(silent=True) or request.form.to_dict() or {}
    client_id = payload.get('clientId') or payload.get('viewerId') or request.headers.get('X-Client-Id')
    normalized = _normalize_presence_id(client_id, 'student')
    if not normalized:
        return jsonify({'error': 'clientId is required'}), 400
    active = _to_bool(payload.get('active'), True)
    if active:
        touch_student_presence(normalized)
    else:
        remove_student_presence(normalized)
    return jsonify({'status': 'success', 'active_students': get_active_student_count()})

# ---------- bus APIs ----------
@app.route('/api/buses', methods=['GET'])
def get_all_buses():
    # Enrich with latest stop state for polling fallback
    with _buses_lock:
        if not _buses:
            return jsonify({})
        result = {}
        for bus_id, data in _buses.items():
            entry = dict(data)
            ss = _bus_stop_state.get(bus_id, {})
            entry['atStop'] = ss.get('atStop')
            entry['nearestStopName'] = ss.get('nearestStopName') if 'nearestStopName' not in entry else entry['nearestStopName']
            entry['direction'] = ss.get('direction')
            entry['terminalState'] = ss.get('terminalState')
            result[bus_id] = entry
    return jsonify(result)

@app.route('/api/buses/clear', methods=['POST'])
def clear_all_buses():
    global _buses, _buses_dirty
    removed_ids = []
    with _buses_lock:
        removed_ids = list(_buses.keys())
        _buses = {}
        _bus_destination_ts.clear()
        _bus_last_broadcast.clear()
        _bus_stop_state.clear()
        _kalman_filters.clear()
        _buses_dirty = True
    for bus_id in removed_ids:
        remove_driver_presence_for_bus(bus_id)
    save_json(BUSES_FILE, {})
    try:
        broadcast({'type': 'buses_clear'})
    except Exception:
        pass
    actor = session.get('admin') if has_request_context() else 'system'
    preview = ','.join(removed_ids[:10])
    if len(removed_ids) > 10:
        preview += ',...'
    record_audit(
        'Deleted all transports',
        status='success',
        username=actor or 'anonymous',
        details=f'count={len(removed_ids)} ids={preview or "-"}'
    )
    return jsonify({'status': 'success'})

_bus_last_broadcast = {}  # bus_id -> monotonic timestamp of last broadcast

@app.route('/api/bus/<int:bus_number>', methods=['POST'])
def update_bus_location(bus_number):
    global _buses_dirty
    raw = request.get_json(silent=True) or request.form.to_dict() or {}
    try:
        raw_lat = float(raw.get('lat'))
        raw_lng = float(raw.get('lng'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Provide numeric lat and lng'}), 400
    # Kalman smoothing
    bus_id = str(bus_number)
    touch_driver_presence(_driver_bus_presence_key(bus_id))
    lat, lng = kalman_smooth(bus_id, raw_lat, raw_lng)
    parsed_last_update = parse_iso_timestamp(raw.get('lastUpdate'))
    if parsed_last_update is None:
        last_update = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    else:
        last_update = datetime.fromtimestamp(parsed_last_update, timezone.utc).isoformat().replace('+00:00', 'Z')
    heading = raw.get('heading')  # device heading from driver/simulator
    should_broadcast = True
    now_mono = time.monotonic()
    with _buses_lock:
        existing = _buses.get(bus_id, {})
        route_id = raw.get('routeId', existing.get('routeId'))
        if existing.get('routeId') != route_id:
            _bus_destination_ts.pop(bus_id, None)
        # Skip broadcast if position hasn't meaningfully changed (~0.5m)
        # Always broadcast at least every 3s even when stationary (heartbeat)
        if existing and 'lat' in existing and 'lng' in existing:
            dlat = abs(lat - existing['lat'])
            dlng = abs(lng - existing['lng'])
            if dlat < 0.000005 and dlng < 0.000005:
                last_bc = _bus_last_broadcast.get(bus_id, 0)
                if (now_mono - last_bc) < 3:
                    should_broadcast = False
        entry = {'lat': lat, 'lng': lng, 'lastUpdate': last_update, 'routeId': route_id}
        if heading is not None:
            try:
                entry['heading'] = float(heading)
            except (TypeError, ValueError):
                pass
        _buses[bus_id] = entry
        _buses_dirty = True
        current_data = dict(entry)
    # Server-side stop detection — enrich broadcast payload
    stop_info = detect_stop_info(bus_id, lat, lng, route_id)
    current_data.update(stop_info)
    with _buses_lock:
        if stop_info.get('terminalState') == 'at_destination':
            _bus_destination_ts.setdefault(bus_id, now_mono)
            current_data['removeInSec'] = max(0, round(DESTINATION_REMOVE_SEC - (now_mono - _bus_destination_ts[bus_id]), 2))
        else:
            _bus_destination_ts.pop(bus_id, None)
    try:
        if should_broadcast:
            _bus_last_broadcast[bus_id] = now_mono
            broadcast({'type': 'bus_update', 'bus': bus_id, 'data': current_data})
    except Exception:
        pass
    return jsonify({'status': 'success', 'bus': bus_number})

@app.route('/api/bus/<int:bus_number>', methods=['DELETE'])
def stop_bus(bus_number):
    global _buses_dirty
    bus_id = str(bus_number)
    with _buses_lock:
        removed = _buses.pop(bus_id, None)
        if removed:
            _buses_dirty = True
    route_id = removed.get('routeId') if removed else None
    # Clean up Kalman + stop state
    _kalman_filters.pop(bus_id, None)
    _bus_stop_state.pop(bus_id, None)
    _bus_last_broadcast.pop(bus_id, None)
    _bus_destination_ts.pop(bus_id, None)
    remove_driver_presence_for_bus(bus_id)
    try:
        broadcast({'type': 'bus_stop', 'bus': bus_id, 'routeId': route_id})
    except Exception:
        pass
    actor = session.get('admin') if has_request_context() else 'system'
    if removed:
        record_audit(
            'Deleted transport',
            status='success',
            username=actor or 'anonymous',
            details=f'bus={bus_id} route={route_id or "-"} lat={removed.get("lat")} lng={removed.get("lng")}'
        )
    else:
        record_audit(
            'Deleted transport',
            status='failed',
            username=actor or 'anonymous',
            details=f'bus={bus_id} reason=not_found'
        )
    return jsonify({'status': 'success'})

@app.route('/api/bus/<int:bus_number>/route', methods=['POST'])
def set_bus_route(bus_number):
    global _buses_dirty
    data = request.get_json(silent=True) or {}
    route_id = data.get('routeId')
    bus_id = str(bus_number)
    with _buses_lock:
        if bus_id in _buses:
            if _buses[bus_id].get('routeId') != route_id:
                _buses[bus_id]['routeId'] = route_id
                _buses_dirty = True
            _bus_destination_ts.pop(bus_id, None)
        # Don't create a bus entry just for route assignment
    try:
        broadcast({'type': 'route_set', 'bus': bus_id, 'routeId': route_id})
    except Exception:
        pass
    return jsonify({'status': 'success'})

@app.route('/api/bus-routes', methods=['GET'])
def get_bus_routes():
    with _buses_lock:
        result = {k: v.get('routeId') for k, v in _buses.items()}
    return jsonify(result)

# ---------- location APIs ----------
@app.route('/api/locations', methods=['GET'])
def get_locations():
    return jsonify(get_locations_readonly())

@app.route('/api/hostels', methods=['GET'])
def get_hostels():
    locs = get_locations_readonly()
    return jsonify(locs.get('hostels', []))

@app.route('/api/classes', methods=['GET'])
def get_classes():
    locs = get_locations_readonly()
    return jsonify(locs.get('classes', []))

@app.route('/api/routes', methods=['GET'])
def get_routes():
    locs = get_locations_readonly()
    return jsonify(locs.get('routes', []))

@app.route('/api/route-snap-settings', methods=['GET'])
def get_route_snap_settings_api():
    creds = load_credentials()
    return jsonify(get_route_snap_settings(creds))

@app.route('/api/route', methods=['POST'])
@login_required
def create_route():
    data = request.json or {}
    locs = get_locations_for_update()
    waypoints = sanitize_waypoint_list(data.get('waypoints'))
    if len(waypoints) < 2:
        return jsonify({'error': 'Route requires at least 2 valid waypoints'}), 400

    path_points = sanitize_waypoint_list(data.get('path_points'))
    if len(path_points) < 2:
        path_points = [list(wp) for wp in waypoints]

    stops_raw = data.get('stops', [])
    stops = []
    if isinstance(stops_raw, list):
        for idx in range(min(len(stops_raw), len(waypoints))):
            v = stops_raw[idx]
            stops.append(str(v).strip() if v is not None else '')
    while len(stops) < len(waypoints):
        stops.append('')

    route_id = str(data.get('id') or f"route_{len(locs.get('routes', [])) + 1}")
    route_name = str(data.get('name') or '').strip()
    if not route_name:
        return jsonify({'error': 'Route name is required'}), 400
    color = str(data.get('color') or '#FF5722').strip() or '#FF5722'
    follow_roads_input = _to_bool(data.get('follow_roads'), False)
    follow_roads_segments = sanitize_follow_road_segments(data.get('follow_roads_segments'), len(waypoints) - 1, follow_roads_input)
    follow_roads = any(follow_roads_segments)
    creds = load_credentials()
    global_snap = get_route_snap_settings(creds)
    route_snap = sanitize_route_snap_override(data.get('snap_settings'), global_snap)

    route = {
        'id': route_id,
        'name': route_name,
        'waypoints': waypoints,
        'path_points': path_points,
        'stops': stops,
        'color': color,
        'follow_roads': follow_roads,
        'follow_roads_segments': follow_roads_segments,
        'snap_settings': route_snap,
    }
    routes = locs.get('routes', [])
    idx = next((i for i, r in enumerate(routes) if str(r.get('id')) == str(route['id'])), -1)
    if idx >= 0:
        routes[idx] = route
    else:
        routes.append(route)
    locs['routes'] = routes
    if not save_locations(locs):
        return jsonify({'error': 'Failed to persist route data on server'}), 500
    return jsonify({'status': 'success', 'route': route})

@app.route('/api/route/<route_id>', methods=['DELETE'])
def delete_route(route_id):
    locs = get_locations_for_update()
    locs['routes'] = [r for r in locs.get('routes', []) if r['id'] != route_id]
    if not save_locations(locs):
        return jsonify({'error': 'Failed to persist route deletion on server'}), 500
    return jsonify({'status': 'success'})

@app.route('/api/hostel', methods=['POST'])
def create_hostel():
    data = request.json
    locs = get_locations_for_update()
    hostel = {
        'id': f"hostel_{len(locs.get('hostels', [])) + 1}",
        'name': data['name'],
        'lat': data['lat'],
        'lng': data['lng'],
        'capacity': data.get('capacity', 100)
    }
    locs['hostels'].append(hostel)
    save_locations(locs)
    return jsonify({'status': 'success', 'hostel': hostel})

@app.route('/api/hostel/<hostel_id>', methods=['DELETE'])
def delete_hostel(hostel_id):
    locs = get_locations_for_update()
    locs['hostels'] = [h for h in locs.get('hostels', []) if h['id'] != hostel_id]
    save_locations(locs)
    return jsonify({'status': 'success'})

@app.route('/api/class', methods=['POST'])
def create_class():
    data = request.json
    locs = get_locations_for_update()
    cls = {
        'id': f"class_{len(locs.get('classes', [])) + 1}",
        'name': data['name'],
        'lat': data['lat'],
        'lng': data['lng'],
        'department': data.get('department', 'Unknown')
    }
    locs['classes'].append(cls)
    save_locations(locs)
    return jsonify({'status': 'success', 'class': cls})

@app.route('/api/class/<class_id>', methods=['DELETE'])
def delete_class(class_id):
    locs = get_locations_for_update()
    locs['classes'] = [c for c in locs.get('classes', []) if c['id'] != class_id]
    save_locations(locs)
    return jsonify({'status': 'success'})

# ---------- health / status ----------
@app.route('/healthz')
def healthz():
    return jsonify({'status': 'ok', 'uptime_sec': int(time.time() - APP_START_TS)}), 200

@app.route('/status')
def status():
    locs = get_locations_readonly()
    with _subscribers_lock:
        sse_clients = sum(len(v) for v in _subscribers.values())
    with _buses_lock:
        buses_count = len(_buses)
    with _metrics_lock:
        requests_total = REQUESTS_TOTAL
    active_students = get_active_student_count()
    active_drivers = max(get_active_driver_count(), buses_count)
    return jsonify({
        'uptime_sec': int(time.time() - APP_START_TS),
        'requests_total': requests_total,
        'sse_clients': sse_clients,
        'buses_count': buses_count,
        'active_students': active_students,
        'active_drivers': active_drivers,
        'routes_count': len(locs.get('routes', [])),
        'on_render': ON_RENDER,
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
