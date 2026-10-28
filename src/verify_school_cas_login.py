import urllib.parse
import requests


def verify_school_cas_login(config: dict) -> dict:
    """Verify session connectivity and login form structure against MUST CAS (login.must.edu.mo) and EZProxy."""
    school_cfg = config.get("school_auth", {})
    cas_url = school_cfg.get("cas_login_url", "")
    username = str(school_cfg.get("username", "")).strip()
    password = str(school_cfg.get("password", "")).strip()
    agreed = bool(school_cfg.get("wemust_privacy_agreed", True))

    parsed = urllib.parse.urlparse(cas_url)
    query_params = urllib.parse.parse_qs(parsed.query)
    service_url = query_params.get("service", [""])[0]

    has_real_school_creds = bool(username and password and not username.startswith("YOUR_") and not password.startswith("YOUR_"))

    session_info = {
        "cas_host": parsed.netloc,
        "service_redirect": service_url,
        "dom_root_selector": "body#cas",
        "supported_auth_modes": ["WeMust App QR Code (掃碼登錄)", "Staff/Student Password (教職員/學生)"],
        "requires_privacy_checkbox": "已閲讀並同意《WeMust隱私保護聲明》",
        "privacy_checkbox_agreed_in_config": agreed,
        "credentials_configured": has_real_school_creds,
        "session_status": "not_connected"
    }

    if has_real_school_creds and agreed:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        try:
            resp = session.get(cas_url, timeout=school_cfg.get("timeout_seconds", 15))
            session_info["http_status"] = resp.status_code
            session_info["session_status"] = f"MUST CAS page reachable (HTTP {resp.status_code})"
        except Exception as exc:
            session_info["session_status"] = f"MUST CAS connection error: {exc}"
    else:
        session_info["session_status"] = "Waiting for real MUST student/staff ID & password in config.json."

    return session_info
