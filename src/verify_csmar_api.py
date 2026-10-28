import importlib.util
import json
import os
import platform


def verify_csmar_api(config: dict) -> dict:
    """Verify CSMAR Python API environment, fix websocket-client namespace conflict if present, and validate real token authentication."""
    school_cfg = config.get("school_auth", {})
    csmar_cfg = config.get("csmar_api", {})

    try:
        import websocket
        if not hasattr(websocket, "create_connection"):
            setattr(websocket, "create_connection", None)
    except ImportError:
        pass

    csmar_spec = importlib.util.find_spec("csmarapi")
    csmar_installed = csmar_spec is not None
    current_os = platform.system()

    acct = str(csmar_cfg.get("account", "")).strip()
    pwd = str(csmar_cfg.get("password", "")).strip()
    lang = str(csmar_cfg.get("lang", "0")).strip()
    has_real_credentials = bool(acct and pwd and not acct.startswith("YOUR_") and not pwd.startswith("YOUR_"))

    direct_api_result = {
        "can_institutional_account_call_python_api_directly": False,
        "os_platform": current_os,
        "csmarapi_package_installed": csmar_installed,
        "personal_csmar_credentials_configured": has_real_credentials,
        "institutional_gateway_url": school_cfg.get("cas_login_url", ""),
        "target_ezproxy_url": school_cfg.get("ezproxy_target_url", "https://global.csmar.com/"),
        "csmar_login_ready": False,
        "api_call_test_status": "not_attempted"
    }

    if csmar_installed and has_real_credentials:
        try:
            from csmarapi.CsmarService import CsmarService
            csmar = CsmarService()
            login_info = csmar.logon(acct, pwd, lang, "0")
            if isinstance(login_info, dict) and login_info.get("code") == 0:
                token_str = str(login_info.get("data", {}).get("token", "")).strip()
                csmar.writeToken(token_str, lang, "0")
                dbs = csmar.getListDbs()
                db_count = len(dbs) if isinstance(dbs, list) else 0
                direct_api_result["csmar_login_ready"] = bool(token_str)
                direct_api_result["api_call_test_status"] = (
                    f"SUCCESS: Logged in as {acct} (token acquired, accessible databases={db_count})."
                )
            else:
                msg = login_info.get("msg", str(login_info)) if isinstance(login_info, dict) else str(login_info)
                direct_api_result["api_call_test_status"] = f"CSMAR_LOGIN_REJECTED: {msg}"
        except Exception as exc:
            direct_api_result["api_call_test_status"] = f"FAILED: {type(exc).__name__}: {exc}"
    else:
        direct_api_result["api_call_test_status"] = (
            f"csmarapi_installed={csmar_installed}, personal_credentials_configured={has_real_credentials}."
        )

    report_path = config.get("pipeline_settings", {}).get(
        "verification_report_path", "data/api_verification_report.json"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(direct_api_result, f, indent=4, ensure_ascii=False)

    return direct_api_result
