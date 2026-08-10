import os
from pathlib import Path

from optus_ai.auth import StandardAuth


def load_env_file_if_available() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file_if_available()

client_id = os.getenv("LLM_CLIENT_ID", "").strip()
client_secret = os.getenv("LLM_CLIENT_SECRET", "").strip()
oauth_url = os.getenv("LLM_OAUTH_URL", "").strip()
verify_ssl = os.getenv("LLM_VERIFY_SSL", "").strip().lower() in {"1", "true", "yes", "on", "y"}

if not client_id:
    raise RuntimeError("LLM_CLIENT_ID is required")
if not client_secret:
    raise RuntimeError("LLM_CLIENT_SECRET is required")

auth = StandardAuth(
    client_id=client_id,
    client_secret=client_secret,
    token_endpoint=oauth_url or None,
    verify=verify_ssl,
)

token_data = auth.fetch_token()
access_token = token_data["access_token"]
masked_token = access_token[:20] + "..." if len(access_token) > 20 else access_token

print("Token fetched successfully")
print("token_type:", token_data.get("token_type", "Bearer"))
print("expires_in:", token_data.get("expires_in"))
print("access_token_prefix:", masked_token)
