import os
import base64
import requests
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    client_id = os.environ.get("EBAY_CLIENT_ID", "").strip()
    client_secret = os.environ.get("EBAY_CLIENT_SECRET", "").strip()

    print(f"Checking keys... Client ID starts with: {client_id[:5] if client_id else 'MISSING'}")

    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    resp = requests.post(
        "https://api.ebay.com/identity/v1/oauth2/token",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {credentials}",
        },
        data={
            "grant_type": "client_credentials",
            "scope": "https://api.ebay.com/oauth/api_scope",
        }
    )

    print(f"eBay Response Code: {resp.status_code}")
    if resp.ok:
        # Never print the access token itself.
        print(f"Token acquired, expires in {resp.json().get('expires_in')}s")
    else:
        print(f"eBay Response Details: {resp.text}")


# Guarded so pytest (which collects test_*.py) doesn't fire a live eBay call
# on import.
if __name__ == "__main__":
    main()
