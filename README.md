## rose-server

Flask backend that serves the compiled React app from `../rose-ui/`, proxies chat requests to an AWS Bedrock Agent, and protects endpoints with Auth0-issued JWT access tokens.

### Prerequisites

- Python 3.11+
- Built React assets in `../rose-ui/build` (`npm install && npm run build`)
- Auth0 tenant with a configured API (audience) and SPA application credentials
- AWS credentials with permission to invoke the target Bedrock Agent

### Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `AUTH0_DOMAIN` | ✅ | Auth0 domain (e.g. `your-tenant.us.auth0.com` or full `https://` URL) |
| `AUTH0_AUDIENCE` | ✅ | API audience configured in Auth0 |
| `AUTH0_CLIENT_ID` | ➖ | SPA Client ID to assert against the token azp/client_id claim (optional) |
| `BEDROCK_AGENT_ID` | ✅ | AWS Bedrock Agent identifier |
| `BEDROCK_AGENT_ALIAS_ID` | ✅ | Alias ID for the Bedrock Agent |
| `BEDROCK_REGION` | ➖ | AWS region for the Bedrock Agent (`us-east-1` default) |
| `REACT_BUILD_PATH` | ➖ | Override React build directory (defaults to `../rose-ui/build`) |
| `REQUIRE_AUTH_FOR_HEALTH` | ➖ | Set to `true` to enforce Auth0 auth on `/health/` |
| `REQUIRE_AUTH_FOR_UI` | ➖ | Set to `true` to force Auth0 auth before serving the SPA shell |

AWS credentials are resolved via the standard [boto3 credential chain](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

### Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Server

```bash
export FLASK_APP=app:create_app
flask run --host 0.0.0.0 --port 5000
```

The server:

- `GET /ping/` – lightweight readiness probe
- `GET /health/` – detailed health (optional auth)
- `POST /api/chat` – Auth0-protected proxy to the configured Bedrock Agent
- `GET /` and front-end routes – serve the SPA after verifying an Auth0 access token

### Chat Endpoint Contract

`POST /api/chat`

```jsonc
{
  "message": "User prompt",
  "sessionId": "optional-session-guid",
  "enableTrace": false
}
```

Response body:

```jsonc
{
  "sessionId": "resolved-session-id",
  "messages": [
    { "role": "assistant", "content": "..." }
  ],
  "trace": { "events": [...] } // present when enableTrace=true
}
```

### Production Notes

- Attach your Auth0-issued bearer token in the `Authorization: Bearer <token>` header for SPA and API requests.
- Configure your deployment pipeline to rebuild the React assets on release and copy them into `rose-ui/build/`.
- For container deployments, bake the build artifacts and environment variables into the image or runtime configuration.
