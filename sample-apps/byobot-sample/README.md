# Byobot

WebSocket server (WSS) for session/media and bots: echo and Dialogflow CX.

## Features

- **Session**: `session.start` → `session.started`, `session.end` → `session.ended`.
- **Bots** (via `bot.start` with `payload.botId`):
  - **`botId = "echo"`** — Echo bot (no Dialogflow).
  - **`botId = "projects/<Project ID>/locations/<Location ID>/agents/<Agent ID>"`** — Dialogflow CX (transcripts, liveAgentHandoff, transferCall).

## Quick Start (first time)

1. **Unzip** the sample into a directory and open a terminal in that directory.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the server** using one of the options below (TCP, TLS, JWT, port).

   | Goal | Command |
   |------|--------|
   | WebSocket over **TCP** (no TLS) | `python main.py` |
   | WebSocket over **TLS** (WSS) | `python main.py --ssl-cert cert/cert.pem --ssl-key cert/key.pem` |
   | WSS + **JWT** auth | `python main.py --ssl-cert cert/cert.pem --ssl-key cert/key.pem --enable-auth` |
   | Custom **port** (e.g. 9443) | `python main.py --ssl-cert cert/cert.pem --ssl-key cert/key.pem --port 9443` |
   | Combine options | `python main.py --ssl-cert cert/cert.pem --ssl-key cert/key.pem --port 9443 --enable-auth --jwt-secret-key MY_SECRET` |

   Optional: `--host`, `--log-file`, `--message-log-file`, `-v` (verbose).

## Layout

- **main.py** — Entry point; loads bot_echo and bot_service.
- **byobot_server.py** — Server (session, media, plugin registry).
- **services/bot_echo.py** — Echo plugin.
- **services/bot_dialogflow.py** — Dialogflow CX bot.
- **services/bot_service.py** — Bot plugin (dispatches by botId to echo or Dialogflow).

## Requirements

- Python 3.10+
- For WSS (TLS): SSL certificate and key (see **TLS** and **Certificate and key** below).
- For Dialogflow CX: Google Cloud credentials and a CX agent.

### TLS (optional)

The server can run with or without TLS:

- **With TLS (WSS):** Pass `--ssl-cert` and `--ssl-key`. Clients connect with `wss://`.
- **Without TLS (TCP):** Omit both options. The server listens over plain WebSocket (`ws://`). Use for local or trusted networks only.

### Certificate and key

The **`cert/`** directory in this sample contains a **self-signed** certificate and key (`cert.pem`, `key.pem`) for quick testing. For production, **replace these with a CA-signed certificate and key** (from your PKI or a public CA). The server does not validate certificate chains; it only uses the key and cert you provide for the TLS handshake.

### Using Dialogflow

The Dialogflow bot needs a **Google Cloud service account key** (JSON). You can provide it in either of these ways:

**Option A: File (recommended for local runs)**

1. In [Google Cloud Console](https://console.cloud.google.com/): IAM & Admin → Service accounts → select or create a service account with Dialogflow API access → Keys → Add key → Create new key → JSON. Download the key file.
2. In the byobot project directory, create a file named **`google_credentials.txt`**.
3. Put the **contents** of that JSON key file into `google_credentials.txt` (or copy the key file to the project and rename it to `google_credentials.txt`).

The server loads this file at startup. If it is present, `bot.start` requests that use a Dialogflow agent path can omit `botCredentials` and will use these credentials.

**Option B: Per-request (botCredentials)**

In the **`bot.start`** payload, you can send the same service account JSON as a **base64-encoded string** in the **`botCredentials`** field. The server decodes it and uses it for that bot session. This avoids storing a key file on the server.

If neither is provided, the server returns an error when starting a Dialogflow bot.

### Using JWT (optional)

The server can require a **JWT Bearer token** on the WebSocket handshake. When enabled, each connection must send a valid token in the **`Authorization`** header (e.g. `Authorization: Bearer <token>`). The server verifies the token with HS256 using the configured secret key.

**Enable JWT**

```bash
python main.py --ssl-cert cert/cert.pem --ssl-key cert/key.pem --enable-auth
```

Optional: set your own secret (otherwise a default is used):

```bash
python main.py --ssl-cert cert/cert.pem --ssl-key cert/key.pem --enable-auth --jwt-secret-key YOUR_SECRET
```

**Client side:** Build a JWT (e.g. with PyJWT or your language’s JWT library) signed with the same secret and algorithm **HS256**. Send it in the `Authorization` header when opening the WebSocket: `Bearer <base64-encoded-jwt>`.

Without `--enable-auth`, the server does not check tokens and accepts all connections.

### Install dependencies

From this directory:

```bash
pip install -r requirements.txt
```

Optional (G722 codec support):

```bash
pip install g722
```

## Run

From this directory. **TCP (no TLS):**

```bash
python main.py
```

**WSS (TLS)** using the sample cert in `cert/`:

```bash
python main.py --ssl-cert cert/cert.pem --ssl-key cert/key.pem
```

Options: `--host`, `--port`, `--log-file`, `--message-log-file`, `--enable-auth`, `--jwt-secret-key`, `-v`. See **Quick Start** for more examples.

## Protocol

1. Client sends **session.start** → server **session.started**.
2. Client sends **bot.start** with `payload.botId` and `payload.endpointId`. Server runs echo or Dialogflow CX by botId.
3. Media and events flow; client sends **bot.end** then **session.end** → **session.ended**.
