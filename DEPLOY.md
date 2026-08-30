# VetAI Doctor UI — Deploy & persistent results

## What is persisted

| Data | Location (local) | Location (Docker) |
|------|------------------|-------------------|
| Consultation index + SOAP JSON | `doctor_ui/data/consultations.db` | volume `vetai_data` → `/data` |
| Full run artifacts (transcript, NER, grounding, flags, Phase 2) | `doctor_ui/runs/<id>/` | volume `vetai_runs` → `/runs` |

Override paths with env vars:

- `VETAI_DATA_DIR` — SQLite folder
- `VETAI_RUNS_DIR` — pipeline artifact root

## Option A — Docker Compose (recommended)

1. Ensure Docker Desktop is running.
2. Keep your clinic Postgres running on the host (or point `PGHOST` at a remote DB).
3. From repo root:

```powershell
# First time: ensure .env has API keys + PGPASSWORD
docker compose up -d --build
```

4. Open http://localhost:8501 (or `http://<machine-ip>:8501` on LAN).

Useful commands:

```powershell
docker compose logs -f doctor-ui
docker compose restart doctor-ui
docker compose down          # keeps volumes (results stay)
docker compose down -v       # DELETES saved results volumes — avoid unless intentional
```

Inspect saved data:

```powershell
docker volume ls | findstr vetai
docker run --rm -v vetai_data:/data -v vetai_runs:/runs alpine ls -la /data /runs
```

## Option C — Public internet URL (temporary tunnel)

Your PC hosts the app; Cloudflare gives a public `https://....trycloudflare.com` link.

1. Start the UI:
   ```powershell
   .\scripts\run_doctor_ui_lan.ps1
   ```
2. In a **second** PowerShell window:
   ```powershell
   .\scripts\deploy_public_tunnel.ps1
   ```
3. Copy the `https://….trycloudflare.com` URL from that window and share it.

Notes:
- Both windows must stay open.
- Free quick-tunnel URLs change every restart.
- This is real remote access, not only localhost.


Copy these folders/volumes regularly:

- Local: `doctor_ui/data/` and `doctor_ui/runs/`
- Docker: backup named volumes `vetai_data` and `vetai_runs`

## Notes

- Secrets stay in `.env` (gitignored). Use `.env.deploy.example` as a template.
- Docker reaches host Postgres via `host.docker.internal` (set in compose).
- History in the UI reads SQLite; artifact folders hold the full pipeline trail.
