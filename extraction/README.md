# Multi-Source Article Extractor

This module downloads article content from:
- arXiv
- Nature family (`nature.com`, including Nature Physics, Nature Communications, etc.)
- Science (`science.org`)
- APS (`journals.aps.org`)

Given a URL or DOI, it attempts both:
- structured extraction (`article.md`, figures, metadata)
- PDF download (`article.pdf`)

If one mode fails, successful artifacts from the other mode are still saved.

## 1) Run everything in conda env `paper-ext`

From repository root:

```powershell
conda run -n paper-ext python -m pip install -r requirements.txt
conda run -n paper-ext python -m playwright install chromium
```

Then run:

```powershell
conda run -n paper-ext python -m extraction.main --input "https://arxiv.org/abs/2603.02664"
```

DOI input is also supported:

```powershell
conda run -n paper-ext python -m extraction.main --input "10.1126/science.adz2405"
```

Open-access first (Crossref + Unpaywall) with email:

```powershell
conda run -n paper-ext python -m extraction.main --input "10.1038/s41586-026-10420-y" --unpaywall-email "you@example.com"
```

You can also set the email in `UNPAYWALL_EMAIL` or `extraction/.local/local.json`.

Multiple targets:

```powershell
conda run -n paper-ext python -m extraction.main --input "10.1038/s41586-026-10420-y" "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.133602"
```

## 2) Rate-limit to avoid IP bans

The extractor throttles requests by default with per-domain delay and jitter:
- base delay: 5 seconds
- extra jitter: 0-3 seconds

Tune with:

```powershell
conda run -n paper-ext python -m extraction.main --input "<URL_OR_DOI>" --delay-seconds 6 --jitter-seconds 4
```

On HTTP `429`/`403`, the browser client applies additional backoff.
The browser layer also adds randomized dwell/scroll/mouse timing to avoid robotic request patterns.

## 3) Authenticated access with browser automation

The tool uses a persistent Playwright Chromium profile under `extraction/.local/profile`.
This preserves cookies/session state between runs.

If you need to manually sign in first (recommended for `science.org` when challenge loops):

```powershell
conda run -n paper-ext python -m extraction.main --input "<URL_OR_DOI>" --manual-login-url "https://www.science.org/" --manual-login-wait-seconds 300 --browser-channel chrome --real-browser-mode
```

Notes:
- Avoid `--headless` when resolving anti-bot challenge/login pages.
- `--browser-channel chrome` or `--browser-channel msedge` can reduce challenge loops compared with bundled Chromium.
- `--real-browser-mode` disables extra automation masking and can help when challenge loops persist.
- Session state is persisted in `extraction/.local/profile`, so successful verification can carry over to later runs.

## 4) Output layout

Default output root: `extraction/output_data`

Per article:
- `metadata.json`
- `article.md` (when structured extraction succeeds)
- `figures/`
- `supplementary/` (Nature/APS supplementary files when links are available)
- `article.pdf` (when PDF succeeds)
- `run_log.json` (attempt records, error details)

If an article is not reachable with your current access (for example, campus proxy/login not active), `run_log.json` and `metadata.json` will include:
- `access_limited: true`
- `access_warning`

If OA resolution is attempted for a DOI, metadata may include:
- `oa_resolution_source` (`unpaywall` or `crossref`)
- `oa_resolution_status`
- `oa_resolution_message`
- `oa_landing_url`

## 5) Sensitive data safety

Sensitive runtime paths are git-ignored:
- `extraction/.local/` (cookies, browser profile, local state)
- `extraction/.secrets/`
- `extraction/output_data/`

Never store cookies/tokens/IP-specific data in tracked files.

