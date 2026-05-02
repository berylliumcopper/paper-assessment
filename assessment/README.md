# Single-Paper AI Assessment Workflow

This workflow evaluates one target paper (`PDF` or `DOI/URL`) and writes:
- `assessment.md`
- `assessment.json`
- `related_work.json`
- `assessment_run.json`
- `ai_traces/assessment_prompt.txt` and `ai_traces/assessment_raw.json`

## Setup

From repository root:

```powershell
conda run -n paper-ext python -m pip install -r requirements.txt
conda run -n paper-ext python -m playwright install chromium
```

Create/edit your local API config (untracked):

` .secrets/assessment_api.json`

Template (tracked): `config/assessment_api_example.json`

You can also set Gemini env vars:

```powershell
$env:GEMINI_API_KEY = "your_key"
$env:GEMINI_MODEL = "gemini-1.5-flash"
```

## Run

Minimal one-command usage:

```powershell
conda run -n paper-ext python -m assessment_cli "papers/test/original.pdf"
```

Assess DOI/URL:

```powershell
conda run -n paper-ext python -m assessment_cli --input "10.1038/s41586-026-10420-y"
```

Assess local PDF:

```powershell
conda run -n paper-ext python -m assessment_cli --input "D:\papers\test\original.pdf"
```

Or via helper script:

```powershell
powershell -ExecutionPolicy Bypass -File run_assessment.ps1 -InputTarget "10.1038/s41586-026-10420-y"
```
