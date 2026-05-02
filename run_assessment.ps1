param(
    [Parameter(Mandatory = $true)]
    [string]$InputTarget,

    [string]$OutputDir = "extraction/output_data",
    [string]$OpenAIModel = ""
)

conda run -n paper-ext python -m assessment_cli `
    --input "$InputTarget" `
    --output-dir "$OutputDir" `
    --openai-model "$OpenAIModel"

