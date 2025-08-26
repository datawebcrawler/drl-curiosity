## Configurar o VSCode para encontrar os arquivos de código
## e a dependência do sample factory (caso use o vscode)

1. CTRL+SHIFT+P
2. Open User Settings (JSON)
3. Adicionar o trecho a seguir ao arquivo:

{
    // Outras configurações...

    "python.analysis.extraPaths": [
        "./src",
        "./sample-factory"
    ]
}

## Lembrar de adicionar ao .toml
uv add --editable ../sample-factory → linka no modo dev (recomendado se você está mexendo no código).

---
### pyproject.toml
[project]
name = "curious-agent"
version = "0.1.0"
description = "Agente de RL usando Sample Factory, VizDoom e Gymnasium"
readme = "README.md"
requires-python = ">=3.12"

dependencies = [
    "gymnasium==0.29.1",
    "vizdoom==1.2.4",
    "torch==2.6.0",
    "opencv-python==4.11.0.86",
    "numpy==1.26.4",
    "tensorboard==2.19.0",
    "protobuf==6.30.2",
]

[tool.uv]  
# O uv vai gerar um uv.lock para travar as dependências exatas

[tool.uv.sources]
# Aponta para a cópia local do Sample Factory (modo editável)
sample-factory = { path = "../sample-factory", editable = true }
---

⚠️ **Warning:**  In order to run the *play* action you need to change the line 218 of the python file learner
inside the sample-factory lib as below.

```py
checkpoint_dict = torch.load(latest_checkpoint, map_location=device)
```

to

```py
checkpoint_dict = torch.load(latest_checkpoint, map_location=device, weights_only=False)
```