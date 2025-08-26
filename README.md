# Curious Agent 🧠🎮

Um projeto de **Aprendizado por Reforço Profundo (Deep Reinforcement Learning)** utilizando **[Sample Factory](https://github.com/alex-petrenko/sample-factory)**, **VizDoom**, **Gymnasium** e **PyTorch**.  
O objetivo é treinar agentes curiosos capazes de explorar ambientes complexos.

---

## 🚀 Tecnologias Utilizadas

- [Python 3.12+](https://www.python.org/)
- [PyTorch 2.6.0](https://pytorch.org/)
- [Sample Factory (customizado)](https://github.com/alex-petrenko/sample-factory)
- [VizDoom 1.2.4](https://vizdoom.cs.put.edu.pl/)
- [Gymnasium 0.29.1](https://gymnasium.farama.org/)
- [OpenCV 4.11.0.86](https://opencv.org/)
- [NumPy 1.26.4](https://numpy.org/)
- [TensorBoard 2.19.0](https://www.tensorflow.org/tensorboard)

---

## ⚙️ Configuração do Ambiente

Este projeto usa o **[uv](https://github.com/astral-sh/uv)** para gerenciar dependências de forma rápida e eficiente.

### 1. Instale as dependências
```bash
uv sync
````

### 2. Adicione o Sample Factory no modo editável (caso esteja modificando o código fonte dele)

```bash
uv add --editable ../sample-factory
```

Isso cria o vínculo no modo desenvolvimento, permitindo alterações diretas.

---

## 📝 pyproject.toml

```toml
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
```

---

## 🛠️ Configuração do VSCode (opcional)

Caso utilize o **VSCode**, adicione as pastas relevantes ao `python.analysis.extraPaths`:

1. Pressione `CTRL+SHIFT+P`
2. Selecione **Open User Settings (JSON)**
3. Inclua o trecho abaixo:

```json
{
    "python.analysis.extraPaths": [
        "./src",
        "./sample-factory"
    ]
}
```

Isso garante que o VSCode encontre os módulos do projeto e do Sample Factory.

---

## ⚠️ Patch Necessário no Sample Factory

Para rodar a função de **play** corretamente, é necessário modificar o arquivo `learner.py` dentro da pasta do **Sample Factory**:

No arquivo learner.py procure por:
 **noinspection PyBroadException**
...substitua:

```python
checkpoint_dict = torch.load(latest_checkpoint, map_location=device)
```

por

```python
checkpoint_dict = torch.load(latest_checkpoint, map_location=device, weights_only=False)
```

---

## ▶️ Como Rodar

Exemplo de execução de treino:

```bash
python src/train.py
```

Exemplo de execução de play/teste:

```bash
python src/play.py
```

Logs de treinamento podem ser visualizados no **TensorBoard**:

```bash
tensorboard --logdir runs
```

---

## 📂 Estrutura do Projeto inicial

```
curious-agent/
│── src/                  # Código principal do agente
│── sample-factory/       # Dependência local do Sample Factory (linkada via uv)
│── docs/                 # Documentação (não versionada)
│── README.md             # Este arquivo
│── pyproject.toml        # Configurações do projeto
│── uv.lock               # Lockfile de dependências
│── .gitignore
```

---

## 📌 Roadmap (ideias futuras)

* [ ] Implementar agentes com **curiosidade intrínseca (RND, ICM)**
* [ ] Implementar agentes com **Denoising Autoencoder (DAE)**
* [ ] Suporte a ambientes **parcialmente observáveis (POMDP)**
* [ ] Treino distribuído em múltiplos workers
* [ ] Benchmarks comparando diferentes abordagens de curiosidade

---

## 🤝 Contribuição

Sinta-se à vontade para abrir **issues** e **pull requests**.
Contribuições são bem-vindas! ✨

---

## 📜 Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

```