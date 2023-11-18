# [ChatDocs-Streamlit](https://github.com/Vidminas/chatdocs-streamlit)

A fork of [ChatDocs](https://github.com/marella/chatdocs) [![PyPI](https://img.shields.io/pypi/v/chatdocs)](https://pypi.org/project/chatdocs/)

Chat with your documents offline using AI. No data leaves your system. Internet connection is only required to install the tool and download the AI models. It is based on [PrivateGPT](https://github.com/imartinez/privateGPT) but has more features.

**Contents**

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [GPU](#gpu)

## Features

- Supports GGML/GGUF models via [CTransformers](https://github.com/marella/ctransformers)
- Supports 🤗 Transformers models
- Web UI
- GPU support
- Highly configurable via `chatdocs.yml`

<details>
<summary><strong>Show supported document types</strong></summary><br>

| Extension       | Format                         |
| :-------------- | :----------------------------- |
| `.csv`          | CSV                            |
| `.docx`, `.doc` | Word Document                  |
| `.enex`         | EverNote                       |
| `.eml`          | Email                          |
| `.epub`         | EPub                           |
| `.html`         | HTML                           |
| `.md`           | Markdown                       |
| `.msg`          | Outlook Message                |
| `.odt`          | Open Document Text             |
| `.pdf`          | Portable Document Format (PDF) |
| `.pptx`, `.ppt` | PowerPoint Document            |
| `.txt`          | Text file (UTF-8)              |

</details>

## Installation

### CPU-only setup
Run `pip install git+https://github.com/Vidminas/chatdocs-streamlit.git`

### Setup with CUDA
1. Install PyTorch with CUDA enabled by following the instructions [here](https://pytorch.org/get-started/locally/).
2. `pip install ctransformers[cuda]`
3. `pip install git+https://github.com/Vidminas/chatdocs-streamlit.git`

If pip takes too long to resolve dependency versions, you can also use `pip install git+https://github.com/Vidminas/chatdocs-streamlit.git --use-deprecated=legacy-resolver`. This may result in some dependency version conflicts, but should be fine to ignore (some libraries just haven't updated the supported version bounds for their dependencies).

Download the AI models using:

```sh
chatdocs download
```

Now it can be run offline without internet connection.

## Usage

Add a directory containing documents to chat with using:

```sh
chatdocs add /path/to/documents
```

> The processed documents will be stored in `db` directory by default.

Chat with your documents using:

```sh
chatdocs ui
```

Open http://localhost:8501 in your browser to access the web UI.

It also has a nice command-line interface:

```sh
chatdocs chat
```

<details>
<summary><strong>Show preview</strong></summary><br>

![Demo](https://github.com/marella/chatdocs/raw/main/docs/cli.png)

</details>

## Configuration

All the configuration options can be changed using the `chatdocs.yml` config file. Create a `chatdocs.yml` file in some directory and run all commands from that directory. For reference, see the default [`chatdocs.yml`](https://github.com/Vidminas/chatdocs-streamlit/blob/main/chatdocs/data/chatdocs.yml) file.

You don't have to copy the entire file, just add the config options you want to change as it will be merged with the default config. For example, see [`tests/fixtures/chatdocs.yml`](https://github.com/Vidminas/chatdocs-streamlit/blob/main/tests/fixtures/chatdocs.yml) which changes only some of the config options.

### Embeddings

To change the embeddings model, add and change the following in your `chatdocs.yml`:

```yml
embeddings:
  model: hkunlp/instructor-large
```

> **Note:** When you change the embeddings model, delete the `db` directory and add documents again.

### LLMs

You can configure multiple LLMs to use for chatdocs.
The command line interface uses the first one from the list.
The UI provides radio buttons to select which one to use.

Each model in the list must specify which framework to use: either CTransformers (GGML/GGUF) or 🤗 Transformers.
To add more models, use the following template in your `chatdocs.yml`:

```yml
llms:
  - model_framework: ctransformers
    model: TheBloke/orca_mini_3B-GGML
    model_file: orca-mini-3b.ggmlv3.q4_0.bin
    model_type: llama
    config:
      context_length: 1024
      max_new_tokens: 256
  - model_framework: huggingface
    model: TheBloke/Wizard-Vicuna-7B-Uncensored-HF
    pipeline_kwargs:
      max_new_tokens: 256
```

CTransformers requires specifying `model_type` (between llama, gpt2, gpt3, falcon, ...).

> **Note:** When you add a new model for the first time, run `chatdocs download` to download the model before using it.

You can also use an existing local model file, for example:

```yml
llms:
  - model_framework: ctransformers
    model: /path/to/ggml-model.bin
    model_type: llama
```

Finally, if you wish to compare results with an OpenAI model, you can add:

```yml
  - model_framework: openai
    model: gpt-3.5-turbo-0613
    openai_api_key: YOUR-KEY-HERE (starting with sk-)
```
This is for testing purposes -- if you use the OpenAI models, your documents and chat data will be sent over the API, unlike with local LLMs.

## GPU

### Embeddings

To enable GPU (CUDA) support for the embeddings model, add the following to your `chatdocs.yml`:

```yml
embeddings:
  model_kwargs:
    device: cuda
```


### CTransformers

To enable GPU (CUDA) support for a CTransformers (GGML/GGUF) model, add the following to your `chatdocs.yml`:

```yml
llms:
  - model_framework: ctransformers
  # ...
    config:
      gpu_layers: 50
```

### 🤗 Transformers

To enable GPU (CUDA) support for the 🤗 Transformers model, add the following to your `chatdocs.yml`:

```yml
llms:
  - model_framework: huggingface
  # ...
    device: 0
```

## License

[MIT](https://github.com/marella/chatdocs/blob/main/LICENSE)
