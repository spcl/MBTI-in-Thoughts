# Run 16 Personality Types Questionaire

The code in this directory runs the official MBTI questionaire with a given LLM.


## Installation

Before running the installation, make sure to activate your Python environment (if any) beforehand.

In order to run the MBTI questionaire, you need to install Graph of Thoughts (GoT) from the original sources:
```bash
git clone https://github.com/spcl/graph-of-thoughts
cd graph-of-thoughts
pip install .
```
Please make sure to update the API keys if you intend to use an OpenAI model in [config_template.json](../../config/config_template.json).


## Execution

You can run the 60 questions from the 16Personalities questionare for a given model by executing the `test.py` script.

```bash
export TRANSFORMERS_CACHE={PATH_TO_CACHE_DIR}
python3 test.py
```


## Plotting

The results can be plotted with the plotting script `plot.py`.

```bash
python3 plot.py
```
