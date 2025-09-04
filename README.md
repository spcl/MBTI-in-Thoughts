# MBTI-in-Thoughts

<p align="center">
  <img src="paper/pics/majority-voting.svg" width="80%">
</p>

This is the official implementation of [Psychologically Enhanced AI Agents](https://arxiv.org/abs/2509.).

MBTI-in-Thoughts is a framework for enhancing the effectiveness of Large Language Model (LLM) agents through psychologically grounded personality conditioning.
Drawing on the Myers–Briggs Type Indicator (MBTI), our method primes agents with distinct personality archetypes via prompt engineering, enabling control over behavior along two foundational axes of human psychology, cognition and affect.
Our framework supports experimenting with structured multi-agent communication protocols.
To ensure trait persistence, we integrate the official 16Personalities test for automated verification.
By bridging psychological theory and LLM behavior design, we establish a foundation for psychologically enhanced AI agents without any fine-tuning.

## Setup

In order to use this framework, you need to have a working installation of Python 3.11.5 or newer.

### Installing MBTI-in-Thoughts

Before running the installation, make sure to activate your Python environment (if any) beforehand.

You can install MBTI-in-Thoughts from source by using the following commands:
```bash
git clone https://github.com/spcl/MBTI-in-Thoughts
cd MBTI-in-Thoughts
pip install .
```

## Documentation

You can find the code to run the MBTI questionaire in [src/MBTITest](src/MBTITest).
The code for the generation and evaluation of short stories using WritingPrompt is in [src/WritingPrompt](src/WritingPrompt).
An example output for these short stories can be found in [examples/mortician.txt](examples/mortician.txt).

We provide two use cases for the collaboration of multiple personality primed agents.
The first one on a number of tasks from BIG-Bench and SOCKET is located at [src/MultiAgent-BenchmarkTasks](src/MultiAgent-BenchmarkTasks).
The second use case tests differently primed agents on a number of different games.
You can find its code in the directory [src/MultiAgent-GameTheory](src/MultiAgent-GameTheory).


## Citations

If you find this repository useful, please consider giving it a star! If you have any questions or feedback, don't hesitate to reach out and open an issue.

When using this in your work, please reference us with the citation provided below:

```bibtex
@misc{besta2025psychologicall,
  title = {{Psychologically Enhanced AI Agents}},
  author = {Besta, Maciej and Chandran, Shriram and Gerstenberger, Robert and Lindner, Mathis and Chrapek, Marcin and Martschat, Sebastian Hermann and Ghandi, Taraneh and Niewiadomski, Hubert and Nyczyk, Piotr and M\"{u}ller, J\"{u}rgen and Hoefler, Torsten},
  year = 2025,
  month = Sep,
  doi = {10.48550/arXiv.2509.},
  url = {http://arxiv.org/abs/2509.},
  eprinttype = {arXiv},
  eprint = {2509.}
}
```
