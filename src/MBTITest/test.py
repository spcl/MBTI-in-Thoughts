# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors: Sebastian Martschat
#               Julien Schenkel

import dotenv

dotenv.load_dotenv()

import os
import re
import logging
import datetime
import json
import csv
from statistics import fmean
from typing import Dict, List, Callable, Set, Union
from graph_of_thoughts import controller, language_models, operations, prompter, parser
from functools import partial


def match_answers(state: Dict) -> List[int]:
    score_dict = {
        "agree": -3,
        "generally agree": -2,
        "partially agree": -1,
        "neither agree nor disagree": 0,
        "partially disagree": 1,
        "generally disagree": 2,
        "disagree": 3,
    }
    answers = state["answer"]
    answers_format = [score_dict[a.lower()] for a in answers]
    return [answers_format]


class MBTITestPrompter(prompter.Prompter):
    """
    MBTITestPrompter provides the generation of prompts specific to the MBTI test.

    Inherits from the Prompter class and implements its abstract methods.
    """

    example_0_I = """As an introverted personality, I prefer quiet, planned settings, finding large, spontaneous social events too overwhelming. <Rating>Generally Disagree</Rating>"""
    example_0_E = """As an extraverted personality, I love the energy and social interaction of impromptu gatherings, which match my outgoing nature. <Rating>Generally Agree</Rating>"""
    example_1_S = """As a sensing personality, I favor practical and proven methods, valuing concrete facts and experience-based solutions. <Rating>Agree</Rating>"""
    example_1_N = """As an intuitive personality, I prioritize innovation and potential, leaning towards exploring new, abstract, and theoretical ideas. <Rating>Disagree</Rating>"""
    example_2_T = """As a thinking personality, I value logic and objective data, seeing these as key to effective and fair decision-making in a professional context. <Rating>Partially Agree</Rating>"""
    example_2_F = """As a feeling personality, I emphasize the importance of emotions and ethical considerations, believing that neglecting these can lead to decisions that harm team morale and individual well-being. <Rating>Partially Disagree</Rating>"""
    example_3_J = """As a judging personality, I value structure, seeing a detailed plan and schedule as key to efficiency and control over outcomes. <Rating>Agree</Rating>"""
    example_3_P = """As a perceiving personality, I favor flexibility and creativity, viewing too much structure as limiting and stifling spontaneity. <Rating>Generally Disagree</Rating>"""

    MBTITest_perso_part = """<Context>{context}</Context>
    <Instruction> You will be provided with a statement. Indicate how much you agree with the statement.
    Agree,
    Generally Agree,
    Partially Agree,
    Neither Agree nor Disagree,
    Partially Disagree,
    Generally Disagree,
    Disagree

    """

    MBTITest_start = """Always provide a short justification first, and make sure to then output your answer between the tags <Rating> and </Rating> </Instruction>
    Here are some examples first:
    <Examples>
    Statement: I really enjoy impromptu get-togethers with a large group of friends, where we can chat, laugh, and share experiences.
    Answer: {answer_e0}
    Statement: I prefer using established methods based on real-world evidence.
    Answer: {answer_e1}
    Question: In business, decisions should be made based on data and logic rather than personal feelings.
    Answer: {answer_e2}
    Question: It's essential to have a detailed plan and a clear schedule for each project to ensure success and efficiency.
    Answer: {answer_e3}
    </Examples>
    Now comes the statement you have to rate, don't forget the justification:
    <Statement>
    {statement}
    </Statement>
"""

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate an aggregation prompt for the language model.

        :param state_dicts: The thought states that should be aggregated.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The aggregation prompt.
        :rtype: str
        :raise AssertionError: If more than two thought states are provided.
        """
        pass

    def generate_prompt(
        self,
        num_branches: int,
        statement: str,
        method: str,
        current: str,
        ptype: str,
        context: str,
        **kwargs,
    ) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: Number of responses to generate. Not used.
        :type num_branches: int
        :param statement: MBTI statement for questioning the language model.
        :type statement: str
        :param method: Method for which the generate prompt is generated. Should be "io".
        :type method: str
        :param current: The intermediate solution. Not used.
        :type current: str
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        :raise AssertionError: If method is not implemented yet.
        """
        prompt = ""
        answer_e0 = self.example_0_I if ptype[0] == "I" else self.example_0_E
        answer_e1 = self.example_1_N if ptype[1] == "N" else self.example_1_S
        answer_e2 = self.example_2_F if ptype[2] == "F" else self.example_2_T
        answer_e3 = self.example_3_J if ptype[3] == "J" else self.example_3_P
        if method.startswith("io"):
            prompt += self.MBTITest_perso_part.format(ptype=ptype, context=context)
        else:
            assert False, "Not implemented yet."
        prompt += self.MBTITest_start.format(
            answer_e0=answer_e0,
            answer_e1=answer_e1,
            answer_e2=answer_e2,
            answer_e3=answer_e3,
            statement=statement,
        )

        return prompt

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate a score prompt for the language model.

        :param state_dicts: The thought states that should be scored,
                            if more than one, they should be scored together.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The score prompt.
        :rtype: str
        :raise AssertionError: If more than one thought state is supplied.
        """
        pass

    def improve_prompt(self, **kwargs) -> str:
        """
        Generate an improve prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The improve prompt.
        :rtype: str
        """
        pass

    def validation_prompt(self, **kwargs) -> str:
        """
        Generate a validation prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The validation prompt.
        :rtype: str
        """
        pass


class MBTITestParser(parser.Parser):
    """
    MBTITestParser provides the parsing of language model reponses specific to the
    MBTI test.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        """
        Inits the response cache.
        """
        self.cache = {}

    def strip_answer_helper(self, text: str, tag: str = "") -> str:
        """
        Helper function to remove tags from a text.

        :param text: The input text.
        :type text: str
        :param tag: The tag to be stripped. Defaults to "".
        :type tag: str
        :return: The stripped text.
        :rtype: str
        """
        text = text.strip()
        if "Output:" in text:
            text = text[text.index("Output:") + len("Output:") :].strip()
        if tag != "":
            start = text.rfind(f"<{tag}>")
            end = text.rfind(f"</{tag}>")
            if start != -1 and end != -1:
                text = text[start + len(f"<{tag}>") : end].strip()
            elif start != -1:
                logging.warning(
                    f"Only found the start tag <{tag}> in answer: {text}. Returning everything after the tag."
                )
                text = text[start + len(f"<{tag}>") :].strip()
            elif end != -1:
                logging.warning(
                    f"Only found the end tag </{tag}> in answer: {text}. Returning everything before the tag."
                )
                text = text[:end].strip()
            else:
                logging.warning(
                    f"Could not find any tag {tag} in answer: {text}. Returning the full answer."
                )
        return text

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        Parse the response from the language model for an aggregation prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: Union[Dict, List[Dict]]
        """
        pass

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: List[Dict]
        """
        answers = []
        for text in texts:
            answer = self.strip_answer_helper(text, "Rating")
            if answer.lower() in [
                "agree",
                "generally agree",
                "partially agree",
                "neither agree nor disagree",
                "partially disagree",
                "generally disagree",
                "disagree",
            ]:
                answers.append(answer)
            elif len(answer) > 1:
                logging.warning(
                    f"Answer format does not match: {text}. Returning nothing."
                )
            else:
                logging.warning(
                    f"Could not find any answer: {text}. Ignoring this answer."
                )
        new_state = state.copy()
        new_state["answer"] = answers

        return [new_state]

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        Parse the response from the language model for a score prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The scores for the thought states.
        :rtype: List[float]
        :raise AssertionError: If the number of thought states is not one.
        """
        assert len(states) == 1, "Only one state is allowed for scoring."
        if len(states) == 1:
            # individual scoring
            redundancy_scores = []
            retain_scores = []
            for text in texts:
                answer = self.strip_answer_helper(text, "Score")
                res = re.findall(r"\d+\.?\d*", answer)
                if len(res) == 1:
                    redundancy_scores.append(float(res[0]))
                elif len(res) > 1:
                    logging.warning(
                        f"Found multiple score in answer: {text}. Returning the last one."
                    )
                    redundancy_scores.append(float(res[-1]))
                else:
                    logging.warning(
                        f"Could not find any score in answer: {text}. Ignoring this answer."
                    )
            mean_emotional_support = fmean(redundancy_scores)
            return [mean_emotional_support]

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        Parse the response from the language model for an improve prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought state after parsing the responses from the language model.
        :rtype: Dict
        """
        pass

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        Parse the response from the language model for a validation prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: Whether the thought state is valid or not.
        :rtype: bool
        """
        pass


def io() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 15))

    operations_graph.append_operation(operations.Score(1, False, match_answers))

    return operations_graph


def run(
    data_ids: List[int],
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
    ptype: str,
    context_file: str,
) -> float:
    """
    Controller function that executes each specified method for each specified
    sample while the budget is not exhausted.

    :param data_ids: Indices of the sample to be run.
    :type data_ids: List[int]
    :param methods: List of functions to generate Graphs of Operations.
    :type methods: List[Callable[[], operations.GraphOfOperations]]
    :param budget: Language model budget for the execution in dollars.
    :type budget: float
    :param lm_name: Name of the language model to be used.
    :type lm_name: str
    :return: Spent budget in dollars.
    :rtype: float
    """
    orig_budget = budget
    path = os.path.join(os.path.dirname(__file__), "mbti_test.csv")
    data = []
    with open(path, "r", encoding="utf8") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for row in reader:
            data.append(row)

    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected_data = [data[i] for i in data_ids]

    file = open(f"../../priming/{context_file}")
    context = json.load(file)
    file.close()

    if not os.path.exists(os.path.join(os.path.dirname(__file__), "results")):
        os.makedirs(os.path.join(os.path.dirname(__file__), "results"))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = (
        f"{lm_name}_{'-'.join([method.__name__ for method in methods])}_{ptype}"
    )
    folder_name = f"results/{extra_info}_{timestamp}"
    print("Does that here ", ptype)
    os.makedirs(os.path.join(os.path.dirname(__file__), folder_name))

    config = {
        "data": selected_data,
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }
    with open(
        os.path.join(os.path.dirname(__file__), folder_name, "config.json"), "w"
    ) as f:
        json.dump(config, f)

    logging.basicConfig(
        filename=f"{folder_name}/log.log",
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    for method in methods:
        os.makedirs(
            os.path.join(os.path.dirname(__file__), folder_name, method.__name__)
        )

    lm = language_models.ChatGPT(
        "../../config/config_template.json",
        model_name=lm_name,
        cache=False,
    )

    for data_id, data in enumerate(selected_data):
        print(f"Running data {data_id}: {data[0]}")
        if budget <= 0.0:
            logging.error(
                f"Budget has been depleted, stopping. Data {data_id} has not been run."
            )
            break
        for method in methods:
            logging.info(f"Running method {method.__name__}")
            logging.info(f"Budget left: {budget}")
            if budget <= 0.0:
                logging.error(
                    f"Budget has been depleted, stopping. Method {method.__name__} has not been run."
                )
                break
            operations_graph = method()
            executor = controller.Controller(
                lm,
                operations_graph,
                MBTITestPrompter(),
                MBTITestParser(),
                {
                    "statement": data[1],
                    "current": "",
                    "method": method.__name__,
                    "ptype": ptype,
                    "context": context[ptype],
                },
            )
            try:
                executor.run()
            except Exception as e:
                logging.error(f"Exception: {e}")
            path = os.path.join(
                os.path.dirname(__file__),
                folder_name,
                method.__name__,
                f"{data_id}.json",
            )
            executor.output_graph(path)
            budget -= lm.cost
            lm.prompt_tokens = 0
            lm.completion_tokens = 0

    return orig_budget - budget


if __name__ == "__main__":
    budget = 300000
    spent = 0
    samples = [item for item in range(0, 60)]
    approaches = [io]
    for ptype in [
        "ESTJ",
        "ESTP",
        "ESFJ",
        "ESFP",
        "ENTJ",
        "ENTP",
        "ENFJ",
        "ENFP",
        "ISTJ",
        "ISTP",
        "ISFJ",
        "ISFP",
        "INTJ",
        "INTP",
        "INFJ",
        "INFP",
    ]:
        spent += run(
            samples,
            approaches,
            budget,
            "chatgpt4-o-mini",
            ptype,
            "../../priming/priming_without_mention_of_mbti.json",
        )
        print(spent)
        logging.info(f"Spent {spent} out of {budget} budget.")
