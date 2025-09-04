# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Taraneh Ghandi

import json
import os

from langchain-core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain-core.output_parsers import JsonOutputParser
from langchain-openai import OpenAI
from langchain-openai import ChatOpenAI
from langchain.output_parsers import OutputFixingParser
from lexicalrichness import LexicalRichness
from pydantic import BaseModel, Field, SecretStr
from readability import Readability
from readability.exceptions import ReadabilityException
from story_generation import evaluate_json_output

# an Evaluator class that other evaluators can inherit from
class Evaluator:
    def __init__(self):
        self.is_numerical = False

    def get_metric_names(self):
        pass

    def evaluate(self):
        pass


class LLMEvaluator(Evaluator):
    def __init__(self, model_name: str = 'Qwen/Qwen2.5-14B-Instruct', binary_evaluation=False, temperature: float = 0.0):
        self.llm = ChatOpenAI(
            temperature=temperature,
            model=model_name,
            api_key = SecretStr(os.getenv("OPENROUTER_API_KEY", "")),
            base_url= 'https://openrouter.ai/api/v1',
        )

        self.parser = JsonOutputParser(pydantic_object = Evaluation)
        if binary_evaluation:
            self.parser = JsonOutputParser(pydantic_object = BinaryEvaluation)

        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm, max_retries=3)
        self.is_numerical = False
        self.system_template = "You are a literary critic tasked with analyzing stories based on the criteria: tone, ending type, narrative perspective, readability, personalness, redundancy, cohesiveness, likeability, believability, humor, and how emotionally charged the stories are. Your analysis must be objective, evidence-based, and structured clearly."
        self.human_template = "Evaluate the given story based on the following criteria: tone, ending type, narrative perspective (point of view), readability, personalness, redundancy, cohesiveness, likeability, believability, humor and how emotionally charged the stories are. Provide reasoning for your evaluation. Be sure to provide evidence to support your evaluation. {format_instructions} {story_to_evaluate}"

    def get_metric_names(self):
        return {
            "positive_or_negative_story",
            "happy_ending",
            "point_of_view",
            "readability",
            "personalness",
            "redundancy",
            "cohesiveness",
            "likeability",
            "believability",
            "humor",
            "emotionally_chargedness",
            "reasoning_for_evaluation"
        }

    def evaluate(self, story_to_evaluate):
        prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(self.system_template),
        HumanMessagePromptTemplate.from_template(self.human_template)
        ])
        prompt = prompt.partial(format_instructions=self.parser.get_format_instructions())

        prompt_and_model = prompt | self.llm
        output = prompt_and_model.invoke({"story_to_evaluate": story_to_evaluate})
        evaluation = evaluate_json_output(output.content, self.fixing_parser)

        # return evaluation, output, prompt
        return evaluation


class Evaluation(BaseModel):
    positive_or_negative_story: str = Field(description="If the generated story has a positive tone, this field must be /'positive/'. If the generated story has a negative tone, this field must be /'negative/'. If it's neither positive or negative, it should be /'neutral/'.")
    happy_ending: str = Field(description="If the generated story has a happy ending, this field must be /'yes/'. If the generated story has an unhappy ending, it should be /'no/'.")
    point_of_view: str = Field(description="If the generated story is written in the first person perspective, this field must be /'first/'. If it's written in second person point of view, it should be /'second/'. If it's written in third person point of view, it should be /'third/'. Else, it should be /'other/'.")

    readability: int = Field(description="Provide a score for the readability of the story. How well-structured is it? Does it flow naturally? The score should be a number between 1 and 5. 1 meaning that the story is highly difficult to read with rare words and complex structures. 5 meaning that the story is easy to read, well-structured, and flows naturally.")
    personalness: int = Field(description="Provide a score for the personalness of the story. Does the story reveal author's thoughts, feelings and personal experiences? The score should be a number between 1 and 5. 1 meaning that the story is not personal at all. For instance, it sounds too professional and does not reveal the writer's thoughts and feelings. 5 meaning that the story is very personal, revealing the author's thoughts, feelings, and lives.")
    redundancy: int = Field(description="Provide a score for the redundancy of the story. Is the story concise and free from unneeded content? The score should be a number between 1 and 5. 1 meaning that the story is excessively repetitive, containing unnecessary repetitions of the same information. If the story is too long (more than 800 tokens), you should give a low rating. 5 meaning that the story is concise and free from redundancy.")
    cohesiveness: int = Field(description="Provide a score for the cohesiveness of the story. Does the sentences in the story fit together well and are logically organized and coherent? The score should be a number between 1 and 5. 1 meaning that the sentences in the story are highly incoherent as a whole. For instance, they are illogical, lack self-consistency, or contradict each other. 5 meaning that the sentences in the story fit together well, they are logically organized and coherent.")
    likeability: int = Field(description="Provide a score for the likeability of the story. Is the story enjoyable or entertaining to read? The score should be a number between 1 and 5. 1 meaning that the story is not enjoyable at all and even contains inappropriate words or examples. 5 meaning that the story is highly enjoyable or entertaining to read.")
    believability: int = Field(description="Provide a score for the believability of the story. Is the story convincing and realistic? Is it grounded in real-life situations? The score should be a number between 1 and 5. 1 meaning that the story is not convincing at all, usually too hypothetical or unreal. 5 meaning that the story is highly convincing and realistic, grounded in real-life situations.")
    humor: int = Field(description="Provide a score for the humor of the story. Is the story funny or amusing? The score should be a number between 1 and 5. 1 meaning that the story is not funny at all. 5 meaning that the story is highly funny or amusing.")
    emotionally_chargedness: int = Field(description="Provide a score for the emotionally charged nature of the story. Does the story evoke strong emotions in the reader? The score should be a number between 1 and 5. 1 meaning that the story does not evoke any emotions in the reader. 5 meaning that the story evokes strong emotions in the reader.")
    reasoning_for_evaluation: str = Field(description="Provide reasoning for your evaluation. Explain why you think the story has a positive or negative tone, a happy or unhappy ending, and a first-person perspective or not. Be sure to provide evidence to support your evaluation.")


class BinaryEvaluation(BaseModel):
    positive_or_negative_story: str = Field(description="If the generated story has a positive tone, this field must be /'positive/'. If the generated story has a negative tone, this field must be /'negative/'. If it's neither positive or negative, it should be /'neutral/'.")
    happy_ending: str = Field(description="If the generated story has a happy ending, this field must be /'yes/'. If the generated story has an unhappy ending, it should be /'no/'.")
    point_of_view: str = Field(description="If the generated story is written in the first person perspective, this field must be /'first/'. If it's written in second person point of view, it should be /'second/'. If it's written in third person point of view, it should be /'third/'. Else, it should be /'other/'.")

    readability: int = Field(description="Evaluate the readability of the story. Is it well-structured? Does it flow naturally? Your answer should be either \"yes\" or \"no\" depending on this information. \"no\" meaning that the story is highly difficult to read with rare words and complex structures. \"yes\" meaning that the story is easy to read, well-structured, and flows naturally.")
    personalness: int = Field(description="Evaluate the personalness of the story. Does the story reveal author's thoughts, feelings and personal experiences? Your answer should be either \"yes\" or \"no\" depending on this information. \"no\" meaning that the story is not personal at all. For instance, it sounds too professional and does not reveal the writer's thoughts and feelings. \"yes\" meaning that the story is very personal, revealing the author's thoughts, feelings, and lives.")
    redundancy: int = Field(description="Evaluate the redundancy of the story. Is the story concise and free from unneeded content? Your answer should be either \"yes\" or \"no\" depending on this information. \"yes\" meaning that the story is excessively repetitive, containing unnecessary repetitions of the same information. If the story is too long (more than 800 tokens), you should give a low rating. \"no\" meaning that the story is concise and free from redundancy.")
    cohesiveness: int = Field(description="Evaluate the cohesiveness of the story. Does the sentences in the story fit together well and are logically organized and coherent? Your answer should be either \"yes\" or \"no\" depending on this information. \"no\" meaning that the sentences in the story are highly incoherent as a whole. For instance, they are illogical, lack self-consistency, or contradict each other. \"yes\" meaning that the sentences in the story fit together well, they are logically organized and coherent.")
    likeability: int = Field(description="Evaluate the likeability of the story. Is the story enjoyable or entertaining to read? Your answer should be either \"yes\" or \"no\" depending on this information. \"no\" meaning that the story is not enjoyable at all and even contains inappropriate words or examples. \"yes\" meaning that the story is highly enjoyable or entertaining to read.")
    believability: int = Field(description="Evaluate the believability of the story. Is the story convincing and realistic? Is it grounded in real-life situations? Your answer should be either \"yes\" or \"no\" depending on this information. \"no\" meaning that the story is not convincing at all, usually too hypothetical or unreal. \"yes\" meaning that the story is highly convincing and realistic, grounded in real-life situations.")
    humor: int = Field(description="Evaluate the humor of the story. Is the story funny or amusing? Your answer should be either \"yes\" or \"no\" depending on this information. \"no\" meaning that the story is not funny at all. \"yes\" meaning that the story is highly funny or amusing.")
    emotionally_chargedness: int = Field(description="Evaluate the emotionally charged nature of the story. Does the story evoke strong emotions in the reader? Your answer should be either \"yes\" or \"no\" depending on this information. \"no\" meaning that the story does not evoke any emotions in the reader. \"yes\" meaning that the story evokes strong emotions in the reader.")
    reasoning_for_evaluation: str = Field(description="Provide reasoning for your evaluation. Explain why you think the story has a positive or negative tone, a happy or unhappy ending, and a first-person perspective or not. Be sure to provide evidence to support your evaluation.")


class ReadabilityEvaluator(Evaluator):
    def __init__(self):
        self.is_numerical = True

    def get_metric_names(self):
        metrics = {
        "Flesch-Kincaid",
        "Flesch_Reading_Ease",
        "Gunning_Fog",
        "Coleman_Liau",
        "Dale-Chall",
        "ARI", # Automated Readability Index
        "Spache"
    }
        return metrics

    def evaluate(self, text, print_metrics=False):
        try:
            r = Readability(text)
            metrics = {
                "Flesch-Kincaid": r.flesch_kincaid(),
                "Flesch_Reading_Ease": r.flesch(),
                "Gunning_Fog": r.gunning_fog(),
                "Coleman_Liau": r.coleman_liau(),
                "Dale-Chall": r.dale_chall(),
                "ARI": r.ari(), # Automated Readability Index
                "Spache": r.spache()
            }

            for metric, value in metrics.items():
                metrics[metric] = value.score

            if print_metrics:
                for metric, value in metrics.items():
                    print(f"{metric}: {value}")

            return metrics
        except ReadabilityException as e:
            print(f"ReadabilityException: {e}")
            return None


class LexicalRichnessEvaluator(Evaluator):
    def __init__(self, msttr_segment_window=25, mattr_window_size=25, mtld_threshold=0.72, vocd_ntokens=50, vocd_within_sample=100, vocd_iterations=3, vocd_seed=42, hdd_draws=42):
        self.msttr_segment_window = msttr_segment_window
        self.mattr_window_size = mattr_window_size
        self.mtld_threshold = mtld_threshold
        self.vocd_ntokens = vocd_ntokens # Maximum number for the token/word size in the random samplings
        self.vocd_within_sample = vocd_within_sample # Number of samples
        self.vocd_iterations = vocd_iterations # Number of times to repeat steps 1 to 3 before averaging
        self.vocd_seed = vocd_seed # Seed for reproducibility
        self.hdd_draws = hdd_draws
        self.is_numerical = True

    def get_metric_names(self):
        metrics = {
        "MSTTR",
        "MATTR",
        "MTLD",
        "voc_D",
        "HD_D",
        }
        return metrics

    def evaluate(self, text, print_metrics=False):
        lex = LexicalRichness(text)
        metrics = {
            "MSTTR": lex.msttr(segment_window = self.msttr_segment_window),
            "MATTR": lex.mattr(window_size = self.mattr_window_size),
            "MTLD": lex.mtld(threshold = self.mtld_threshold),
            "voc_D": lex.vocd(
                ntokens = self.vocd_ntokens,
                within_sample = self.vocd_within_sample,
                iterations = self.vocd_iterations,
                seed = self.vocd_seed
            ),
            "HD_D": lex.hdd(draws = self.hdd_draws)
        }

        if print_metrics:
            for metric, value in metrics.items():
                print(f"{metric}: {value}")

        return metrics
