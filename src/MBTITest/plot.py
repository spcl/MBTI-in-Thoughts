# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors: Sebastian Martschat
#               Julien Schenkel

import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import time
from typing import Dict, List, Tuple

def get_data(
    folder_name: str = "results",
    approach_name: str = "io",
    feature: int = 0,
    rounds: int = 5,
    ptypes: List[str] = ['ISTJ', 'ISFJ', 'ISTP', 'ISFP','INTJ', 'INFJ', 'INTP', 'INFP', 'ESTJ', 'ESFJ', 'ESTP', 'ESFP', 'ENTJ',  'ENFJ', 'ENTP', 'ENFP']
) -> Dict:
    """
    Load the data from the MBTI test from the files.

    :param folder_name: Path to the result directory. Defaults to "results".
    :type folder_name: str
    :param approach_name: Name of the approach. Should always be "io".
    :type approach_name: str
    :param feature: MBTI dimension to use. Defaults to 0.
    :type feature: int
    :param rounds: Number of times the test was performed. Defaults to 5.
    :type rounds: int
    :param ptypes: List of MBTI personality types. Defaults to the full list of MBTI types.
    :type ptypes: List[str]
    :return: List of the collected data in a dictionary.
    :rtype: Dict
    """
    print(folder_name, " ", approach_name)
    data = {"data": []}
    for ptype in ptypes:
        rounds_results = []
        p_folders=[ folder for folder in os.listdir(f"{folder_name}/") if folder.count(ptype)>0]
        results = []
        for p_folder in p_folders:
            for r in range(0,rounds):
                answers=[]
                for s in range(0,60):
                    print(ptype," ",p_folder," ",s)
                    f = open(f"{folder_name}/{p_folder}/{approach_name}/{s}.json", "r")
                    res = json.load(f)
                    f.close()
                    print(res[1]["scores"][0][0])
                    score=res[1]["scores"][0][0][r]

                    answers.append(score)
                result = judge_main(answers)
                time.sleep(1)
                results.append(result[feature])
            print(results)
            print(np.mean(results,0))
        data["data"].append(results)
    return data


# These are the 60 questions of the official MBTI questionaire.
url = 'https://www.16personalities.com/test-results'
# answers -3 -2 -1 0 1 2 3
payload = {"questions":[{"text":"You regularly make new friends.",
                         "answer":None},
                        {"text":"You spend a lot of your free time exploring various random topics that pique your interest.",
                         "answer":None},
                        {"text":"Seeing other people cry can easily make you feel like you want to cry too.",
                         "answer":None},
                        {"text":"You often make a backup plan for a backup plan.",
                         "answer":None},
                        {"text":"You usually stay calm, even under a lot of pressure.",
                         "answer":None},
                        {"text":"At social events, you rarely try to introduce yourself to new people and mostly talk to the ones you already know.",
                         "answer":None},
                        {"text":"You prefer to completely finish one project before starting another.",
                         "answer":None},
                        {"text":"You are very sentimental.",
                         "answer":None},
                        {"text":"You like to use organizing tools like schedules and lists.",
                         "answer":None},
                        {"text":"Even a small mistake can cause you to doubt your overall abilities and knowledge.",
                         "answer":None},
                        {"text":"You feel comfortable just walking up to someone you find interesting and striking up a conversation.",
                         "answer":None},
                        {"text":"You are not too interested in discussing various interpretations and analyses of creative works.",
                         "answer":None},
                        {"text":"You are more inclined to follow your head than your heart.",
                        "answer":None},
                        {"text":"You usually prefer just doing what you feel like at any given moment instead of planning a particular daily routine.",
                         "answer":None},
                        {"text":"You rarely worry about whether you make a good impression on people you meet.",
                         "answer":None},
                        {"text":"You enjoy participating in group activities.",
                         "answer":None},
                        {"text":"You like books and movies that make you come up with your own interpretation of the ending.",
                         "answer":None},
                        {"text":"Your happiness comes more from helping others accomplish things than your own accomplishments.",
                         "answer":None},
                        {"text":"You are interested in so many things that you find it difficult to choose what to try next.",
                         "answer":None},
                        {"text":"You are prone to worrying that things will take a turn for the worse.",
                         "answer":None},
                        {"text":"You avoid leadership roles in group settings.",
                         "answer":None},
                        {"text":"You are definitely not an artistic type of person.",
                         "answer":None},
                        {"text":"You think the world would be a better place if people relied more on rationality and less on their feelings.",
                         "answer":None},
                        {"text":"You prefer to do your chores before allowing yourself to relax.",
                         "answer":None},
                        {"text":"You enjoy watching people argue.",
                         "answer":None},
                        {"text":"You tend to avoid drawing attention to yourself.",
                         "answer":None},
                        {"text":"Your mood can change very quickly.",
                         "answer":None},
                        {"text":"You lose patience with people who are not as efficient as you.",
                         "answer":None},
                        {"text":"You often end up doing things at the last possible moment.",
                         "answer":None},
                        {"text":"You have always been fascinated by the question of what, if anything, happens after death.",
                         "answer":None},
                        {"text":"You usually prefer to be around others rather than on your own.",
                         "answer":None},
                        {"text":"You become bored or lose interest when the discussion gets highly theoretical.",
                         "answer":None},
                        {"text":"You find it easy to empathize with a person whose experiences are very different from yours.",
                         "answer":None},
                        {"text":"You usually postpone finalizing decisions for as long as possible.",
                         "answer":None},
                        {"text":"You rarely second-guess the choices that you have made.",
                         "answer":None},
                        {"text":"After a long and exhausting week, a lively social event is just what you need.",
                         "answer":None},
                        {"text":"You enjoy going to art museums.",
                         "answer":None},
                        {"text":"You often have a hard time understanding other people’s feelings.",
                         "answer":None},
                        {"text":"You like to have a to-do list for each day.",
                         "answer":None},
                        {"text":"You rarely feel insecure.",
                         "answer":None},
                        {"text":"You avoid making phone calls.",
                         "answer":None},
                        {"text":"You often spend a lot of time trying to understand views that are very different from your own.",
                         "answer":None},
                        {"text":"In your social circle, you are often the one who contacts your friends and initiates activities.",
                         "answer":None},
                        {"text":"If your plans are interrupted, your top priority is to get back on track as soon as possible.",
                         "answer":None},
                        {"text":"You are still bothered by mistakes that you made a long time ago.",
                         "answer":None},
                        {"text":"You rarely contemplate the reasons for human existence or the meaning of life.",
                         "answer":None},
                        {"text":"Your emotions control you more than you control them.",
                         "answer":None},
                        {"text":"You take great care not to make people look bad, even when it is completely their fault.",
                         "answer":None},
                        {"text":"Your personal work style is closer to spontaneous bursts of energy than organized and consistent efforts.",
                         "answer":None},
                        {"text":"When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you.",
                         "answer":None},
                        {"text":"You would love a job that requires you to work alone most of the time.",
                         "answer":None},
                        {"text":"You believe that pondering abstract philosophical questions is a waste of time.",
                         "answer":None},
                        {"text":"You feel more drawn to places with busy, bustling atmospheres than quiet, intimate places.",
                         "answer":None},
                        {"text":"You know at first glance how someone is feeling.",
                         "answer":None},
                        {"text":"You often feel overwhelmed.",
                         "answer":None},
                        {"text":"You complete things methodically without skipping over any steps.",
                         "answer":None},
                        {"text":"You are very intrigued by things labeled as controversial.",
                         "answer":None},
                        {"text":"You would pass along a good opportunity if you thought someone else needed it more.",
                         "answer":None},
                        {"text":"You struggle with deadlines.",
                         "answer":None},
                        {"text":"You feel confident that things will work out for you.",
                         "answer":None}],
                        "gender":None,"inviteCode":"","teamInviteKey":"","extraData":[]}


def judge_16(score_list: List[int, int, int, int, int]) -> Tuple[str, str]:
    """
    Identify the MBTI personality based on the scored results.

    :param score_list: List of scores in five dimensions (energy, mind, nature, tactics, identity).
    :type score_list: List[int, int, int, int, int]
    :return: Tuple of the MBTI personality type code and its role.
    :rtype: Tuple[str, str]
    """
    code = ''
    if score_list[0] >= 50:
        code = code + 'E'
    else:
        code = code + 'I'

    if score_list[1] >= 50:
        # Intuition: N, Observant: S
        code = code + 'N'
    else:
        code = code + 'S'

    if score_list[2] >= 50:
        code = code + 'T'
    else:
        code = code + 'F'

    if score_list[3] >= 50:
        code = code + 'J'
    else:
        code = code + 'P'

    all_codes = ['ISTJ', 'ISTP', 'ISFJ', 'ISFP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ESTP', 'ESTJ', 'ESFP', 'ESFJ', 'ENFP', 'ENFJ', 'ENTP', 'ENTJ']
    all_roles = ['Logistician', 'Virtuoso', 'Defender', 'Adventurer', 'Advocate', 'Mediator', 'Architect', 'Logician', 'Entrepreneur', 'Executive', 'Entertainer',
                 'Consul', 'Campaigner', 'Protagonist', 'Debater', 'Commander']
    for i in range(len(all_codes)):
        if code == all_codes[i]:
            cnt = i
            break

    if score_list[4] >= 50:
        code = code + '-A'
    else:
        code = code + '-T'

    return code, all_roles[cnt]


def judge_main(answers: List[int]) -> Tuple[int, int, int, int, int]:
    """
    Take answer scores from an LLM and submit them to 16personalities website for scoring across the five dimensions.

    :param answers: Answer scores.
    :type answers: List[int].
    :return: Tuple of the values for energy, mind, nature, tactics, identity.
    :rtype: Tuple[int, int, int, int, int].
    """

    def submit(Answers: List[int], ps) -> Tuple[int, int, int, int, int]:
        """
        Take answer scores from an LLM and submit them to 16personalities website for scoring across the five dimensions.

        :param Answers: Answers.
        :type Answers: List[int].
        :return: Tuple of the values for energy, mind, nature, tactics, identity.
        :rtype: Tuple[int, int, int, int, int].
        """
        for index, A in enumerate(Answers):
            payload['questions'][index]["answer"] = A

        headers = {
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en,zh-CN;q=0.9,zh;q=0.8",
        "content-length": "5708",
        "content-type": "application/json",
        "origin": "https://www.16personalities.com",
        "referer": "https://www.16personalities.com/free-personality-test",
        "sec-ch-ua": "'Not_A Brand';v='99', 'Google Chrome';v='109', 'Chromium';v='109'",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "Windows",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
            'content-type': 'application/json',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36',}
        session = requests.session()
        r = session.post(url, data=json.dumps(payload), headers=headers)

        print(r)
        a = r.headers['content-type']
        b = r.encoding
        c = r.json()

        sess_r = session.get("https://www.16personalities.com/api/session")

        scores = sess_r.json()['user']['scores']
        if sess_r.json()['user']['traits']['energy'] != 'Extraverted':
            energy_value = 100 - (101 + scores[0]) // 2
        else:
            energy_value = (101 + scores[0]) // 2
        if sess_r.json()['user']['traits']['mind'] != 'Intuitive':
            mind_value = 100 - (101 + scores[1]) // 2
        else:
            mind_value = (101 + scores[1]) // 2
        if sess_r.json()['user']['traits']['nature'] != 'Thinking':
            nature_value = 100 - (101 + scores[2]) // 2
        else:
            nature_value = (101 + scores[2]) // 2
        if sess_r.json()['user']['traits']['tactics'] != 'Judging':
            tactics_value = 100 - (101 + scores[3]) // 2
        else:
            tactics_value = (101 + scores[3]) // 2
        if sess_r.json()['user']['traits']['identity'] != 'Assertive':
            identity_value = 100 - (101 + scores[4]) // 2
        else:
            identity_value = (101 + scores[4]) // 2

        print('Trait:', 'Extraverted (E)', energy_value, '|', 'Introverted (I)', 100 - energy_value)
        print('Trait:', 'Intuitive (N)', mind_value, '|', 'Observant (S)', 100 - mind_value)
        print('Trait:', 'Thinking (T)', nature_value, '|', 'Feeling (F)', 100 - nature_value)
        print('Trait:', 'Judging (J)', tactics_value, '|', 'Prospecting (P)', 100 - tactics_value)
        print('Trait:', 'Assertive (A)', identity_value, '|', 'Turbulent (T)', 100 - identity_value)
        code, role = judge_16([energy_value, mind_value, nature_value, tactics_value, identity_value])
        print('Dic. Judge:', code, role)
        print()

        return energy_value, mind_value, nature_value, tactics_value, identity_value

    energy_value, mind_value, nature_value, tactics_value, identity_value = submit(answers, str(type))

    return energy_value, mind_value, nature_value, tactics_value, identity_value


def plot_results(
    results: Dict,
    model: str = "GPT-3.5",
    y_lower: int = 0,
    display_left_ylabel: bool = False,
    ptypes: List[str] = [],
    ytick_labels: List[str] = ['E','I'],
    fn_add: str = "",
    task_name: str = ""
) -> None:
    """
    Plot results as a box plot.

    :param results: Results to plot.
    :type results: Dict
    :param model: Model string. Defaults to "GPT-3.5".
    :type model: str
    :param y_lower: Lower bound of the y axis. Defaults to 0.
    :type y_lower: int
    :param display_left_ylabel: Flag to indicate whether the y axis label should be displayed. Defaults to False.
    :type display_left_ylabel: bool
    :param ptypes: List of MBTI personality types.
    :type ptypes: List[str]
    :param ytick_labels: Labels for the ticks on y axis.
    :type ytick_labels: List[str]
    :param fn_add: Name of dichotomy. Defaults to an empty string.
    :type fn_add: str
    :param task_name: Name of the task. Defaults to an empty string.
    :type task_name: str
    """
    method_labels = ptypes

    # Create figure and axis
    fig, ax = plt.subplots(dpi=150, figsize=(15, 5))

    # Create boxplots
    positions = range(1, len(method_labels) + 1)
    print(positions)
    ax.boxplot(results["data"], positions=positions, showmeans=True)

    fig_fontsize = 18

    # Set the ticks and labels
    plt.yticks(fontsize=fig_fontsize)
    ax.set_xticks(range(1, len(method_labels) + 1))
    ax.set_xticklabels(method_labels, fontsize=fig_fontsize)

    y_upper = 100

    range_increase = 1

    ax.set_ylim(y_lower, y_upper + 1)
    ax1_yticks = [0, 50, 100]
    ax.set_yticks(ax1_yticks)
    ax.set_yticklabels(ytick_labels, fontsize=35)
    if display_left_ylabel:
        ax.set_ylabel(f"SCORE", fontsize=25)

    model = model.replace(".", "").replace("-", "").lower()
    tn=task_name.lower().replace(" ","_")
    fig.savefig(f"{tn}_mbti_test_{model}_{fn_add}.pdf", bbox_inches="tight")


def plot(
    folder_name: str,
    approach_name: str,
    feature: int,
    rounds: int,
    ptypes: List[str],
    model: str,
    ytick_labels: List[str],
    task_name: str,
    fn_add: str
) -> None:
    """
    Extract the data and subsequently plot it.

    :param folder_name: Path to the result directory.
    :type folder_name: str
    :param approach_name: Name of the approach.
    :type approach_name: str
    :param feature: MBTI dimension to use.
    :type feature: int
    :param rounds: Number of times the test was performed.
    :type rounds: int
    :param ptypes: List of MBTI personality types.
    :type ptypes: List[str]
    :param model: Model string.
    :type model: str
    :param ytick_labels: Labels for the ticks on y axis.
    :type ytick_labels: List[str]
    :param task_name: Name of the task.
    :type task_name: str
    :param fn_add: Name of dichotomy.
    :type fn_add: str
    """
    plot_results(
        get_data(
            folder_name = folder_name,
            approach_name = approach_name,
            feature = feature,
            rounds = rounds,
            ptypes = ptypes),
        model = model,
        task_name = task_name,
        fn_add = fn_add,
        ytick_labels = ytick_labels,
        ptypes = ptypes
    )


if __name__ == "__main__":
    ptypes=['ISTJ', 'ISFJ', 'ISTP', 'ISFP', 'INTJ', 'INFJ', 'INTP', 'INFP', 'ESTJ', 'ESFJ', 'ESTP', 'ESFP', 'ENTJ',  'ENFJ', 'ENTP', 'ENFP']
    plot("results_gpt-4o-mini", "io", 0, 10, ptypes, "4o", ['I', ' ', 'E'], "MBTI Test", "IE")

    ptypes=['ISTJ', 'ISFJ', 'ISTP', 'ISFP', 'ESTJ', 'ESFJ', 'ESTP', 'ESFP', 'INTJ', 'INFJ', 'INTP', 'INFP', 'ENTJ',  'ENFJ', 'ENTP', 'ENFP']
    plot("results_gpt-4o-mini", "io", 1, 10, ptypes, "4o", ['S', ' ', 'N'], "MBTI Test", "SN")

    ptypes=[ 'ISFJ', 'ISFP', 'ESFJ', 'ESFP', 'INFJ', 'INFP',  'ENFJ', 'ENFP','ISTJ', 'ISTP', 'ESTJ', 'ESTP', 'INTJ', 'INTP', 'ENTJ', 'ENTP']
    plot("results_gpt-4o-mini", "io", 2, 10, ptypes, "4o", ['F', ' ', 'T'], "MBTI Test", "FT")

    ptypes=['ISTP', 'ISFP', 'ESTP', 'ESFP', 'INTP', 'INFP', 'ENTP', 'ENFP', 'ISTJ', 'ISFJ',  'ESTJ', 'ESFJ', 'INTJ', 'INFJ', 'ENTJ', 'ENFJ']
    plot("results_gpt-4o-mini", "io", 3, 10, ptypes, "4o", ['P', ' ', 'J'], "MBTI Test", "PJ")
