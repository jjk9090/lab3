import itertools
import queue
import re
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import backoff
from langchain.experimental.generative_agents.memory import (
    GenerativeAgentMemory,
    BaseMemory,
)
from langchain.prompts import PromptTemplate
from langchain.experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)
from utils.event import Event
from concurrent.futures import ThreadPoolExecutor
from utils.utils import judge_cate,process_data
from typing import List
import time
import openai
import asyncio
class UserAgent(GenerativeAgent):
    id: int

    name: str

    gender: str

    age: int

    cash: int

    personality: str

    interest: str

    feature: str

    occupation: str

    event: Event

    memory: BaseMemory

    def get_summary(
            self,
            now: Optional[datetime] = None,
            observation: str = None,
    ) -> str:
        """Return a descriptive summary of the agent."""
        summary = (
            f"Name: {self.name}\n"
            f"Age: {self.age}\n"
            f"Gender: {self.gender}\n"
            f"Cash: {self.cash}\n"
            f"Traits: {self.traits}\n"
            f"Interest: {self.interest}\n"
            f"Feature: {self.feature}\n"
        )
        return summary

    def _generate_reaction(
        self,
        observation: str,
        suffix: str,
        args: Optional[dict[str, any]] = None,
        now: Optional[datetime] = None,
    ) -> str:
        # 生成对话的提示内容
        """React to a given observation."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\n\n"
            + suffix
        )
        now = datetime.now() if now is None else now
        # 获取代理对象的摘要描述，使用了当前时间和产品对象作为参数
        agent_summary_description = self.get_summary(now=now, observation=observation)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )

        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            # agent_name=self.name,
            agent_name=self.name,
            observation=observation,
        )
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        # 运行对话模型，生成对话内容，并去除首尾空白字符
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        # time.sleep(10)
        response = self.chain(prompt=prompt).run(**kwargs)

        result = response.strip()
        # time.sleep(10)
        return result

    def take_action(self, now, observation):
        # if user
        call_to_action_template = (
                "Imagine you are {agent_name}. \n"
                + "1. Would you being swayed or influenced by this advertisement?\n"
                + "2. Would you consider purchasing the product mentioned in the advertisement?\n"
                + "Please answer clearly in the following format:\n"
                + "Influenced: [Yes/No]\n"
                + "Convert: [Yes/No].\n"
                + "Feeling: [feeling to this advertisement]\n"
        )

        response = self._generate_reaction(
            observation, call_to_action_template, now
        )
        print(response)
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} take action: " f"{response}",
                self.memory.now_key: now,
            },
        )

        forward_match = re.search(r"Influenced: (\w+)", response)
        forward_value = forward_match.group(1) if forward_match else None

        # Extract "Convert" content
        convert_match = re.search(r"Convert: (\w+)", response)
        convert_value = convert_match.group(1) if convert_match else None

        forward = False
        if forward_value.lower() == "yes":
            forward = True

        convert = False
        if convert_value.lower() == "yes":
            convert = True
        return forward, convert

    from typing import List

    def process_record(self,record,now):
        cate = int(record['cate'])
        content = record['content']
        cateName = judge_cate(cate)
        id = random.randint((cate - 1) * 100 + 1,cate * 100)

        cont = " And its content is \"" + content + "\" "
        if record['label'] == '0':
            behavior = f" then {str(self.id)}  neithor forward nor convert the advertisement.\n"
        elif record['label'] == '1':
            behavior = f" then {str(self.id)} forward the advertisement.\n"
        elif record['label'] == '2':
            behavior = f" then {str(self.id)} convert the advertisement.\n"
        else:
            behavior = ""

        observation0 = (str(self.id) + " has watched an advertisement with type of " + cateName + " and number of " + str(id)) + ","
        prompt = observation0  + cont + behavior
        print(prompt)
        self.add_memory(prompt, now=now)

    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    # @lru_cache(maxsize=None)
    # @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def add_memory(self,observation, now):
        self.memory.add_memory(observation, now=now)
    def anew_list(self,data_list:List):
        # 将label为1和2的元素分别提取出来
        label_1_2_data = [item for item in data_list if item['label'] in ['1', '2']]
        label_0_data = [item for item in data_list if item['label'] == '0']
        # # 循环遍历分组列表
        # 打散label为1和2的元素
        random.shuffle(label_1_2_data)
        part1_remaining,part2_remaining = process_data(label_0_data)
        # 计算label为1和2的元素需要分配到的数量
        total_label_1_2 = len(label_1_2_data)
        required_label_1_2_part1 = total_label_1_2 // 2
        required_label_1_2_part2 = total_label_1_2 - required_label_1_2_part1

        # 将label为1和2的元素按照1:3的比例分配到两部分
        part1_label_1_2 = label_1_2_data[:required_label_1_2_part1]
        part2_label_1_2 = label_1_2_data[required_label_1_2_part1:]


        # 合并两部分数据
        part2_data = part1_label_1_2 + part1_remaining
        part1_data = part2_label_1_2 + part2_remaining

        if (len(part1_data) > 5):
            part1_data = part1_data[:5]

        if (len(part2_data) > 15):
            part2_data = part2_data[:15]
        # print("第一部分数据：")
        # print(part1_data)
        #
        # print("第二部分数据：")
        # print(part2_data)
        return part1_data,part2_data

    def user_pre_memory(self, pre: List, now) -> List:
        remain,observation_list = self.anew_list(pre)
        for i in range(len(observation_list)):
            self.process_record(observation_list[i], now)
        # with ThreadPoolExecutor() as executor:
        #     observation_list = list(executor.map(self.process_record, observation_list,now))
        # observation_list = [item for sublist in observation_list for item in sublist]  # 将列表展开
        # observation_list = [x for x in observation_list if x]  # 去掉空字符串
        # # observation = "then ".join(observation_list)
        # max_length = 2049 - len(str(self.id)) - len(" has watched advertisements; then")
        # observation_list = observation_list[:max_length]
        #
        # observation = "".join(observation_list)
        # self.add_memory(observation, now=now)
        return remain
        # self.memory.add_memory(observation, now=now)

    def take_action1(self, adv, now) -> Tuple[str, str]:
        call_to_action_template = (
                "Imagine you are {agent_name}. \n"
                + "1. Would you being swayed or influenced by this advertisement to your friends or family?\n"
                + "2. Would you consider purchasing the product mentioned in the advertisement?\n"
                + "Please answer clearly in the following format:\n"
                + "Influenced: [Yes/No]\n"
                + "Convert: [Yes/No].\n"
                + "Feeling: [feeling to this advertisement]\n"
        )
        type = adv["type"]
        content = adv["content"]
        product = adv["product"]
        observation = f"Here's an advertisement: the type is {type},the content is {content} and the product of the ad is {product}\n"
        args = {
            "agent_name": self.name
        }
        response = self._generate_reaction(observation, call_to_action_template, args, now)
        print(response)
        forward_match = re.search(r"Influenced: (\w+)", response)
        forward_value = forward_match.group(1) if forward_match else None

        # Extract "Convert" content
        convert_match = re.search(r"Convert: (\w+)", response)
        convert_value = convert_match.group(1) if convert_match else None

        # Extract "Feeling" content
        feeling_match = re.search(r"Feeling: ([\w\s]+)", response)
        feeling_value = feeling_match.group(1) if feeling_match else None

        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} take action: " f"{response}",
                self.memory.now_key: now,
            },
        )
        forward = False
        if forward_value.lower() == "yes":
            forward = True
        convert = False
        if convert_value.lower() == "yes":
            convert = True
        return forward, convert

    def generate_feeling(self, observation: str, now) -> str:
        """Feel about each item bought."""
        call_to_action_template = (
            "{agent_id}, how did you feel about the advertisement you just watched? Describe your feelings in one line."
            + "NOTE: Please answer in the first-person perspective."
            + "\n\n"
        )

        full_result = self._generate_reaction(observation, call_to_action_template, now)
        results = full_result.split(".")
        feelings = ""
        for result in results:
            if result.find("language model") != -1:
                break
            feelings += result
        if feelings == "":
            results = full_result.split(",")
            for result in results:
                if result.find("language model") != -1:
                    break
                feelings += result
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} felt: " f"{feelings}",
                self.memory.now_key: now,
            },
        )
        return feelings

