from datetime import datetime
from typing import Any, Dict, Optional
from langchain.experimental.generative_agents.memory import (
    BaseMemory,
)
from langchain.prompts import PromptTemplate
from langchain.experimental.generative_agents import (
    GenerativeAgent,
)

from agent import UserAgent
from utils.event import Event

from utils.group import Group

class AdvAgent(GenerativeAgent):
    id: int
    """The agent's unique identifier"""

    name: str

    funds: int

    type: str
    """The advertiser's type of its advertisement"""

    feature: str
    """The advertiser's action feature"""

    item: dict = {}

    event: Event

    memory: BaseMemory

    users: dict[int, dict] = {}

    group: Group

    time: int = 0

    strategy: str = ""

    def get_user_profile(self,user_agent: UserAgent) -> str:
        summary = (
            "The profile of the audience is:\n"
            + f"name: {user_agent.name}"
            + f"age: {user_agent.age}"
            + f"gender: {user_agent.gender}"
            + f"cash: {user_agent.cash}"
            + f"personality: {user_agent.personality}"
            + f"interest: {user_agent.interest}"
            + f"feature: {user_agent.feature}"
            + f"occupation: {user_agent.occupation}"
       )
        return summary

    def get_list(self):
        res = ""
        for i ,user in self.users.items():
            if i == -1:
                continue
            user: dict
            name = user["name"]
            pv = user["pv"]
            forward = user["forward"]
            convert = user["convert"]
            res += f"name:{name} ,View the ad {pv} times, convert {convert} time, and be influenced {forward} times\n"

        return res

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
            "{agent_summary_description}\n"
            + "Most recent observations: {most_recent_memories}"
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
            # current_time=current_time_str,
            # agent_name=self.name,
            observation=observation,
        )
        if args != None:
            kwargs.update(args)
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        # 运行对话模型，生成对话内容，并去除首尾空白字符
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        result = self.chain(prompt=prompt).run(**kwargs).strip()
        return result

    def generate_adv_decision(self, user_agent: UserAgent, round_cnt, now):
        call_to_action_template = (
            "Imagine you are an advertiser with a budget of {budget} and the type of your advertisement is {type}.\n"
            + "Your advertising style favors {adv_feature}\n"
            + "You have come across an opportunity to advertise to a target audience : {target_feature}.\n"
            + "The cost to advertise to this audience is {cost}.\n"
            + "Considering your budget, style preference, and the target audience profile:\n"
            + "Would you choose to advertise to this audience? Please answer with 'Yes' or 'No'.\n"
        )

        if self.users[-1] == 0:
            observation = "Have not placed ads before"
        else:
            observation = f"It's already been advertised to :{self.get_list()}"

        print(observation)
        args = {
            "budget": self.funds - self.time * 100,
            "type": self.type,
            "adv_feature": self.feature,
            "target_feature": self.get_user_profile(user_agent),
            "cost": 100,
        }

        result = self._generate_reaction(
            observation, call_to_action_template, args, now
        )

        return result

    def get_summary(
        self,
        now: Optional[datetime] = None,
        observation: str = None,
    ) -> str:
        """Return a descriptive summary of the agent."""

        summary = f" Name: {self.name}\n Funds: {self.funds}\n "
        return summary
