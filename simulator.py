import argparse
import json
from collections import namedtuple
from datetime import datetime
import os
import threading
import logging
import random
import math
import dill

from utils.group import Group

logging.basicConfig(level=logging.ERROR)

from tqdm import tqdm
from yacs.config import CfgNode
import faiss
from agent import UserAgent, AdvAgent
from common import Data
from utils import utils
from utils.message import Message

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain.experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)

from utils.event import reset_event
from utils.utils import judge_cate
class ESimulator:
    def __init__(self, config: CfgNode, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.round_cnt = 0
        self.round_msg: list[Message] = []
        self.play_event = threading.Event()
        self.now = datetime.now().replace(hour=8, minute=0, second=0)
        self.result = {}
        self.file_name_path: list[str] = []
        self.groups = {}
        """the audiences selected by every advertiser in each round"""

    def load_simulator(self):
        self.round_cnt = 0
        self.data = Data(self.config, self.logger)
        self.user_agents, self.adv_agents = self.agent_creation()
        self.logger.info("Simulator loaded.")

    def agent_creation(self):
        user_agents = {}
        adv_agents = {}
        api_keys = list(self.config["api_keys"])
        for i in tqdm(range(len(self.data.users))):
            api_key = api_keys[0]
            agent = self.create_user_agent(i, api_key)
            user_agents[agent.id] = agent

        for i in tqdm(range(len(self.data.advertisers))):
            api_key = api_keys[0]
            agent = self.create_adv_agent(i, api_key)
            adv_agents[agent.id] = agent
            for i in range(len(self.data.users)):
                agent.users[i] = {'name':user_agents[i].name,'pv':0,'forward':0,'convert':0}
            agent.users[-1] = 0

        return user_agents,adv_agents

    def relevance_score_fn(self, score: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        # This will differ depending on a few things:
        # - the distance / similarity metric used by the VectorStore
        # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
        # This function converts the euclidean norm of normalized embeddings
        # (0 is most similar, sqrt(2) most dissimilar)
        # to a similarity function (0 to 1)
        return 1.0 - score / math.sqrt(2)

    def create_new_memory_retriever(self):
        """Create a new vector store retriever unique to the agent."""
        # Define your embedding model
        embeddings_model = OpenAIEmbeddings()
        # Initialize the vectorstore as empty
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(
            embeddings_model.embed_query,
            index,
            InMemoryDocstore({}),
            {},
            relevance_score_fn=self.relevance_score_fn,
        )

        # If choose UserAgentMemory, you must use UserAgentRetriever rather than TimeWeightedVectorStoreRetriever.
        RetrieverClass = (
            TimeWeightedVectorStoreRetriever
        )

        return RetrieverClass(
            vectorstore=vectorstore, other_score_keys=["importance"], now=self.now, k=5
        )
    def create_user_agent(self, i, api_key) -> UserAgent:
        LLM = utils.get_llm(config=self.config, logger=self.logger, api_key=api_key)
        MemoryClass = (
            GenerativeAgentMemory
        )
        agent_memory = MemoryClass(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(),
            now_key=self.now.strftime("%Y-%m-%dT%H:%M:%S"),
            verbose=False,
            reflection_threshold=10,
        )

        agent = UserAgent(
            id=i,
            name=self.data.users[i]["name"],
            age=self.data.users[i]["age"],
            gender=self.data.users[i]["gender"],
            cash=self.data.users[i]["cash"],
            status=self.data.users[i]["feature"],
            personality=self.data.users[i]["personality"],
            interest=self.data.users[i]["interest"],
            feature=self.data.users[i]["feature"],
            occupation=self.data.users[i]["occupation"],
            memory_retriever=self.create_new_memory_retriever(),
            llm=LLM,
            memory=agent_memory,
            event=reset_event(self.now),
        )
        return agent

    def get_item_for_adv(self, type):
        result = {}
        index = 0
        for i,item in self.data.items.items():
            if item["type"] == type:
                result[index] = item
                index += 1
        return result

    def create_adv_agent(self, i, api_key) -> AdvAgent:
        LLM = utils.get_llm(config=self.config, logger=self.logger, api_key=api_key)
        MemoryClass = (
            GenerativeAgentMemory
        )
        agent_memory = MemoryClass(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(),
            now_key=self.now.strftime("%Y-%m-%dT%H:%M:%S"),
            verbose=False,
            reflection_threshold=10,
        )

        agent = AdvAgent(
            id=i,
            status="xxx",
            name=self.data.advertisers[i]["name"],
            funds=self.data.advertisers[i]["funds"],
            feature=self.data.advertisers[i]["feature"],
            type=self.data.advertisers[i]["type"],
            memory_retriever=self.create_new_memory_retriever(),
            llm=LLM,
            memory=agent_memory,
            event=reset_event(self.now),
            item=self.get_item_for_adv(self.data.advertisers[i]["type"]),
            group=Group(i)
        )
        return agent

    def play(self):
        self.play_event.set()

    def adv_one_step(self, agent: AdvAgent):
        """Run one step of an advertiser."""
        self.play_event.wait()
        name = agent.name
        message = []
        users = self.user_agents
        for i, user_agent in users.items():
            decision = agent.generate_adv_decision(user_agent, self.round_cnt, self.now)
            self.logger.info(
                f"{name} is going to decide whether to advertise to user {user_agent.name}: {decision}."
            )
            if decision.lower() == "yes":
                agent.users[user_agent.id]['pv'] += 1
                agent.users[-1] += 1
                agent.time += 1
                message.append(
                    Message(
                        agent_id=agent.id,
                        action="ADVERTISE",
                        content=f"{name} is going to advertise to user {user_agent.name}.",
                    )
                )

                self.round_msg.append(
                    Message(
                        agent_id=agent.id,
                        action="ADVERTISE",
                        content=f"{name} is going to advertise to user {user_agent.name}.",
                    )
                )
                self.logger.info(f"{name} is going to advertise to user {user_agent.name}.")

                agent.group.id = agent.id
                agent.group.users.append(user_agent.id)
        self.groups[agent.id] = agent.group

    def group_one_step(self, group: Group, adv, adv_agent):
        self.play_event.wait()
        message = []
        type = adv["type"]
        content = adv["content"]
        product = adv["product"]
        observation = f"Here's an advertisement: the type is {type},the slogan is {content} and the product of the ad is {product}\n"
        self.logger.info(observation)
        for j in group.users:
            if (self.round_cnt == 3):
                print("")
            user_agent = self.user_agents[j]
            forward, convert = user_agent.take_action(self.now, observation)

            if forward:
                adv_agent.users[user_agent.id]["forward"] += 1
                self.logger.info(f"{user_agent.name} forward the ad of {adv_agent.name}")
                adv_agent.memory.add_memory(f"{user_agent.id} being swayed or influenced by your advertisement", self.now)
                message.append(
                    Message(
                        agent_id=user_agent.id,
                        action="SOCIAL",
                        content=f"{user_agent.id} is going to social media.",
                    )
                )
                self.round_msg.append(
                    Message(
                        agent_id=user_agent.id,
                        action="SOCIAL",
                        content=f"{user_agent.id} is going to social media.",
                    )
                )
            if convert:
                adv_agent.users[user_agent.id]["convert"] += 1
                self.logger.info(f"{user_agent.name} convert the ad of {adv_agent.name}")
                adv_agent.memory.add_memory(f"{user_agent.id} convert your advertisement", self.now)
                adv_agent.funds += int(adv["price"])
            if not forward and not convert:
                self.logger.info(f"{user_agent.name} doesn't do anything of the ad of {adv_agent.name}")
                adv_agent.memory.add_memory(f"{user_agent.id} doesn't convert and be influenced by your advertisement",
                                            self.now)

        return message

    def user_one_step(self, agent:UserAgent,observation):
        self.play_event.wait()
        id = agent.id
        self.logger.info(f"{id} remember what you did before when advertising.")
        # 记住之前的行为
        real_ads = agent.user_pre_memory(observation[id], now=self.now)

        message = []
        cate = real_ads[0]['cate']
        ads = self.load_ads(real_ads)
        record = []
        for i in range(len(ads)):
            adv = ads[i]
            forward, convert = agent.take_action(adv, self.now)
            # feeling = agent.generate_feeling(
            #     f"{id} watched the advertisement: {adv}.", self.now
            # )
            # self.logger.info(f"{id}'s feeling: {feeling}")
            if forward:
                res = {'cate':cate,'label':'1','item':adv.id}
                record.append(res)
                self.logger.info(
                    f"{id} is going to social media to forward the advertisement: {adv.id}."
                )
                message.append(
                    Message(
                        agent_id=id,
                        action="SOCIAL",
                        content=f"{id} is going to social media.",
                    )
                )
                self.round_msg.append(
                    Message(
                        agent_id=id,
                        action="SOCIAL",
                        content=f"{id} is going to social media.",
                    )
                )
                observation = f"{id} is going to forward the advertisement."

            if convert:
                res = {'cate': cate, 'label': '2','item':adv.id}
                record.append(res)
                self.logger.info(
                    f"{id} purchased the product corresponding advertisement: {adv.id}."
                )

            if not forward and not convert:
                self.logger.info(f"{id} is doing nothing.")
                res = {'cate': cate, 'label': '0','item':adv.id}
                record.append(res)

        self.result[id] = [real_ads,record]
        return message

    def round(self):
        messages = []
        for i in range(len(self.adv_agents)):
            agent: AdvAgent = self.adv_agents[i]
            self.adv_one_step(agent)

        for i, group in self.groups.items():
            adv_agent_id = i
            adv_agent = self.adv_agents[adv_agent_id]
            adv = adv_agent.item[self.round_cnt - 1]
            msgs = self.group_one_step(group, adv, adv_agent)
            adv_agent.group.users = []
            messages.append(msgs)

        self.logger.info("Round {} finished. Advertising Data: ".format(self.round_cnt))
        return messages

    def load_ads(self,ads):
        items = self.data.items
        item_list = []
        for i in range(len(ads)):
            Ad = namedtuple('Ad', ['id', 'category', 'content'])
            cate_id = int(ads[i]['cate'])
            category = judge_cate(cate_id)
            start_idx = (cate_id - 1) * 100 + 1
            end_idx = cate_id * 100
            random_numbers = random.randint(start_idx,end_idx)
            item_id = items[str(random_numbers)]['item_id']
            content = items[str(random_numbers)]['content']
            ad = Ad(id=item_id, category=category, content=content)
            item_list.append(ad)
        # random_numbers = random.sample(range(start_idx, end_idx), ads_num)
        # idx = 1;
        # for i in random_numbers:
        #     Ad = namedtuple('Ad', ['id', 'category','content'])
        #
        #     content = items[str(i)]['content']
        #     item_id = items[str(i)]['item_id']
        #     category = items[str(i)]['category']
        #     ad = Ad(id=item_id,category=category,content=content)
        #     # ad = "the " + str(idx)  + " ad : " + "category is " + category + " and content is " + content + ".\n"
        #     item_list.append(ad)

        return item_list

    def save(self, save_dir_name):
        """Save the simulator status of current epoch"""
        utils.ensure_dir(save_dir_name)
        ID = utils.generate_id(self.config["simulator_dir"])
        file_name = f"{ID}-Round[{self.round_cnt}]-AgentNum[{self.config['user_agent_num']}]-{datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}.pkl"
        self.file_name_path.append(file_name)
        save_file_name = os.path.join(save_dir_name, file_name)
        with open(save_file_name, "wb") as f:
            dill.dump(self.__dict__, f)
        self.logger.info("Current simulator Save in: \n" + str(save_file_name) + "\n")
        self.logger.info(
            "Simulator File Path (root -> node): \n" + str(self.file_name_path) + "\n"
        )

    def user_append_item(self,observation:dict):

        for inte in self.data.interaction.items():
                user_id = inte[0]
                for inter in inte[1]:
                    label = inter['label']
                    # item_id = inter['itemid']
                    cate_id = inter['cate']

                    item_id = random.randint(int(cate_id), int(cate_id) * 100)
                    content = self.data.items[str(item_id)]['content']
                    # # 检查用户是否已经存在于字典中
                    if observation.get(user_id):
                        # 如果用户已存在，将当前记录添加到用户的记录列表中
                        observation[user_id].append({'label': label,'cate':cate_id,'content':content})
                    else:
                        # 如果用户不存在，创建一个新的记录列表，并将当前记录添加进去
                        observation[user_id] = [{'label': label,'cate':cate_id,'content':content}]

        return  observation
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "-o", "--output_file", type=str, required=True, help="Path to output file"
    )
    parser.add_argument(
        "-l", "--log_file", type=str, default="log.log", help="Path to log file"
    )
    parser.add_argument(
        "-n", "--log_name", type=str, default=str(os.getpid()), help="Name of logger"
    )
    parser.add_argument(
        "-p",
        "--play_role",
        type=bool,
        default=False,
        help="Add a user controllable role",
    )
    parser.add_argument(
        "-m", "--agent_memory", type=str, default="recagent", help="Memory mecanism"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    return args

def save_log_file(recagent):
    time_str = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    # 生成带有日期和时间的文件名
    file_name = f"./output/data/data-{time_str}.json"

    # 打开文件并写入数据
    with open(file_name, "w") as file:
        json.dump(recagent.result, file)

def main():
    args = parse_args()
    logger = utils.set_logger(args.log_file, args.log_name)
    logger.info(f"os.getpid()={os.getpid()}")
    # create config
    config = CfgNode(new_allowed=True)
    output_file = os.path.join("output/message", args.output_file)
    # 给config增加变量
    config = utils.add_variable_to_config(config, "output_file", output_file)
    config = utils.add_variable_to_config(config, "log_file", args.log_file)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config = utils.add_variable_to_config(config, "play_role", args.play_role)
    config = utils.add_variable_to_config(config, "agent_memory", args.agent_memory)
    config.merge_from_file(args.config_file)

    logger.info(f"\n{config}")

    os.environ["OPENAI_API_KEY"] = config["api_keys"][0]
    os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.com/v1"

    recagent = ESimulator(config, logger)
    recagent.load_simulator()
    messages = []
    recagent.play()

    for i in range(recagent.round_cnt + 1, config["epoch"] + 1):
        recagent.round_cnt = recagent.round_cnt + 1
        recagent.logger.info(f"Round {recagent.round_cnt}")

        message = recagent.round()
        messages.append(message)
        recagent.save(os.path.join(config["simulator_dir"]))

if __name__ == "__main__":
    main()
