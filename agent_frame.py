import concurrent.futures
from llm_frame import LLMFrame
from agent_models import EvalAgent

class AgentFrame:
    def __init__(self, model_name, llm_name):
        self.model_name = model_name
        self.llm_frame = LLMFrame(llm_name)

    def run(self, user):

        # Initialize the agent for new user
        agent = EvalAgent(self.model_name, self.llm_frame)
        
        # avoid toooooo long history news
        if len(user["history_news"]) > 100:
            user["history_news"] = user["history_news"][-100:]
        
        # build the memory
        agent.build_memory(user["history_news"])

        preds = []
        labels = []

        # execute the prediction in parallel
        def process_test(test):
            user_preds = agent.predict(test["test_news"])
            return user_preds, test["labels"]
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(user["tests"]), 16)) as executor:
            results = list(executor.map(process_test, user["tests"]))
            
        for user_preds, label in results:
            if len(user_preds) != len(label):
                continue
            preds.append(user_preds)
            labels.append(label)
        
        return preds, labels
