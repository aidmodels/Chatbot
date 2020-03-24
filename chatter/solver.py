from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from mlpm.solver import Solver
from datetime import datetime

class ChatbotSolver(Solver):
    def __init__(self, toml_file=None):
        super().__init__(toml_file)
        # Do you Init Work here
        self.chatbot = ChatBot('AID_CHATBOT',
            logic_adapters=[
                'chatterbot.logic.BestMatch',
                'chatterbot.logic.SpecificResponseAdapter',
                'chatterbot.logic.MathematicalEvaluation',
        ])
        trainer = ChatterBotCorpusTrainer(self.chatbot)
        # Train the chatbot based on the english corpus
        trainer.train(
           "chatterbot.corpus.english",
           "chatterbot.corpus.english.greetings",
           "chatterbot.corpus.english.conversations")
        self.ready()

    def infer(self, data):
        # if you need to get file uploaded, get the path from input_file_path in data
        result = self.chatbot.get_response(data["input"]).serialize()
        result['created_at']=datetime.timestamp(result['created_at'])
        return {"output": result}
