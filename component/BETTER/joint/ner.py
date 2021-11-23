import logging
from typing import Any, Mapping, Sequence, Tuple, Union, List
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

class NERPrediction:
    """Class for NER Prediction."""

    def __init__(self, model: str, model_device: str = None) -> None:
        """Initializes NER Class.

        Args:
            model: Model name
            model_device: model device
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForTokenClassification.from_pretrained(model)
        self.model_device = model_device
        # self.postprocessor = TokenClassificationPostprocessor(
        #     tokenizer, aggregation_strategy="simple"
        # )
        self.logger = logging.getLogger(__name__)


    def analyze(
        self, query: List[str]
    ):
        """Query Analyze.

        Args:
            query: query
        """
        orig_to_tok_index = []
        tokens = []
        for i, word in enumerate(query):
            orig_to_tok_index.append(len(tokens))
            word_tokens = self.tokenizer.tokenize(" " + word)
            for sub_token in word_tokens:
                tokens.append(sub_token)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        original_input_ids = torch.tensor([input_ids[i] for i in orig_to_tok_index], dtype=torch.int64).to(self.model_device)
        input_mask = torch.tensor([1] * len(original_input_ids), dtype=torch.int64).to(self.model_device)

        outputs = self.model(input_ids=original_input_ids.unsqueeze(0), attention_mask=input_mask.unsqueeze(0))
        predictions = torch.argmax(outputs[0], dim=2).detach().cpu()[0].tolist()

        return [predictions]
