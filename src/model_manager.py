from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer
)

class ModelManager:
    def __init__(self, model_path: str, model_type: str = "auto"):
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.model, self.tokenizer = self._load()

    def _load(self):
        if self.model_type == "bart":
            model = BartForConditionalGeneration.from_pretrained(self.model_path)
            tokenizer = BartTokenizer.from_pretrained(self.model_path)
        elif self.model_type == "t5":
            model = T5ForConditionalGeneration.from_pretrained(self.model_path)
            tokenizer = T5Tokenizer.from_pretrained(self.model_path)
        elif self.model_type == "auto":
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return model, tokenizer

    @property
    def get_model(self):
        assert self.model is not None, "Model not loaded"
        return self.model

    @property
    def get_tokenizer(self):
        return self.tokenizer
    
    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, model_type: str = "auto"):
        """
        Alternate constructor that loads model/tokenizer from a saved checkpoint folder.
        """
        return cls(model_path=checkpoint_dir, model_type=model_type)

