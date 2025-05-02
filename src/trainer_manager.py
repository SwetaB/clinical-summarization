import os
import pandas as pd
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

class TrainManager:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, save_dir, training_args: dict, trainer_class=Trainer):
        self.trainer_class = trainer_class
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_dir = save_dir
        self.training_args_dict = training_args

        self.trainer = self._build_trainer()

    def _build_trainer(self):
        args = TrainingArguments(
            output_dir=self.save_dir,
            num_train_epochs=self.training_args_dict["num_train_epochs"],
            per_device_train_batch_size=self.training_args_dict["per_device_batch_size"],
            per_device_eval_batch_size=self.training_args_dict["per_device_batch_size"],
            learning_rate=self.training_args_dict["learning_rate"],
            warmup_steps=self.training_args_dict["warmup_steps"],
            weight_decay=0.01,
            logging_dir=os.path.join(self.save_dir, "logs"),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_strategy="epoch",
            logging_steps=100,
            report_to="none"
        )

        collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        trainer = self.trainer_class(
                model=self.model,                         
                args=args,                  
                train_dataset=self.train_dataset,         
                eval_dataset=self.val_dataset,            
                tokenizer=self.tokenizer,                 
                data_collator=collator          
            )

        return trainer

    def _save_model(self):
        final_dir = os.path.join(self.save_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        self.trainer.save_model(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        print(f"Model saved to {final_dir}")

    def _save_metrics(self):
        metrics = self.trainer.state.log_history
        metrics_file = os.path.join(self.save_dir, "training_metrics.csv")
        pd.DataFrame(metrics).to_csv(metrics_file, index=False)
        print(f"Training metrics saved to {metrics_file}")
    
    def train(self):
        print("Starting training...")
        self.trainer.train()
        self._save_model()
        self._save_metrics()

    