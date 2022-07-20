import ast
import json
import os

# To access config like attribute
from addict import Dict as AddDict


class Config(object):
    def __init__(self, filenames=None, kwargs=None, set_defaults=True):
        # Experiment configs
        self.exp_dir = None
        self.exp_name = None
        self.overwrite_exp = False
        if set_defaults:
            self.set_defaults()

        if filenames:
            for filename in filenames.split("+"):
                if not os.path.exists(filename):
                    filename = os.path.join(os.getenv("CONFIG_PATH", default="configs"), filename)
                self.update_kwargs(json.load(open(filename)), eval=False)
        if kwargs:
            self.update_kwargs(kwargs)

        self.set_exp_dir()

    def set_defaults(self):
        self.cache_dir = None
        self.allow_skip_exp = True
        self.seed = 42

        # Model Configs
        self.model_name_or_path = None
        self.tokenizer_name_or_path = None
        self.max_seq_len = 512
        self.resize_position_embeddings = False

        # Dataset Configs
        self.data_args = AddDict(
            {
                "dataset_name": "allenai/mup-full",
                "test_dataset_name": "tmp/test.jsonl",
                "batch_size": 8,
                "eval_batch_size": 8,
                "num_workers": 32,
                "use_fast_tokenizer": False,
                "max_source_length": 4096,
            }
        )

        # Compute backend configs
        self.compute_precision = "fp32"
        self.compute_strategy = "ddp"

        # Trainer configs
        self.trainer_args = AddDict(
            {
                "num_steps": 10000,
                "optimizer": "adam",
                "learning_rate": 3e-5,
                "warmup_steps": 1000,
                "weight_decay": 0.0,
                "gradient_accumulation_steps": 1,
                "save_top_k": 1,
                "limit_val_batches": 0.5,
                "val_interval": 2000,
            }
        )

    def set_exp_dir(self):
        """
        Updates the config default values based on parameters passed in from config file
        """

        if self.exp_name is not None:
            self.exp_dir = os.path.join(os.getenv("OUTPUT_PATH", default="exp_out"), self.exp_name)
        else:
            self.exp_dir = os.getenv("OUTPUT_PATH", default="exp_out")
        if os.path.exists(self.exp_dir) and self.overwrite_exp:
            print(f"Experiment directory {self.exp_dir} already exists, deleting it.")
            os.system(f"rm -rf {self.exp_dir}")
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        if self.exp_dir is not None:
            self.dev_score_file = os.path.join(self.exp_dir, "dev_scores.json")
            self.test_pred_file = os.path.join(self.exp_dir, "test_pred.txt")
            self.test_score_file = os.path.join(self.exp_dir, "test_scores.json")
            self.save_config(os.path.join(self.exp_dir, os.path.join("config.json")))
            self.finish_flag_file = os.path.join(self.exp_dir, "exp_completed.txt")

    def update_kwargs(self, kwargs, eval=True):
        for (k, v) in kwargs.items():
            if eval:
                if v in ["true", "false"]:
                    v = v.capitalize()
                try:
                    v = ast.literal_eval(v)
                except (ValueError, SyntaxError) as e:
                    v = v
            else:
                v = v
            if not hasattr(self, k):
                # check nested:
                if "." in k:
                    pass
                else:
                    print(f"{k} is not in the config, adding it.")
            if isinstance(v, dict):
                existing_value = getattr(self, k, None)
                if existing_value is not None:
                    v = AddDict(existing_value, v)
                else:
                    v = AddDict(v)
            if "." in k:  # deal with nested attributes
                k_split = k.split(".")
                root, sub_key = k_split[0], ".".join(k_split[1:])
                try:
                    setattr(getattr(self, root), sub_key, v)
                except AttributeError:
                    setattr(self, root, AddDict())
                    setattr(getattr(self, root), sub_key, v)
            else:
                setattr(self, k, v)

    def to_json(self):
        """
        Converts parameter values in config to json
        :return: json
        """
        return json.dumps(self.__dict__, indent=4, sort_keys=False)

    def to_kwargs(self):
        """
        :return: json
        """
        return json.loads(self.to_json())

    def load_config(self, filename):
        """
        Loads the config
        """
        with open(filename, "r") as fin:
            self.update_kwargs(json.load(fin), False)

    def save_config(self, filename):
        """
        Saves the config
        """
        with open(filename, "w+") as fout:
            fout.write(self.to_json())
            fout.write("\n")

    @staticmethod
    def load_config_from_json(dict_json, set_defaults=False):
        """
        Loads the config from json string
        """
        config = Config(set_defaults=set_defaults)
        config.update_kwargs(dict_json, False)
        return config


if __name__ == "__main__":
    config = Config()
    config.tokenizer_name_or_path = "nnnnnnn"
    config.save_config("/tmp/config.json")
    config = Config()
    config.load_config("/tmp/config.json")
    print(config.to_json())
