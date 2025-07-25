from attr import attrib, attrs
import json

@attrs
class PrefillModelConfig:
    pred_model = attrib(type=str) 
    num_labels = attrib(type=int)
    mtype = attrib(type=str)
    activation = attrib(type=str)
    path = attrib(type=str, default="")
    max_length = attrib(type=int, default=1024)
    max_batch_size = attrib(type=int, default=512)

@attrs
class PrefillPredictorConfig:
    model = attrib(type=PrefillModelConfig)
    @classmethod
    def from_json(cls, config_path):
        with open(config_path) as config_file:
            config = json.load(config_file)
            return PrefillPredictorConfig.from_dict(config)

    @classmethod
    def from_dict(cls, config):
        config["model"] = PrefillModelConfig(**config["model"])
        return cls(**config)
        
    @classmethod
    def to_json(cls, config, config_path):
        content = {}
        content["model"] = config.model.__dict__
        print("json content: ", content)
        with open(config_path, "w") as outfile: 
            json.dump(content, outfile)