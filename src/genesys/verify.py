import json
import ast
from genesys.verifier.verifier import verify
from pydantic_config import BaseConfig, parse_argv

class Config(BaseConfig):
    file: str

def main(config: Config):
    to_verify = []
    with open(config.file, "r") as f:
        for line in f:
            d = json.loads(line)
            d["verification_info"] = ast.literal_eval(d["verification_info"])
            d["metadata"] = ast.literal_eval(d["metadata"])
            
            to_verify.append(d)
            
    responses = [d["llm_response"] for d in to_verify]
    verification_infos = [d["verification_info"] for d in to_verify]
    task_types = [d["task_type"] for d in to_verify]

    scores = verify(responses, verification_infos, task_types)
    
    all_results = []
    for s, d in zip(scores, to_verify):
        d["score"] = s
        all_results.append(d)
        
    out_file = f"{config.file.split('-')[-2]}_verified.jsonl"
    with open(out_file, "w") as f:
        for result in all_results:
            json.dump(result, f)
            f.write('\n')
    
if __name__ == "__main__":
    config = Config(**parse_argv())
    main(config)
