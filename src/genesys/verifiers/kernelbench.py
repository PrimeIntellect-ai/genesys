import json
from datasets import load_dataset
import pandas as pd

def load_sakana_dataset(level: str = "level_1") -> pd.DataFrame:
    dataset = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive")
    df = dataset[level].to_pandas()
    return df

def transform_row_to_genesys_record(row: pd.Series) -> dict:
    """
    Convert a single row from the Sakana dataset into a Genesys record.
    This maps KernelBench fields into the Genesys schema.
    """
    record = {
        "problem_id": f"{row['Task_ID']}_{row['Kernel_Name']}",
        "source": "SakanaAI/AI-CUDA-Engineer-Archive",
        "task_type": "kernelbench",
        "prompt": "",   
        "gold_standard_solution": None,   
        "verification_info": {
            "Correct": row["Correct"],
            "CUDA_Speedup_Native": row["CUDA_Speedup_Native"],
            "CUDA_Speedup_Compile": row["CUDA_Speedup_Compile"],
            "Error": row["Error"],
            "Max_Diff": row["Max_Diff"],
            "NCU_Profile": row["NCU_Profile"],
            "Torch_Profile": row["Torch_Profile"],
            "Clang_Tidy": row["Clang_Tidy"],
        },
        "metadata": {
            "Op_Name": row["Op_Name"],
            "Level_ID": row["Level_ID"],
            "Task_ID": row["Task_ID"],
            "Kernel_Name": row["Kernel_Name"],
            "CUDA_Runtime": row["CUDA_Runtime"],
            "PyTorch_Native_Runtime": row["PyTorch_Native_Runtime"],
            "PyTorch_Compile_Runtime": row["PyTorch_Compile_Runtime"],
        },
        "llm_response": row["CUDA_Code"],  
        "response_id": f"{row['Task_ID']}_{row['Kernel_Name']}",
        "model_name": "Sakana-AI",  
        "generation_config": {},
        "machine_info": {}
    }
    return record

def build_genesys_records(level: str = "level_1") -> list:
    df = load_sakana_dataset(level)
    records = df.apply(transform_row_to_genesys_record, axis=1).tolist()
    return records

def save_records_to_jsonl(records: list, filename: str):
    with open(filename, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    records = build_genesys_records("level_1")
    print(f"Loaded {len(records)} records from level_1.")
    save_records_to_jsonl(records, "genesys_kernelbench_level1.jsonl")
    print("Saved records to genesys_kernelbench_level1.jsonl")
