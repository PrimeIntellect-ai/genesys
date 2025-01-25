from pydantic import BaseModel, Field

# todo
class LLMJudgeVerification(BaseModel):
    type: str = Field("llm_judge", const=True)
