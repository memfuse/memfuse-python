from pydantic import BaseModel, Field
from typing import List


class MultipleChoiceResponse(BaseModel):
    """Response model for multiple choice questions"""
    index: int = Field(..., description="The index of the chosen answer (0-based)")
    reasoning: str = Field(..., description="Brief explanation of the choice")
    description: str = Field(..., description="Short description of the conversation content")


class DistractorResponse(BaseModel):
    """Response model for generating multiple choice distractors"""
    distractors: List[str] = Field(..., description="List of plausible but incorrect answer choices")