"""Validated data models used by the local analysis pipeline."""

from typing import List
from pydantic import BaseModel, Field


class FileSummary(BaseModel):
    title: str
    purpose: str
    detailed_description: str = ""
    document_type: str = "unknown"
    responsibilities: List[str] = Field(default_factory=list)
    concepts: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    important_dates: List[str] = Field(default_factory=list)
    interfaces: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    suggested_actions: List[str] = Field(default_factory=list)


class RelationshipDecision(BaseModel):
    related: bool
    relationship_type: str = "semantic"
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    evidence: List[str] = Field(default_factory=list)
