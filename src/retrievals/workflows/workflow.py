"""A simple implementation for workflow and pipeline refer to Microsoft GraphRAG"""

import gc
import logging
import time
from dataclasses import dataclass as dc_dataclass
from dataclasses import field
from typing import Any, Generic, List, Optional, TypeVar, cast

from pydantic import BaseModel
from pydantic import Field as pydantic_Field

"""Represent a step in a workflow."""
PipelineWorkflowStep = dict[str, Any]

"""Represent a configuration for a workflow."""
PipelineWorkflowConfig = dict[str, Any]


class PipelineWorkFlowReference(BaseModel):
    """Represent a reference to a workflow, and can optionally be the workflow itself."""

    name: Optional[str] = pydantic_Field(description="Name of the workflow.", default=None)
    steps: List[PipelineWorkflowStep] = pydantic_Field(description="The optional steps for the workflow.", default=None)
    config: PipelineWorkflowConfig = pydantic_Field(
        description="The optional configuration for the workflow.", default=None
    )


@dc_dataclass
class PipelineRunStats:
    """Pipeline running stats."""

    total_runtime: float = field(default=0)
    """Float representing the total runtime."""

    num_documents: int = field(default=0)
    """Number of documents."""

    input_load_time: float = field(default=0)
    """Float representing the input load time."""

    workflows: dict[str, dict[str, float]] = field(default_factory=dict)
    """A dictionary of workflows."""


Context = TypeVar("Context")


class Workflow(Generic[Context]):
    # https://github.com/microsoft/datashaper/blob/main/python/datashaper/datashaper/workflow/workflow.py
    def __init__(self):
        pass


@dc_dataclass
class WorkflowToRun:
    """Workflow to run class definition."""

    workflow: 'workflow'
    config: dict[str, Any]
