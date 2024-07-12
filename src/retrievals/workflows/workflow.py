"""A simple implementation for workflow and pipeline refer to Microsoft GraphRAG"""

import gc
import logging
import time
from dataclasses import dataclass as dc_dataclass
from typing import Any, Generic, List, Optional, TypeVar, cast

from pydantic import BaseModel
from pydantic import Field as pydantic_Field

logger = logging.getLogger(__name__)

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


def load_workflows(workflows_to_load):
    workflow_graph: dict[str, WorkflowToRun] = {}

    global anonymous_workflow_count
    for reference in workflows_to_load:
        name = reference.name
        is_anonymous = name is None or name.strip() == ""
        if is_anonymous:
            name = f"Anonymous Workflow {anonymous_workflow_count}"
            anonymous_workflow_count += 1
        name = cast(str, name)

        config = reference.config
        workflow = create_workflow(
            name or "MISSING NAME!",
            reference.steps,
            config,
        )
        workflow_graph[name] = WorkflowToRun(workflow, config=config or {})


def create_workflow(
    name: str,
    steps: list[PipelineWorkflowStep] | None = None,
    config: PipelineWorkflowConfig | None = None,
    memory_profile: bool = False,
):
    """Create a workflow from the given config."""

    steps = steps
    return Workflow(
        verbs={},
        schema={
            "name": name,
            "steps": steps,
        },
        validate=False,
        memory_profile=memory_profile,
    )


def run_pipeline(workflows: List):
    loaded_workflows = load_workflows(workflows)
    workflows_to_run = loaded_workflows.workflows

    for workflow_to_run in workflows_to_run:
        gc.collect()

        workflow = workflow_to_run.workflow
        workflow_name: str = workflow.name
        # last_workflow = workflow_name

        logger.info("Running workflow: %s...", workflow_name)

        # workflow_start_time = time.time()
        # result = await workflow.run(context, callbacks)
        #
        # output = await emit_workflow_output(workflow)
        # yield PipelineRunResult(workflow_name, output, None)
