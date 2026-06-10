"""Custom governance kernel that wraps a Strands agent.

The :class:`StrandsKernel` is an ``agent-governance-toolkit`` ``BaseIntegration``
that knows how to wrap and unwrap a Strands :class:`~strands.Agent`. Wrapping
returns a :class:`GovernedStrandsAgent` which enforces the full governance
lifecycle around every agent invocation and tool call:

    pre_execute  ->  agent / tool runs  ->  post_execute (drift check)

If a policy is violated at any stage a :class:`PolicyViolationError` is raised.
"""

from __future__ import annotations

from typing import Any

from agent_governance_toolkit import (
    BaseIntegration,
    ExecutionContext,
    GovernancePolicy,
    PolicyInterceptor,
    PolicyViolationError,
)


class StrandsKernel(BaseIntegration):
    """Governance integration for Strands agents.

    Bridges a :class:`GovernancePolicy` to the Strands runtime by producing a
    :class:`GovernedStrandsAgent` wrapper. One kernel can wrap many agents; each
    wrapped agent gets its own :class:`ExecutionContext`.
    """

    def __init__(self, policy: GovernancePolicy) -> None:
        """Initialise the kernel with a governance policy.

        Args:
            policy: The loaded governance policy to enforce.
        """
        super().__init__(policy)
        # Interceptor validates tool calls against the policy before dispatch.
        self.interceptor = PolicyInterceptor(policy)

    def wrap(self, agent: Any) -> "GovernedStrandsAgent":
        """Wrap a Strands agent, returning a governed proxy.

        Args:
            agent: The raw Strands ``Agent`` to govern.

        Returns:
            A :class:`GovernedStrandsAgent` enforcing the policy lifecycle.
        """
        # Each governed agent carries an ExecutionContext that accumulates the
        # audit trail (call count, tokens, tool calls, drift scores).
        ctx = ExecutionContext(
            policy=self.policy,
            agent_id=getattr(agent, "name", None) or "strands-agent",
        )
        return GovernedStrandsAgent(original=agent, kernel=self, ctx=ctx)

    def unwrap(self, governed: "GovernedStrandsAgent") -> Any:
        """Return the original, ungoverned agent.

        Args:
            governed: A previously wrapped governed agent.

        Returns:
            The original Strands ``Agent`` instance.
        """
        return governed._original


class GovernedStrandsAgent:
    """A Strands agent proxy that enforces governance on every interaction."""

    def __init__(
        self,
        original: Any,
        kernel: StrandsKernel,
        ctx: ExecutionContext,
    ) -> None:
        """Initialise the governed agent.

        Args:
            original: The wrapped Strands ``Agent``.
            kernel: The owning :class:`StrandsKernel`.
            ctx: The execution context tracking this agent's audit trail.
        """
        self._original = original
        self._kernel = kernel
        self._ctx = ctx

    def __call__(self, prompt: str, **kwargs: Any) -> Any:
        """Invoke the agent with the full governance lifecycle.

        Args:
            prompt: The user prompt to send to the agent.
            **kwargs: Additional keyword arguments forwarded to the agent.

        Returns:
            The agent's result.

        Raises:
            PolicyViolationError: If the prompt is blocked pre-execution, or if
                output drift exceeds the policy threshold post-execution.
        """
        # 1. PRE-EXECUTE — validate the inbound prompt against the policy.
        #    A blocked decision raises immediately; nothing reaches the model.
        pre = self._kernel.pre_execute(self._ctx, prompt, **kwargs)
        if pre.blocked:
            raise PolicyViolationError(pre.reason, agent_id=self._ctx.agent_id)

        # 2. EXECUTE — run the underlying Strands agent.
        result = self._original(prompt, **kwargs)

        # 3. POST-EXECUTE — score the output for drift / policy compliance.
        #    Exceeding the configured drift_threshold is a violation.
        post = self._kernel.post_execute(self._ctx, prompt, result)
        if post.blocked:
            raise PolicyViolationError(post.reason, agent_id=self._ctx.agent_id)

        # 4. Return the (now-governed) result.
        return result

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a named tool through the policy interceptor.

        The interceptor validates the arguments before dispatch. If the policy
        rewrites arguments (e.g. redaction), the modified arguments are used.

        Args:
            tool_name: The name of the tool to invoke.
            arguments: The arguments to pass to the tool.

        Returns:
            The tool's return value.

        Raises:
            PolicyViolationError: If the tool call is blocked by the policy.
        """
        # Validate the tool call against the policy before dispatching it.
        result = self._kernel.interceptor.validate_tool_call(
            self._ctx, tool_name, arguments
        )
        if result.blocked:
            raise PolicyViolationError(result.reason, agent_id=self._ctx.agent_id)

        # Honour any argument modifications the policy applied (e.g. redaction).
        effective_args = (
            result.modified_arguments
            if result.modified_arguments is not None
            else arguments
        )

        # Dispatch to the underlying Strands tool registry.
        tool_fn = self._original.tool_registry[tool_name]
        return tool_fn(**effective_args)

    def get_context(self) -> ExecutionContext:
        """Return the live execution context for this agent.

        Returns:
            The :class:`ExecutionContext` accumulating this agent's audit trail.
        """
        return self._ctx

    def get_audit_summary(self) -> dict[str, Any]:
        """Return a serialisable summary of the governance audit trail.

        Returns:
            A dict with ``agent_id``, ``call_count``, ``total_tokens``,
            ``tool_calls`` and ``drift_scores`` drawn from the context.
        """
        return {
            "agent_id": self._ctx.agent_id,
            "call_count": self._ctx.call_count,
            "total_tokens": self._ctx.total_tokens,
            "tool_calls": self._ctx.tool_calls,
            "drift_scores": self._ctx.drift_scores,
        }
