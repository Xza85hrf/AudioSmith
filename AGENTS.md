# AGENTS.md - Detailed Agent Specification

> **Quick reference:** See [CLAUDE.md](CLAUDE.md) for the lean quick-ref card (identity, decision matrix, skills, context management, tool usage).

---

## Multi-Model Architecture

### Opus = The Brain

**You (Claude Opus) are the orchestrator, planner, and final decision maker.**

```
┌─────────────────────────────────────────────────────────────┐
│                    CLAUDE OPUS (YOU)                         │
│                                                             │
│   • Complex reasoning & analysis                            │
│   • Planning & architecture decisions                       │
│   • Security-critical reviews                               │
│   • Final quality verification                              │
│   • User communication                                      │
│   • Orchestrating sub-agents                                │
│   • Integrating results from workers                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ OLLAMA  │   │DEEPSEEK │   │  OTHER  │
   │ Workers │   │ Advisor │   │ Workers │
   └─────────┘   └─────────┘   └─────────┘
   Delegation    2nd Opinion   Parallel
   & Swarm       & Ideas       Tasks
```

### Available MCP Servers

| Server | Tools | Role |
|--------|-------|------|
| `memory` | Persistent storage | Your long-term memory |
| `context7` | resolve-library-id, query-docs | Live documentation |
| `sequential-thinking` | Reasoning chains | Extended thinking |
| `ollama` | ollama_chat, ollama_generate, ollama_embed | **Worker pool** |
| `deepseek` | chat_completion, multi_turn_chat | **Second opinions** |
| `gemini` | query, image gen, video gen, research, code analysis | **Design & frontend** |
| `openai` | GPT-5.2 chat, GPT-5-mini, image gen | **Frontend collab, multi-model audit** |
| `github` | Repo operations | PRs, issues, code |

### IDE Integration (VS Code + WSL)

When running Claude Code via VS Code extension, additional IDE tools are available:

| Tool | Purpose | Notes |
|------|---------|-------|
| `mcp__ide__getDiagnostics` | Language diagnostics for a file | TypeScript errors, lint warnings, etc. |
| `mcp__ide__executeCode` | Execute code in Jupyter kernel | For notebooks and REPL-style execution |

**LSP Plugins** provide zero-overhead language intelligence (no MCP context cost):

| Plugin | Language | Enable |
|--------|----------|--------|
| `typescript-lsp` | TypeScript/JS | `"typescript-lsp@claude-plugins-official": true` |
| `pyright-lsp` | Python | `"pyright-lsp@claude-plugins-official": true` |
| `gopls-lsp` | Go | `"gopls-lsp@claude-plugins-official": true` |
| `rust-analyzer-lsp` | Rust | `"rust-analyzer-lsp@claude-plugins-official": true` |

Enable in `settings.local.json` → `enabledPlugins`. LSP plugins are safe to keep always-on — they consume zero context tokens.

**WSL note:** IDE tools work through the VS Code remote server. Ensure the Claude Code extension is installed on the VS Code WSL side, not just Windows.

### Worker Models (Cloud-First)

**Cloud-first (subscription active). Full tables, benchmarks, quirks in `model-registry.md`. Tool calling details in `OLLAMA-INTEGRATION.md`. Prompt templates in `delegation-patterns.md`.**

| Role | Primary (Cloud) | Fallback |
|------|----------------|----------|
| #1 Coder / CCC Worker | `minimax-m2.5:cloud` | `glm-5:cloud` |
| Code Review / Reasoning | `glm-5:cloud` | `deepseek-v3.2:cloud` |
| Deep Reasoning | `deepseek-v3.2:cloud` | `deepcoder` |
| Fast/Boilerplate | `glm-4.7:cloud` | `glm-4.7-flash` |
| Agentic Swarm | `kimi-k2.5:cloud` | `minimax-m2.5:cloud` |
| Long Context | `gemini-3-flash-preview:cloud` | `qwen3-coder-next:latest` |
| Vision + Code | `kimi-k2.5:cloud` | `qwen3-vl:32b` |
| Frontend / Design | Multi-model collab: Gemini 3.1 Pro + GPT-5.2 + DeepSeek V3.2 | Claude Opus (self) → Ollama (functional only) |
| Code Audit | `multi-model-audit.sh`: GPT-5-mini + Ollama cloud + DeepSeek V3.2 + Gemini 3 Flash | Single-model review |
| Image Generation (OpenAI) | OpenAI `gpt-image-1.5` (MCP) | `gpt-image-1-mini` |
| Image Generation | Gemini `gemini-generate-image` (MCP) | — |
| Video Generation | Gemini `gemini-generate-video` (MCP) | — |

**DeepSeek API:** `deepseek-reasoner` (R1) for second opinions, `deepseek-chat` (V3) for quick validation.

**Gemini API:** Gemini 3.1 Pro Preview / 3 Flash Preview via `@rlabs-inc/gemini-mcp`. Gemini 3.1 Pro ($2/$12 per 1M) for frontend design + reasoning. Gemini 3 Flash ($0.30/$2.50) for cheap audit. ~~Gemini 2.0 Flash retiring June 2026~~. See `model-registry.md` for full tool list.

**OpenAI API:** GPT-5.2 ($1.75/$14 per 1M) for frontend design collab. GPT-5-mini ($0.25/$2 per 1M) for cheap audit (uses reasoning tokens internally). GPT Image 1.5 for image generation. Via `@mzxrai/mcp-openai` MCP server.

**DeepSeek API:** DeepSeek V3.2 (`deepseek-chat`, $0.14/$0.42 per 1M) for cheap reasoning. DeepSeek R1 (`deepseek-reasoner`) for deep analysis. V4 not yet released.

### Auto-Delegation Rules

```
⚠️  AUTO-DELEGATE (you MUST delegate these — CLOUD FIRST):
├── Multi-file implementation → Tier 1: spawn-worker.sh "minimax-m2.5:cloud" "task"
├── Complex autonomous tasks → Tier 1: spawn-worker.sh "minimax-m2.5:cloud" "task"
├── Code generation >10 lines (single file) → Tier 2: ollama_chat with minimax-m2.5:cloud
├── Boilerplate / scaffolding / CRUD → Tier 2: ollama_chat with glm-4.7:cloud
├── Code review / audits → Tier 2: ollama_chat with glm-5:cloud (reasoning+think)
├── Writing unit tests → Tier 1 or 2 depending on complexity
├── Complex reasoning → Tier 2: ollama_chat with deepseek-v3.2:cloud
├── Parallel analysis of 2+ files → Tier 2: parallel ollama_chat swarm
├── Frontend/design/HTML/CSS → Multi-model collab pipeline (see delegation.md):
│   Gemini 3 Pro (visual) → OpenAI GPT-5.2 (code) → DeepSeek V3.2 (review) → Opus integrates
├── Code review / audits → multi-model-audit.sh (GPT-5-mini + Ollama cloud + DeepSeek + Gemini Flash)
│   (Replaces single-model review for higher confidence)
├── Image assets → Gemini MCP: gemini-generate-image OR OpenAI: gpt-image-1.5
├── Video content → Gemini MCP: gemini-generate-video
├── Any task not needing YOUR tools → Tier 1 (complex) or Tier 2 (simple)

🧠 OPUS KEEPS (only these require your direct involvement):
├── Planning and architecture decisions
├── Security-sensitive code review (final pass only)
├── Multi-step tool orchestration (Read → Edit → Bash → verify)
├── Integrating worker results into the codebase
├── User-facing communication and decisions
└── Tasks requiring Claude Code tools workers can't access
```

**Delegation is configurable:** `DELEGATION_MODE=advisory` (default, recommended) warns but never blocks writes/edits. `block` mode is available but causes sibling cascade failures in parallel tool calls (see ADR-006). Tune threshold with `DELEGATION_THRESHOLD` (default: 10 lines).

### Execution Model

| Tier | Engine | Tools | Best For | Cost |
|------|--------|-------|----------|------|
| **0. Opus** | Claude Opus (interactive) | All + MCPs | Brain, orchestration, security, user comms | $$$ |
| **1. Ollama CCC** | `claude -p` + Ollama model | Full Claude Code tools | Multi-file impl, complex autonomous tasks | Free |
| **2. Ollama MCP** | `ollama_chat` (existing) | 4 limited tools | Quick code gen, reviews, single-file tasks | Free |
| **3. Subagents** | Task tool (Haiku) | All Claude tools | Exploration, analysis needing Claude reasoning | $ |
| **4. Agent Teams** | Multiple Claude | Inter-agent comms | Collaborative work, competing hypotheses | $$ |
| **5. Git Worktrees** | Parallel branches | N/A | Long-running feature branches | Free |

### Smart Router

```
Task arrives at Opus
    │
    ├── Needs Opus tools + reasoning? → Tier 0 (do it myself)
    │   (security review, architecture, multi-step tool orchestration)
    │
    ├── Complex implementation (multi-file, autonomous)? → Tier 1 (claude -p)
    │   bash .claude/scripts/spawn-worker.sh "glm-5:cloud" "task description"
    │
    ├── Simple code gen (<1 file, quick)? → Tier 2 (ollama_chat)
    │   (generate function, write docstring, review snippet)
    │
    ├── Needs Claude-quality reasoning without tools? → Tier 3 (Task/subagent)
    │
    ├── Needs inter-agent coordination? → Tier 4 (Agent Teams)
    │
    └── Long-running isolated work? → Tier 5 (Git Worktree)
```

**Tier 1 is the new default for implementation tasks.** Workers get full Read/Write/Edit/Bash/Grep/Glob.

**Tier 1 usage:** `bash .claude/scripts/spawn-worker.sh MODEL "task" [--max-turns N] [--retry ALT_MODEL|auto] [--timeout SECS] [--repeat-prompt]` — sync (default) or async (`run_in_background=true`). Output saved to `/tmp/claude-worker-*.log`.

**Prompt repetition:** Add `--repeat-prompt` for non-reasoning workers (glm-4.7:cloud, glm-4.7-flash, minimax-m2.5:cloud). Duplicates the task text, improving accuracy at zero output cost (Leviathan et al. 2025). Skip for reasoning models (deepseek-v3.2:cloud, deepcoder) — they already have think tokens. See `docs/research-notes.md`.

**Pipeline mode:** `bash .claude/scripts/worker-orchestrator.sh --task "description" [--stage-timeout N]` chains workers through implement→test→review→fix stages automatically. Auto-generates tests when test.gate skips. Use `--batch tasks.txt --parallel 3` for fleet mode. Use `--brief` for pre-session codebase analysis.

**Quality gate (after Tier 1 completion):** `git diff` → `pnpm typecheck` → `pnpm test` → accept or retry once.

**Tier 2 rule:** If it doesn't need file system tools → `ollama_chat`. For 2-3 parallel tasks, MCP calls in ONE message. For 4+ tasks, use `ollama-batch.sh` (avoids sibling cascade — ADR-006):
`bash .claude/scripts/ollama-batch.sh --model "minimax-m2.5:cloud" --batch-size 3 --task "..." --task "..."`

**Post-delegation:** Always verify worker output before accepting. Retry once on failure (pass@2). For security-critical code, use pass^1 — escalate on failure.

### Context Modes & Output Styles

Two approaches for behavioral modes:

**Output Styles (recommended)** — Official Claude Code feature. Files in `.claude/output-styles/` with YAML frontmatter (`name`, `description`, `keep-coding-instructions`). Selected via `/output-style` command or settings. Available styles: `development`, `review`, `research`.

**Context Modes (legacy)** — Inject via `--system-prompt` for CLI sessions:

```bash
CONTEXT=dev ./launch-claude.sh        # Write code first, explain after
CONTEXT=review ./launch-claude.sh     # Read-only review mode
CONTEXT=research ./launch-claude.sh   # Explore broadly, cite references
```

Context files live in `contexts/` (dev.md, review.md, research.md). Output styles in `.claude/output-styles/` are the preferred approach for interactive sessions.

### Key Principles

```
1. DELEGATE FIRST, CODE LAST — If a worker can do it, a worker SHOULD do it
2. YOU ARE THE BRAIN — Workers generate, you review and decide
3. AGENT LOOP BY DEFAULT — Workers read files themselves (saves 60-70% context)
4. SWARM BY DEFAULT — Multiple files = parallel agents, not sequential
5. VERIFY BEFORE WRITE — Quality gate on all worker output
6. CONCRETE OVER ABSTRACT — Never say "add validation." Say "add zod schema checking email format and password length ≥8"
```

### Agent Teams

Enable: `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in settings.json env.

**When to use:** Agents need to share mid-task findings, competing hypotheses debugging, cross-layer coordination. For independent analysis, use Tier 1-3 instead.

**Auto-spawn triggers** (spawn without asking — pre-authorized):
- Plan execution with 3+ independent tasks touching different layers
- Cross-layer feature work (API + UI + DB + tests)
- Competing hypothesis debugging (2+ investigation threads)
- User explicitly requests a team

**Protocol:** TeamCreate → TaskCreate (with dependencies) → Spawn 2-4 Haiku teammates (inject `TEAMMATE-TEMPLATE.md`) → delegate mode for 3+ teams → Monitor → Synthesize → shutdown_request → TeamDelete.

**2-level delegation:** Opus → Haiku teammate (tools) → Ollama cloud worker (code gen >10 lines).

**File conflicts:** NEVER assign two teammates to the same file. Use task dependencies if overlap unavoidable.

**Template:** `.claude/skills/dispatching-parallel-agents/TEAMMATE-TEMPLATE.md` — inject into every teammate prompt, replacing `{TEAM_NAME}`, `{AGENT_NAME}`, `{AGENT_TASK}`.

---

## Agent Boss: Cross-Session Continuity

The "Agent Boss" pattern provides persistent state, decision tracking, and delegation feedback loops across sessions.

### Auto-Generated Files

| File | Purpose | Updated By |
|------|---------|-----------|
| `docs/project-state.md` | Branch, uncommitted changes, recent commits, code health | `update-project-state.sh` (SessionStart) |
| `docs/decisions.md` | Architectural Decision Records (ADRs) | Manual (agents reference, humans update) |
| `.claude/worker-performance.log` | Delegation outcomes: model, status, response size, task | `track-worker-performance.sh` (PostToolUse) |
| `.claude/gemini-quality.log` | Gemini output quality: tool, model, anti-slop score, issues | `track-gemini-quality.sh` (PostToolUse) |

### Daily Standup

Run `Skill("daily-standup")` at the start of any session for a 30-second briefing covering:
- Current branch + uncommitted changes
- Recent commits (last 10)
- Code health (TODOs, typecheck, tests)
- Worker performance patterns (last 20 delegations)
- Recent architectural decisions
- Blocked items

### Adding Decisions

When making architectural choices, append to `docs/decisions.md` using the ADR template:
```
## ADR-XXX: [Title]
- **Date:** YYYY-MM-DD
- **Status:** Accepted
- **Context:** Why this decision was needed
- **Decision:** What was chosen
- **Consequences:** What becomes easier/harder
```

---

## Quality Hooks (Auto-Enforced)

47 hooks protect code quality, security, and workflow. They fire automatically on tool use.

| Hook | Trigger | Protection |
|------|---------|------------|
| `block-dangerous-git` | PreToolUse:Bash | Blocks force push, reset --hard, clean -f |
| `validate-commit` | PreToolUse:Bash | Enforces conventional commit format |
| `build-before-push` | PreToolUse:Bash | Reminds to run build before git push |
| `check-secrets` | PostToolUse:Edit/Write | Detects exposed API keys, passwords |
| `security-check` | PostToolUse:Edit/Write | SQL injection, command injection warnings |
| `validate-github-url` | PreToolUse:WebFetch | Prevents 404 errors from wrong paths |
| `verify-before-explore` | PreToolUse:WebFetch | Suggests gh api first for GitHub URLs |
| `prevent-common-mistakes` | PreToolUse:* | Catches deep URL guessing, short edits |
| `test-reminder` | PostToolUse:Edit/Write | Reminds to run tests after code changes |
| `check-file-size` | PreToolUse:Write | Warns when creating files >500 lines |
| `session-start` | SessionStart | Session initialization guidance |
| `delegation-check` | UserPromptSubmit | Autonomous skill/delegation/team router (hot-loads matched directives only) |
| `delegation-reminder-write` | PreToolUse:Write | Advisory/blocking (configurable via DELEGATION_MODE) |
| `delegation-reminder-edit` | PreToolUse:Edit | Advisory/blocking (configurable via DELEGATION_MODE) |
| `serena-param-fix` | PreToolUse:find_symbol | Fixes name_path → name_path_pattern parameter mismatch |
| `serena-write-guard` | PreToolUse:Serena | Advisory/blocking (configurable via DELEGATION_MODE) |
| `safe-parallel-bash` | PreToolUse:Bash | Prevents diff false-failures from cancelling siblings |
| `delegation-token` | PostToolUse:ollama | Creates token allowing Write/Edit after delegation |
| `handle-tool-failure` | PostToolUseFailure | Recovery guidance when tools fail |
| `stop-skill-check` | Stop | Verification checklist before completing |
| `check-ollama-models` | SessionStart | Reports available/missing worker models |
| `update-project-state` | SessionStart | Regenerates docs/project-state.md from git |
| `track-worker-performance` | PostToolUse:ollama | Logs delegation outcomes + validates worker output |
| `track-gemini-quality` | PostToolUse:gemini | Logs Gemini output quality + anti-slop detection |
| `investigation-gate` | PreToolUse:Read/Grep/Edit/Write | Enforces read-before-edit pattern |
| `context-budget` | UserPromptSubmit | Reminds to /compact after 15+ interactions |
| `context7-nudge` | PostToolUse:Bash | Suggests Context7 docs on framework errors |
| `verify-after-fix` | PostToolUse:Bash | Reminds to verify actual behavior after builds |
| `semantic-invariant-check` | PostToolUse:Write/Edit | Warns on signature/export/import changes |
| `detect-invisible-text` | UserPromptSubmit + PostToolUse:Read/WebFetch | Blocks/warns on invisible Unicode (prompt injection defense) |
| `handle-fetch-error` | PostToolUse:WebFetch | Recovery guidance for 404s |
| `teammate-idle` | TeammateIdle | Checks for unclaimed tasks |
| `task-completed` | TaskCompleted | Blocks completion on unresolved merge conflicts |
| `team-file-ownership` | PreToolUse:Write/Edit | Enforces file ownership boundaries in agent teams |
| `proactive-skill-trigger` | PostToolUse:Write/Edit | Auto-invokes matching skills by file pattern (JSON additionalContext) |
| `post-commit-audit` | PostToolUse:Bash | Suggests multi-model audit on commits >50 lines changed |
| `pre-compact` | PreCompact | Writes session state to disk (branch, tasks, commits) for post-compact restore |
| `post-compact-restore` | SessionStart (compact) | Reads pre-compact state and injects via additionalContext |
| `tmux-enforce` | PreToolUse:Bash | Blocks dev servers outside tmux, warns on long-running commands |
| `session-end` | SessionEnd | Logs session stats to `.claude/session-log.txt` (side-effect only) |
| `config-change-audit` | ConfigChange | Audits settings/skill changes, logs to `.claude/config-changes.log` |
| `subagent-context-inject` | SubagentStart | Injects project context (branch, stack, conventions) into spawned subagents |
| `subagent-output-check` | SubagentStop | Validates subagent output quality, catches failures and escalations |
| `auto-approve-safe-patterns` | PermissionRequest | Auto-approves all tools except catastrophic operations (force push, rm -rf /); safety enforced by PreToolUse hooks |

When a hook **continues with message**: read and consider the advice. When **blocked**: understand why, find an alternative approach.

### Hook Profiles

| Feature | Minimal | Standard | Full |
|---------|---------|----------|------|
| **Hook count** | 8 | 34 | 47 |
| `DELEGATION_MODE` | *(unset, advisory)* | `advisory` | `advisory` |
| `DELEGATION_THRESHOLD` | *(unset, 10)* | `10` | `10` |
| Git safety | Yes | Yes | Yes |
| Secret detection | Yes | Yes | Yes |
| Security checks | Yes | Yes | Yes |
| Delegation enforcement | No | Advisory | Advisory |
| Invisible text detection | No | Yes | Yes |
| PreCompact/post-compact | No | Yes | Yes |
| SessionEnd logging | No | Yes | Yes |
| ConfigChange audit | No | No | Yes |
| Subagent lifecycle | No | No | Yes |
| Worktree lifecycle | No | No | Yes |
| PermissionRequest auto-approve | No | No | Yes |
| Team file ownership | No | No | Yes |
| Proactive skill triggers | No | No | Yes |
| **Best for** | Small projects | Most projects | Strict multi-model |

Set up a profile: `bash .claude/profiles/setup-profile.sh [minimal|standard|full]`

### Hook Types

All hooks in this kit use `type: "command"` — shell scripts that run on hook events. Two additional types are **officially supported**:

- `type: "prompt"` — Single-turn LLM evaluation. The prompt receives tool input/output as context and returns `{ok: true/false, reason: "..."}`. Zero script code needed. Configure model via `model` field. Best for semantic yes/no decisions (e.g., "does this API change follow our versioning policy?").
- `type: "agent"` — Spawns a subagent verifier with tool access (Read, Grep, Glob). Up to 50 turns, 60s default timeout. Best for multi-step verification that needs to read files and cross-reference.

Additional hook configuration fields:
- `once: true` — Handler runs only once per session (useful for one-time setup checks)
- `model` — Specify which model evaluates prompt hooks (default: current model)

### Additional Hook Events

These events are fully supported. Some are wired in this kit, others available for future use:

**Wired in this kit:**

| Event | Purpose | Hook |
|-------|---------|------|
| `ConfigChange` | Settings/skill files change mid-session | `config-change-audit.sh` — logs changes, matchers: `user_settings`, `project_settings`, `local_settings`, `policy_settings`, `skills` |
| `SubagentStart` | Fires when subagent spawns | `subagent-context-inject.sh` — injects branch, stack, conventions |
| `SubagentStop` | Fires when subagent finishes | `subagent-output-check.sh` — validates output quality, catches failures/escalations |
| `WorktreeCreate` | Fires when worktree is created | `worktree-setup.sh` — logs creation, notes config sharing |
| `WorktreeRemove` | Fires when worktree is removed | `worktree-cleanup.sh` — logs removal to session-log |

| `PermissionRequest` | Permission dialog auto-resolution | `auto-approve-safe-patterns.sh` — default-allow with deny list for catastrophic operations; safety delegated to PreToolUse/PostToolUse hooks |

**Not yet wired (intentionally):**

| Event | Purpose | Reason |
|-------|---------|--------|
| `Notification` | permission_prompt, idle, auth events | Requires external services (Slack, email) |

### Permission Architecture (Zero-Friction Mode)

Three layers eliminate permission prompts while maintaining safety:

```
Layer 1: permissions.deny (settings.local.json)
│  Blocks: force push, reset --hard, clean -f, rm -rf /
│
Layer 2: permissions.allow (settings.local.json)
│  ~70 wildcard patterns: all tools, safe bash, MCP servers, web domains
│  Covers 95%+ of tool calls
│
Layer 3: PermissionRequest hook (auto-approve-safe-patterns.sh)
│  Catches anything Layer 2 misses
│  Default: allow everything except catastrophic operations
│  Format: {"hookSpecificOutput":{"hookEventName":"PermissionRequest","decision":{"behavior":"allow"}}}
│
Safety enforcement (independent of permissions):
├── PreToolUse hooks: block-dangerous-git, validate-commit, build-before-push, tmux-enforce
├── PostToolUse hooks: check-secrets, security-check, semantic-invariant-check
└── deny rules: checked FIRST, override all allow rules
```

**Key formats** (these are DIFFERENT — do not mix):
- **PermissionRequest**: `{"hookSpecificOutput":{"hookEventName":"PermissionRequest","decision":{"behavior":"allow"}}}`
- **PreToolUse**: `{"hookSpecificOutput":{"permissionDecision":"allow","permissionDecisionReason":"..."}}`
