# AGENTS.md — Multi-Model Agent Specification

## Architecture

Orchestrator = brain (planner, final decisions). Workers = code generation, reviews, parallel tasks. The brain is whichever model runs Claude Code (Opus, GLM-5, etc.).

@.claude/reference/mcp-ecosystem.md

### Worker Models (Cloud-First)

Ollama Cloud runs on NVIDIA B300 data center hardware. Fixed pricing ($0/$20/$100/mo, no overages). Featured cloud models: `nemotron-3-super:cloud`, `minimax-m2.7:cloud`, `deepseek-v3.1:671b-cloud`, `deepseek-v3.2:cloud`, `gpt-oss:120b-cloud`, `gpt-oss:20b-cloud`, `qwen3-coder:480b-cloud`.

<!-- MODEL-TABLE-START -->
| Role | Primary | Fallback |
|------|---------|----------|
| #1 Coder | `glm-5:cloud` | `deepseek-v3.1:671b-cloud` |
| #2 Coder / Review | `minimax-m2.7:cloud` | `minimax-m2.5:cloud` |
| Deep Reasoning | `deepseek-v3.2:cloud` | `deepcoder` |
| Fast/Boilerplate | `gpt-oss:20b-cloud` | `glm-5:cloud` |
| Agentic Swarm | `nemotron-3-super:cloud` | `deepseek-v3.2:cloud` |
| Long Context | `nemotron-3-super:cloud` | `qwen3-coder-next:cloud` |
| Vision + Code | `qwen3.5:cloud` | `qwen3-vl:32b` |
| Frontend | Brain + Open Pencil/Figma → Skill(frontend-design-pro) + workers | Stitch mockup-to-code + 21st.dev components |
| Code Audit | `multi-model-audit.sh` (codex-mini + Ollama + DeepSeek + Gemini + Codex) | Single-model |
| Image | Gemini `gemini-generate-image` (4K) / OpenAI `gpt-image-1.5` | — |
| Video | LTX-2 `generate-video-ltx2.sh` | Gemini `gemini-generate-video` |
| Voice/TTS | Qwen3-TTS (Ollama local) | Gemini `gemini-speak` |
| Transcription | faster-whisper (local) | Gemini `gemini-youtube` |
| Desktop Apps | CLI-Anything (GIMP, Blender, Audacity, OBS, LibreOffice) | Playwright (screen-click fallback) |
| Codex Plan Review | `codex exec --oss -m qwen3.5:cloud` | `codex exec --oss -m glm-5:cloud` |
| Codex Terminal Tasks | `codex exec --oss -m glm-5:cloud` (shell/DevOps/CLI) | `codex exec --oss -m qwen3.5:cloud` |
<!-- MODEL-TABLE-END -->

Delegation, execution tiers, smart router: `.claude/rules/delegation.md`.

@.claude/reference/ollama-launch.md

### Agent Definitions

<!-- AGENT-TABLE-START -->
| Agent | Role | Model | Teams |
|-------|------|-------|-------|
| `arch-reviewer` | Architecture review specialist. Audits code for SOLID vio... | haiku | review |
| `audit-auth` | Authentication & access control specialist. Detects authe... | haiku | audit |
| `audit-data` | Data protection specialist. Detects sensitive data exposu... | haiku | audit |
| `audit-deps` | Dependencies & config specialist. Detects known CVEs, ins... | haiku | audit |
| `audit-injection` | Injection security specialist. Detects SQL injection, com... | haiku | audit |
| `codebase-explorer` | Read-only codebase exploration agent. Maps architecture, ... | haiku | — |
| `coordinator` | Meta-agent for team coordination. Manages task dependenci... | haiku | feature |
| `feature-lead` | Feature team lead architect. Designs interfaces, reviews ... | sonnet | feature |
| `hypothesis-a` | Debug investigator: data & state theory. Hypothesizes roo... | haiku | debug |
| `hypothesis-b` | Debug investigator: infrastructure & config theory. Hypot... | haiku | debug |
| `hypothesis-c` | Debug investigator: logic & algorithm theory. Hypothesize... | haiku | debug |
| `impl-backend` | Backend implementer. Builds API endpoints, database queri... | haiku | feature |
| `impl-frontend` | Frontend implementer. Builds UI components, styles, clien... | haiku | feature |
| `meta-agent` | Generates new agent .md files from natural language descr... | sonnet | — |
| `perf-reviewer` | Performance review specialist. Audits code for N+1 querie... | haiku | review |
| `reviewer` | Read-only code review agent. Reviews code for bugs, secur... | sonnet | — |
| `security-reviewer` | Security review specialist. Audits code for OWASP top 10,... | haiku | review |
| `skill-reviewer` | Reviews skills, hooks, agents, and commands for quality, ... | haiku | — |
| `worker` | Implementation agent for code generation, refactoring, an... | sonnet | — |
<!-- AGENT-TABLE-END -->

### Agent Teams

Enable: `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`. Use for 3+ independent cross-layer tasks.
Protocol: TeamCreate → TaskCreate → Spawn 2-4 teammates → Monitor → shutdown_request → TeamDelete.
Rules: No file ownership overlap. 2-level delegation: Brain → teammate → Ollama worker.

Team presets (`.claude/team-presets/`): `audit` (4 specialists), `debug` (3 hypotheses), `feature` (lead + 2 implementers), `review` (3 specialists).

## Cross-Session Continuity

State: `docs/project-state.md` (auto), `docs/decisions.md` (ADRs), `.claude/worker-performance.log`. Session briefing: `Skill("daily-standup")`.

@.claude/reference/turnstone.md

@.claude/reference/hooks.md

@.claude/reference/native-features.md
