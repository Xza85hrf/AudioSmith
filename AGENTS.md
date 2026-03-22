# AGENTS.md — Multi-Model Agent Specification

## Architecture

Orchestrator = brain (planner, final decisions). Workers = code generation, reviews, parallel tasks. The brain is whichever model runs Claude Code (Opus, GLM-5, etc.).

### MCP Servers

| Server | Role |
|--------|------|
| `memory` | Long-term memory |
| `context7` (plugin) | Live library docs |
| `ollama` | Worker pool (chat, generate, embed, web_search, web_fetch) |
| `deepseek` | Second opinions |
| `gemini` | Image gen, research, consultation |
| `openai` | GPT-5.4/codex-mini audit, image gen |
| `stitch` | Google Stitch — AI UI generation, mockup-to-code |
| `21st-dev-magic` | 21st.dev — premium React components from natural language |
| `github` | PRs, issues, code ops |
| `playwright` | Browser automation, E2E testing |
| `sequential-thinking` | Multi-step reasoning |
| `animate-ui` | Animated shadcn/ui component registry — Motion + Tailwind + TypeScript |
| `stripe` (disabled) | Payments, subscriptions, invoices |
| `clerk` (disabled) | Auth SDK snippets, patterns |
| `cloudflare-docs` (disabled) | Cloudflare documentation search |
| `open-pencil` (optional) | Open-source Figma — .fig read/write, 87 AI tools, Tailwind JSX export. Add when desktop app running. |

Disabled servers are configured in `~/.claude.json` but turned off per-project via `disabledMcpServers`. Enable with `/mcp` when needed.
On-demand CLIs (not always-loaded): `cloudflare-docs.sh` (Cloudflare docs), `context-mode.sh` (FTS5 knowledge base), `openpencil` CLI (design file ops).

**Local dev:** `slim` (slim.sh) — HTTPS local domains, path routing, public URL sharing. Not an MCP — direct CLI.

IDE: `mcp__ide__getDiagnostics` (TS errors), `mcp__ide__executeCode` (Jupyter). LSP plugins: `typescript-lsp`, `pyright-lsp`, `gopls-lsp`, `rust-analyzer-lsp` (zero context cost).

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

### Auto-Delegation

<!-- DELEGATION-START -->
```
DELEGATE (MUST — cloud first):
├── Multi-file impl → Tier 1: spawn-worker.sh "glm-5:cloud"
├── Code gen >10 lines → Tier 2: ollama_chat glm-5:cloud
├── Boilerplate/CRUD → Tier 2: gpt-oss:20b-cloud
├── Code review → multi-model-audit.sh (includes Codex 5th voice)
├── Plan review → codex exec --oss (second-opinion on architecture)
├── Terminal/DevOps → codex exec --oss -m glm-5:cloud (shell, Makefile, CLI)
├── Tests → Tier 1/2
├── Reasoning → Tier 2: deepseek-v3.2:cloud
├── Frontend design → Open Pencil (.fig) / Figma (cloud) → Stitch → 21st.dev → frontend-design-pro → workers
├── Backend → Skill(backend-design) + workers + audit
├── Security → Skill(security-review) + brain auth logic + worker support
├── Refactoring → Skill(code-refactoring) + workers + audit
├── Image → Gemini/OpenAI MCP
├── Video → LTX-2 / Gemini
├── Desktop app control → CLI-Anything (open-source apps) / Playwright (proprietary)
├── Any non-tool task → Tier 1 (complex) or Tier 2 (simple)

BRAIN KEEPS:
├── Planning, architecture
├── Security review (final pass)
├── Multi-step tool orchestration
├── Worker result integration
├── User communication
└── Tasks needing Claude Code tools
```
<!-- DELEGATION-END -->

### Execution Tiers

| Tier | Engine | Best For | Cost |
|------|--------|----------|------|
| 0 | Orchestrator (interactive) | Brain, security, orchestration | $$$ |
| 1 | `claude -p` + Ollama | Multi-file impl, autonomous tasks | Free |
| 2 | `ollama_chat` MCP | Quick code gen, reviews | Free |
| 3 | Task tool (Haiku) | Exploration, Claude reasoning | $ |
| 4 | Agent Teams | Collaborative, competing hypotheses | $$ |
| 5 | Git Worktrees | Long-running branches | Free |
| 6 | Codex CLI (`codex exec --oss`) | Plan review, terminal tasks, 5th audit voice | Free (Ollama) |

### Smart Router

Brain tools + reasoning? → Tier 0 | Complex multi-file? → Tier 1 | Simple gen? → Tier 2 | Claude reasoning? → Tier 3 | Coordination? → Tier 4 | Long-running? → Tier 5 | Plan review/terminal? → Tier 6

Tier 1 = default for impl. Details: `.claude/rules/delegation.md`.

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

| File | Purpose |
|------|---------|
| `docs/project-state.md` | Branch, changes, health (auto-generated) |
| `docs/decisions.md` | ADRs: `## ADR-XXX: [Title]` |
| `docs/COSMOS.md` | Dashboard architecture, topology, Phase 2 roadmap |
| `.claude/worker-performance.log` | Delegation outcomes |

`Skill("daily-standup")` for session briefing.

@.claude/reference/turnstone.md

@.claude/reference/hooks.md

@.claude/reference/native-features.md
