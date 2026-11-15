import { useEffect, useMemo, useRef, useState } from 'react'
import { fetchClaudeCodeSetup, type ClaudeCodeSetupResponse } from '../api'
import { useAppStore } from '../store'

export const DEFAULT_HTTP_COMMAND =
  'claude mcp add --transport http model-safety-engine http://127.0.0.1:8000/mcp'
export const DEFAULT_STDIO_COMMAND =
  'claude mcp add model-safety-engine -- uv run --directory "/path/to/this/repo" tool mcp-stdio --workspace "/path/to/this/repo/workspace"'
export const DEFAULT_PLUGIN_COMMAND = 'claude --plugin-dir ./plugin'

export type SetupCommandKey = 'http' | 'stdio' | 'plugin'

export interface ResolvedSetupCommands {
  httpCommand: string
  stdioCommand: string
  pluginCommand: string
  workspace: string | null
  mcpUrl: string | null
  notes: string | null
}

export interface ClaudeCodePanelProps {
  setup?: ClaudeCodeSetupResponse | null
  isFakeMode?: boolean
  copiedKeyOverride?: SetupCommandKey | null
}

export function resolveSetupCommands(
  setup?: ClaudeCodeSetupResponse | null,
): ResolvedSetupCommands {
  const httpCommand =
    setup?.http_command?.trim() || setup?.command?.trim() || DEFAULT_HTTP_COMMAND
  const stdioCommand = setup?.stdio_command?.trim() || DEFAULT_STDIO_COMMAND
  const pluginCommand = setup?.plugin_command?.trim() || DEFAULT_PLUGIN_COMMAND

  return {
    httpCommand,
    stdioCommand,
    pluginCommand,
    workspace: setup?.workspace ?? null,
    mcpUrl: setup?.mcp_url ?? null,
    notes: setup?.notes ?? setup?.description ?? null,
  }
}

interface ClipboardNavigatorLike {
  clipboard?: {
    writeText?: (text: string) => Promise<void>
  }
}

export async function copyCommandToClipboard(command: string): Promise<boolean> {
  const nav = (globalThis as { navigator?: ClipboardNavigatorLike }).navigator
  if (!nav?.clipboard?.writeText) {
    return false
  }
  try {
    await nav.clipboard.writeText(command)
    return true
  } catch {
    return false
  }
}

export function ClaudeCodePanel({
  setup: setupProp,
  isFakeMode: isFakeModeProp,
  copiedKeyOverride,
}: ClaudeCodePanelProps = {}) {
  const storeIsFakeMode = useAppStore((s) => s.isFakeMode)
  const isFakeMode = isFakeModeProp !== undefined ? isFakeModeProp : storeIsFakeMode

  const [fetchedSetup, setFetchedSetup] = useState<ClaudeCodeSetupResponse | null>(
    setupProp ?? null,
  )
  const [copiedKey, setCopiedKey] = useState<SetupCommandKey | null>(null)
  const copyResetTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    return () => {
      if (copyResetTimerRef.current !== null) {
        clearTimeout(copyResetTimerRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (setupProp !== undefined || isFakeMode) {
      return
    }
    let cancelled = false
    void fetchClaudeCodeSetup()
      .then((res) => {
        if (cancelled) return
        setFetchedSetup(res)
      })
      .catch(() => {
        // Keep local defaults when engine is offline
      })
    return () => {
      cancelled = true
    }
  }, [setupProp, isFakeMode])

  const effectiveSetup = setupProp !== undefined ? setupProp : fetchedSetup
  const resolved = useMemo(() => resolveSetupCommands(effectiveSetup), [effectiveSetup])
  const activeCopiedKey = copiedKeyOverride !== undefined ? copiedKeyOverride : copiedKey

  const handleCopy = async (key: SetupCommandKey, command: string) => {
    const ok = await copyCommandToClipboard(command)
    if (!ok) return
    setCopiedKey(key)
    if (copyResetTimerRef.current !== null) {
      clearTimeout(copyResetTimerRef.current)
    }
    copyResetTimerRef.current = setTimeout(() => {
      setCopiedKey((prev) => (prev === key ? null : prev))
    }, 2000)
  }

  const commandBlocks: ReadonlyArray<{
    key: SetupCommandKey
    title: string
    badge?: string
    subtitle: string
    command: string
  }> = [
    {
      key: 'http',
      title: 'Live HTTP connection',
      badge: 'Recommended with 3D view open',
      subtitle:
        'Connects Claude Code to the running `tool serve` HTTP server so layer paint, focus, and timeline updates stream live into this browser window.',
      command: resolved.httpCommand,
    },
    {
      key: 'stdio',
      title: 'Stdio MCP server',
      subtitle:
        'Starts the engine directly over stdio against the same workspace folder when `tool serve` is not already running.',
      command: resolved.stdioCommand,
    },
    {
      key: 'plugin',
      title: 'Claude Code plugin',
      subtitle:
        'Loads the local plugin directory with the investigation skill, MCP configuration, and reviewer subagent.',
      command: resolved.pluginCommand,
    },
  ]

  return (
    <div className="claude-setup-panel" role="region" aria-label="Claude Code setup">
      <div className="claude-setup-intro">
        <p style={{ margin: 0 }}>
          Claude Code and this web application connect to the same local engine, workspace, and
          SQLite / Parquet experiment record on your own hardware, in sync between terminal and
          browser.
        </p>
        {isFakeMode && (
          <div style={{ marginTop: '0.45rem' }}>
            <span className="badge badge-placeholder">
              fixture mode — showing default local setup commands
            </span>
          </div>
        )}
      </div>

      <div className="claude-command-grid">
        {commandBlocks.map((block) => {
          const isCopied = activeCopiedKey === block.key
          return (
            <section
              key={block.key}
              className="claude-command-card"
              aria-label={block.title}
            >
              <div className="claude-command-header">
                <div>
                  <div className="claude-command-title-row">
                    <strong>{block.title}</strong>
                    {block.badge && <span className="badge badge-established">{block.badge}</span>}
                  </div>
                  <span className="chart-card-subtitle">{block.subtitle}</span>
                </div>
                <button
                  type="button"
                  className={`btn ${isCopied ? 'btn-active' : ''}`}
                  aria-label={`Copy ${block.title} command`}
                  onClick={() => {
                    void handleCopy(block.key, block.command)
                  }}
                >
                  {isCopied ? 'Copied' : 'Copy'}
                </button>
              </div>

              <pre className="claude-command-pre">
                <code>{block.command}</code>
              </pre>
            </section>
          )
        })}
      </div>
    </div>
  )
}
