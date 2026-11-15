import { useEffect, useRef, useState } from 'react'
import { sharedFakePlayer } from '../dev/fakeEvents'
import { buildWsUrl, computeBackoffDelay } from '../events'
import { formatComponentLabel, useAppStore } from '../store'
import {
  applyIncomingChatFrame,
  approvePlanFromButton,
  submitChatTurn,
} from './chatLogic'
import { SendIcon, StopIcon } from '../icons'
import { Message } from './Message'

// Shown before any model is loaded: the first step is saying which model to check.
const START_SUGGESTIONS: ReadonlyArray<string> = [
  'I want to check a model I have on this machine',
  'Which demo models can I try?',
]

const BASE_SUGGESTIONS: ReadonlyArray<string> = [
  'Did my fine-tune make the model more sycophantic?',
  'Check whether refusals weakened compared with the base model',
  'Check this model for backdoor signals against a clean reference',
]

// Whether this page load has already connected to the chat once (see connect()).
let newPageAnnounced = false

export function ChatPanel() {
  const chatMessages = useAppStore((s) => s.chatMessages)
  const draftInput = useAppStore((s) => s.draftInput)
  const selectedComponentId = useAppStore((s) => s.selectedComponentId)
  const modelMap = useAppStore((s) => s.modelMap)
  const plans = useAppStore((s) => s.plans)
  const findings = useAppStore((s) => s.findings)
  const isFakeMode = useAppStore((s) => s.isFakeMode)

  const setDraftInput = useAppStore((s) => s.setDraftInput)
  const selectComponent = useAppStore((s) => s.selectComponent)
  const openEvidenceForFinding = useAppStore((s) => s.openEvidenceForFinding)

  const [chatWsConnected, setChatWsConnected] = useState(false)
  const socketRef = useRef<WebSocket | null>(null)
  const streamingMsgIdRef = useRef<string | null>(null)
  // True from sending a message until the agent's reply is complete.
  const [awaitingReply, setAwaitingReply] = useState(false)
  const jobRunning = useAppStore((s) =>
    Object.values(s.jobs).some((j) => j.status === 'running' || j.status === 'queued'),
  )
  const showStop = !isFakeMode && chatWsConnected && (awaitingReply || jobRunning)
  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView?.({ behavior: 'auto' })
  }, [chatMessages.length])

  useEffect(() => {
    if (isFakeMode) {
      setChatWsConnected(false)
      return
    }

    let cancelled = false
    let attempt = 0
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null

    const connect = () => {
      if (cancelled || useAppStore.getState().isFakeMode) return
      let ws: WebSocket
      try {
        // The first connection after the page loads tells the engine to start with an
        // empty view; automatic reconnects later in the session do not.
        ws = new WebSocket(buildWsUrl(newPageAnnounced ? '/ws/chat' : '/ws/chat?new_page=1'))
      } catch {
        scheduleReconnect()
        return
      }
      socketRef.current = ws

      ws.onopen = () => {
        newPageAnnounced = true
        if (cancelled) return
        attempt = 0
        setChatWsConnected(true)
      }

      ws.onmessage = (evt) => {
        if (cancelled) return
        // The reply is over when the engine says so, or after a stop.
        try {
          const frame = JSON.parse(String(evt.data)) as { type?: string }
          if (frame.type === 'turn_complete' || frame.type === 'stopped') {
            setAwaitingReply(false)
          }
        } catch {
          // Not JSON; applyIncomingChatFrame handles plain text.
        }
        applyIncomingChatFrame(evt.data, streamingMsgIdRef)
      }

      ws.onerror = () => {
        // Handled by onclose
      }

      ws.onclose = () => {
        socketRef.current = null
        if (cancelled) return
        setChatWsConnected(false)
        scheduleReconnect()
      }
    }

    const scheduleReconnect = () => {
      if (cancelled || reconnectTimer !== null) return
      const delay = computeBackoffDelay(attempt, 600, 15000)
      attempt += 1
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null
        connect()
      }, delay)
    }

    connect()

    return () => {
      cancelled = true
      if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer)
      }
      if (socketRef.current) {
        const ws = socketRef.current
        socketRef.current = null
        ws.onopen = null
        ws.onmessage = null
        ws.onclose = null
        ws.onerror = null
        try {
          ws.close()
        } catch {
          // Ignore close errors
        }
      }
    }
  }, [isFakeMode])

  const sendOverSocket = (payload: string): boolean => {
    const ws = socketRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return false
    }
    ws.send(payload)
    return true
  }

  const handleSend = () => {
    if (draftInput.trim() && chatWsConnected) setAwaitingReply(true)
    void submitChatTurn({
      text: draftInput,
      sendOverSocket,
    })
  }

  // Stop button: ends the agent's reply and any running check, like a chat stop button.
  const handleStop = () => {
    sendOverSocket(JSON.stringify({ stop: true }))
    setAwaitingReply(false)
  }

  const handleApprovePlan = (planId: string) => {
    void approvePlanFromButton(planId, {
      sendOverSocket,
    })
  }

  const handleResetMockConversation = () => {
    sharedFakePlayer.stop()
    sharedFakePlayer.loadPresetModel('qwen25_sycophancy')
    sharedFakePlayer.seedInitialMockConversation()
  }

  const suggestionItems = !modelMap
    ? START_SUGGESTIONS
    : selectedComponentId
      ? [`What is measured at ${formatComponentLabel(selectedComponentId)}?`, ...BASE_SUGGESTIONS]
      : BASE_SUGGESTIONS

  return (
    <div className="chat-panel">
      {isFakeMode && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '0.5rem',
            padding: '0.4rem 0.85rem',
            borderBottom: '1px solid var(--border-subtle)',
            background: 'var(--bg-subtle)',
            fontSize: '0.76rem',
          }}
        >
          <span>Mock conversation (placeholder fixture mode)</span>
          <button
            type="button"
            className="btn"
            style={{ padding: '0.15rem 0.45rem', fontSize: '0.74rem' }}
            onClick={handleResetMockConversation}
          >
            Reset mock conversation
          </button>
        </div>
      )}

      <div className="chat-messages">
        {chatMessages.length === 0 ? (
          <div className="chat-empty">
            {modelMap
              ? 'Ask a safety question about this model in plain language. Click a suggestion below to fill the message box, or click any block in the model view to attach it as context.'
              : 'Start by telling the assistant which model you want to check: one of your own, or one of the demo models. Then ask your safety question in plain language.'}
            {!isFakeMode && !chatWsConnected && (
              <div style={{ marginTop: '0.45rem', fontSize: '0.78rem' }}>
                Not connected to the engine. Start the tool with <code>tool serve</code>, then
                reload this page.
              </div>
            )}
          </div>
        ) : (
          chatMessages.map((msg) => {
            const resolvedPlan = msg.planId ? plans[msg.planId] : undefined
            const resolvedFinding = msg.findingId
              ? findings.find((f) => f.finding_id === msg.findingId)
              : undefined
            return (
              <Message
                key={msg.id}
                message={msg}
                resolvedPlan={resolvedPlan}
                resolvedFinding={resolvedFinding}
                onApprovePlan={handleApprovePlan}
                onShowEvidence={openEvidenceForFinding}
              />
            )
          })
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggestions help with the first message only; after that the conversation leads. */}
      {chatMessages.length === 0 && (
        <div className="chat-suggestions" aria-label="Prompt suggestions">
          {suggestionItems.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              className="suggestion-chip"
              onClick={() => setDraftInput(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}

      <div className="chat-composer">
        {selectedComponentId && (
          <div className="context-chip-row">
            <span className="context-chip">
              <span>
                {formatComponentLabel(selectedComponentId)} ({selectedComponentId})
              </span>
              <button
                type="button"
                aria-label={`Remove ${selectedComponentId} context`}
                onClick={() => selectComponent(null)}
              >
                ×
              </button>
            </span>
          </div>
        )}

        <div className="composer-input-row">
          <textarea
            aria-label="Chat message input"
            className="composer-textarea"
            rows={2}
            placeholder="Ask a question"
            value={draftInput}
            onChange={(e) => setDraftInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
          />
          {showStop ? (
            <button
              type="button"
              className="icon-btn stop-btn"
              aria-label={jobRunning ? 'Stop the reply and the running check' : 'Stop the reply'}
              title="Stop"
              onClick={handleStop}
            >
              <StopIcon />
            </button>
          ) : (
            <button
              type="button"
              className="icon-btn send-btn"
              aria-label="Send message"
              title="Send"
              disabled={draftInput.trim().length === 0}
              onClick={handleSend}
            >
              <SendIcon />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
