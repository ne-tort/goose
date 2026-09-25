//! Retries the turn when the provider fails or returns an empty response.
//!
//! Sits right before `ExitOnErrorOperation` so it only fires where the turn
//! would otherwise end: transient errors (already survived the HTTP-layer
//! retries) and empty responses are retried up to the configured limit, while
//! terminal errors fall through unchanged.

use anyhow::Result;
use async_trait::async_trait;

use crate::agents::provider_retry::{
    classify_error, ProviderRetryPolicy, RetryDecision, RetryLimit, PROVIDER_RETRY_ATTEMPTS_META,
};
use crate::agents::state_machine::effects::GooseEffect;
use crate::agents::state_machine::{
    applied, messages_since_kickoff, not_applicable, trailing_error, Emitter, Operation,
    OperationResult,
};
use crate::conversation::message::{Message, MessageErrorKind, SystemNotificationType};
use crate::conversation::Conversation;
use crate::session::Session;

const EMPTY_RESPONSE_MESSAGE: &str =
    "The model returned an empty response. Please resend your message to continue.";

const MAX_RETRY_ATTEMPTS_EXCEEDED: &str = "Maximum retry attempts (";

/// `MessageErrorKind` has no refusal variant, so a refusal surfaces as `Other`
/// inside the legacy error text; the provider's Display prefix is the stable
/// marker. Refusals are terminal in the legacy loop — retrying resends the
/// same refused conversation.
const REFUSAL_MARKER: &str = "Provider refused request";

pub struct ProviderErrorRetryOperation {
    policy: ProviderRetryPolicy,
}

impl ProviderErrorRetryOperation {
    pub fn new(policy: ProviderRetryPolicy) -> Self {
        Self { policy }
    }

    fn is_empty_response(message: &Message) -> bool {
        message.role == rmcp::model::Role::Assistant
            && message.error_kind().is_none()
            && message.as_concat_text() == EMPTY_RESPONSE_MESSAGE
    }

    fn message_is_refusal(message: &Message) -> bool {
        message
            .content
            .iter()
            .filter_map(|content| content.as_error())
            .any(|error| error.message.contains(REFUSAL_MARKER))
    }

    fn reset_conversation(conversation: &Conversation) -> Result<Conversation> {
        let messages = messages_since_kickoff(conversation)?;
        let kickoff = conversation.len() - messages.len();
        Ok(Conversation::new_unvalidated(
            conversation.messages()[..=kickoff].to_vec(),
        ))
    }

    /// Attempts are counted on the kickoff message because a retry replaces
    /// the conversation with everything up to and including it — the only
    /// message that outlives the attempt it belongs to.
    fn attempts(&self, messages: &[Message]) -> u32 {
        messages
            .first()
            .and_then(|message| self.message_meta(message, PROVIDER_RETRY_ATTEMPTS_META))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as u32
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for ProviderErrorRetryOperation {
    fn name(&self) -> &'static str {
        "provider_error_retry"
    }

    async fn run(
        &self,
        _session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let kind = match trailing_error(conversation) {
            Some(kind) => Some(kind),
            None if conversation.last().is_some_and(Self::is_empty_response) => None,
            None => return not_applicable(),
        };

        if let Some(kind) = kind {
            let is_refusal = conversation.last().is_some_and(Self::message_is_refusal);
            if classify_error(kind) == RetryDecision::Terminal || is_refusal {
                return not_applicable();
            }
        }

        let messages = messages_since_kickoff(conversation)?;
        // A reactive compaction makes the kickoff agent-invisible and carries
        // its content in the summary instead; resetting to it would leave a
        // conversation inference cannot act on, silently swallowing the error.
        // Leave the turn-ending error visible, as before this operation.
        if messages
            .first()
            .is_some_and(|message| !message.is_agent_visible())
        {
            return not_applicable();
        }
        // RetryOperation exhausts its own attempts by appending a synthetic
        // error message ("Maximum retry attempts exceeded"); retrying that
        // would loop between the two operations forever.
        if messages.iter().any(|message| {
            message
                .content
                .iter()
                .filter_map(|content| content.as_error())
                .any(|error| {
                    error.kind == MessageErrorKind::Other
                        && error.message.starts_with(MAX_RETRY_ATTEMPTS_EXCEEDED)
                })
        }) {
            return not_applicable();
        }
        let attempts = self.attempts(messages);
        if let RetryLimit::Finite(limit) = self.policy.max_retries {
            if attempts >= limit {
                return not_applicable();
            }
        }

        let next_attempt = attempts + 1;
        let limit_display = match self.policy.max_retries {
            RetryLimit::Finite(limit) => limit.to_string(),
            RetryLimit::Infinite => "infinite".to_string(),
        };
        tracing::warn!(
            "retrying provider error (attempt {next_attempt}/{limit_display}), waiting {:?}",
            self.policy.interval
        );
        emit.message(Message::assistant().with_system_notification(
            SystemNotificationType::ProgressMessage,
            format!("Provider error, retrying (attempt {next_attempt}/{limit_display})..."),
        ))
        .await;

        tokio::select! {
            biased;
            _ = emit.cancelled() => return not_applicable(),
            _ = tokio::time::sleep(self.policy.interval) => {}
        }

        let mut reset = Self::reset_conversation(conversation)?;
        if let Some(kickoff) = reset.messages_mut().last_mut() {
            self.set_message_meta(
                kickoff,
                PROVIDER_RETRY_ATTEMPTS_META,
                serde_json::json!(next_attempt),
            );
        }
        applied([reset.into()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goose_providers::errors::ProviderError;

    #[test]
    fn refusal_error_message_is_detected_as_refusal() {
        let refusal = Message::from_provider_error(&ProviderError::Refusal {
            details: "violates policy".to_string(),
            category: None,
        });
        assert!(ProviderErrorRetryOperation::message_is_refusal(&refusal));

        let server_error = Message::from_provider_error(&ProviderError::ServerError("boom".into()));
        assert!(!ProviderErrorRetryOperation::message_is_refusal(
            &server_error
        ));

        let empty = Message::assistant().with_text("regular reply");
        assert!(!ProviderErrorRetryOperation::message_is_refusal(&empty));
    }
}
