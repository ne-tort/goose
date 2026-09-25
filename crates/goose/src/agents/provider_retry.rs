//! Turn-level retry policy for provider errors shared by the legacy agent
//! loop and the state machine.
//!
//! The HTTP layer (`goose-provider-types/src/retry.rs`) already retries
//! individual transient requests; this policy only sees errors that survived
//! that layer (plus empty responses), so the two layers never double-retry.
//!
//! Retrying authentication, context-length, or credits errors is pointless or
//! harmful, so they are terminal and pass through untouched.

use std::time::Duration;

use crate::config::Config;
use crate::conversation::message::MessageErrorKind;

pub const DEFAULT_MAX_RETRIES: u32 = 3;
pub const DEFAULT_RETRY_INTERVAL_SECONDS: u64 = 5;

const GOOSE_PROVIDER_ERROR_RETRIES: &str = "GOOSE_PROVIDER_ERROR_RETRIES";
const GOOSE_PROVIDER_RETRY_INTERVAL_SECONDS: &str = "GOOSE_PROVIDER_RETRY_INTERVAL_SECONDS";

/// Metadata key on the kickoff message where turn-level provider retry
/// attempts are counted. State-machine retries reset the conversation to the
/// kickoff, so the kickoff message is the only place a counter survives.
pub const PROVIDER_RETRY_ATTEMPTS_META: &str = "provider_retry_attempts";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryLimit {
    Finite(u32),
    Infinite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderRetryPolicy {
    pub max_retries: RetryLimit,
    pub interval: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryDecision {
    Retry,
    Terminal,
}

pub fn load_policy() -> ProviderRetryPolicy {
    let max_retries = config_value_as_string(GOOSE_PROVIDER_ERROR_RETRIES)
        .map(|raw| parse_retries(&raw))
        .unwrap_or(RetryLimit::Finite(DEFAULT_MAX_RETRIES));
    let interval_seconds = config_value_as_string(GOOSE_PROVIDER_RETRY_INTERVAL_SECONDS)
        .and_then(|raw| parse_interval_seconds(&raw))
        .unwrap_or(DEFAULT_RETRY_INTERVAL_SECONDS);

    ProviderRetryPolicy {
        max_retries,
        interval: Duration::from_secs(interval_seconds),
    }
}

/// `get_param` parses numeric env values as JSON numbers, so a raw `-1` would
/// fail to deserialize as a string; read the untyped value and stringify it.
fn config_value_as_string(key: &str) -> Option<String> {
    match Config::global().get_param::<serde_json::Value>(key) {
        Ok(serde_json::Value::String(s)) => Some(s),
        Ok(serde_json::Value::Number(n)) => Some(n.to_string()),
        _ => None,
    }
}

/// `-1` or `infinite` (case-insensitive) means unlimited retries; any
/// non-negative integer is a finite limit; anything else falls back to the
/// default.
pub fn parse_retries(raw: &str) -> RetryLimit {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("infinite") {
        return RetryLimit::Infinite;
    }
    match trimmed.parse::<i64>() {
        Ok(-1) => RetryLimit::Infinite,
        Ok(count) if count >= 0 => RetryLimit::Finite(count as u32),
        _ => RetryLimit::Finite(DEFAULT_MAX_RETRIES),
    }
}

pub fn parse_interval_seconds(raw: &str) -> Option<u64> {
    raw.trim().parse::<u64>().ok()
}

/// `MessageErrorKind::Other` covers NetworkError, ServerError, and rate-limit
/// style provider errors (see the `From<&ProviderError>` impl in
/// goose-provider-types); everything else is a dead end that retrying cannot
/// fix.
pub fn classify_error(kind: MessageErrorKind) -> RetryDecision {
    match kind {
        MessageErrorKind::Other => RetryDecision::Retry,
        MessageErrorKind::Authentication
        | MessageErrorKind::ContextLengthExceeded
        | MessageErrorKind::CreditsExhausted => RetryDecision::Terminal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_retries_accepts_negative_one_as_infinite() {
        assert_eq!(parse_retries("-1"), RetryLimit::Infinite);
    }

    #[test]
    fn parse_retries_accepts_infinite_keyword() {
        assert_eq!(parse_retries("infinite"), RetryLimit::Infinite);
        assert_eq!(parse_retries("Infinite"), RetryLimit::Infinite);
        assert_eq!(parse_retries(" INFINITE "), RetryLimit::Infinite);
    }

    #[test]
    fn parse_retries_accepts_non_negative_integers() {
        assert_eq!(parse_retries("0"), RetryLimit::Finite(0));
        assert_eq!(parse_retries("5"), RetryLimit::Finite(5));
        assert_eq!(parse_retries(" 7 "), RetryLimit::Finite(7));
    }

    #[test]
    fn parse_retries_falls_back_to_default_on_invalid_input() {
        assert_eq!(
            parse_retries("abc"),
            RetryLimit::Finite(DEFAULT_MAX_RETRIES)
        );
        assert_eq!(parse_retries(""), RetryLimit::Finite(DEFAULT_MAX_RETRIES));
        assert_eq!(parse_retries("-2"), RetryLimit::Finite(DEFAULT_MAX_RETRIES));
        assert_eq!(
            parse_retries("1.5"),
            RetryLimit::Finite(DEFAULT_MAX_RETRIES)
        );
    }

    #[test]
    fn parse_interval_seconds_accepts_non_negative_integers() {
        assert_eq!(parse_interval_seconds("0"), Some(0));
        assert_eq!(parse_interval_seconds("30"), Some(30));
        assert_eq!(parse_interval_seconds(" 9 "), Some(9));
    }

    #[test]
    fn parse_interval_seconds_rejects_invalid_input() {
        assert_eq!(parse_interval_seconds("abc"), None);
        assert_eq!(parse_interval_seconds(""), None);
        assert_eq!(parse_interval_seconds("-1"), None);
        assert_eq!(parse_interval_seconds("1.5"), None);
    }

    #[test]
    fn classify_error_retries_other_errors() {
        assert_eq!(
            classify_error(MessageErrorKind::Other),
            RetryDecision::Retry
        );
    }

    #[test]
    fn classify_error_treats_terminal_kinds_as_terminal() {
        assert_eq!(
            classify_error(MessageErrorKind::Authentication),
            RetryDecision::Terminal
        );
        assert_eq!(
            classify_error(MessageErrorKind::ContextLengthExceeded),
            RetryDecision::Terminal
        );
        assert_eq!(
            classify_error(MessageErrorKind::CreditsExhausted),
            RetryDecision::Terminal
        );
    }

    #[test]
    fn load_policy_defaults_when_nothing_is_set() {
        let _guard = env_lock::lock_env([
            (GOOSE_PROVIDER_ERROR_RETRIES, None::<&str>),
            (GOOSE_PROVIDER_RETRY_INTERVAL_SECONDS, None::<&str>),
        ]);

        let policy = load_policy();

        assert_eq!(policy.max_retries, RetryLimit::Finite(DEFAULT_MAX_RETRIES));
        assert_eq!(
            policy.interval,
            Duration::from_secs(DEFAULT_RETRY_INTERVAL_SECONDS)
        );
    }

    #[test]
    fn load_policy_reads_env_values() {
        let _guard = env_lock::lock_env([
            (GOOSE_PROVIDER_ERROR_RETRIES, Some("-1")),
            (GOOSE_PROVIDER_RETRY_INTERVAL_SECONDS, Some("17")),
        ]);

        let policy = load_policy();

        assert_eq!(policy.max_retries, RetryLimit::Infinite);
        assert_eq!(policy.interval, Duration::from_secs(17));
    }

    #[test]
    fn load_policy_falls_back_to_defaults_on_invalid_env_values() {
        let _guard = env_lock::lock_env([
            (GOOSE_PROVIDER_ERROR_RETRIES, Some("not-a-number")),
            (GOOSE_PROVIDER_RETRY_INTERVAL_SECONDS, Some("soon")),
        ]);

        let policy = load_policy();

        assert_eq!(policy.max_retries, RetryLimit::Finite(DEFAULT_MAX_RETRIES));
        assert_eq!(
            policy.interval,
            Duration::from_secs(DEFAULT_RETRY_INTERVAL_SECONDS)
        );
    }

    #[test]
    fn provider_retry_attempts_meta_key_matches_convention() {
        assert_eq!(PROVIDER_RETRY_ATTEMPTS_META, "provider_retry_attempts");
    }
}
