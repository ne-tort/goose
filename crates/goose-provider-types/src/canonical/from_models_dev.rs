//! Conversion of the models.dev `api.json` payload into canonical model
//! entries.
//!
//! These functions were lifted verbatim out of the `build_canonical_models`
//! binary (in the `goose` crate) so that the bundled-data builder AND any
//! downstream consumer that converts a freshly-fetched `api.json` at runtime
//! share ONE conversion semantics — the bundled canonical data and a
//! runtime-fetched overlay are then built by exactly the same code. The
//! binary is now a thin caller over this module; nothing here changes its
//! behavior.
//!
//! Boundary: pure conversion only — no network, no filesystem. Callers fetch
//! and parse the JSON, then hand the parsed value to [`process_model`] /
//! [`pick_winning_variant`] (and normalize provider slugs with
//! [`normalize_provider_name`]).

use super::{canonical_name, CanonicalModel, Limit, Modalities, Modality, Pricing, ThinkingMode};
use anyhow::{Context, Result};
use serde_json::Value;

/// The context-window default applied when a models.dev entry declares no
/// `limit.context` (the builder's long-standing fallback).
pub const DEFAULT_CONTEXT_LIMIT: usize = 128_000;

/// Map a models.dev provider slug onto the canonical provider slug used by
/// the registry (the few publishers whose models.dev key differs from their
/// canonical name).
pub fn normalize_provider_name(provider: &str) -> &str {
    match provider {
        "llama" => "meta-llama",
        "xai" => "x-ai",
        "mistral" => "mistralai",
        _ => provider,
    }
}

fn get_string(value: &Value, field: &str) -> Option<String> {
    value.get(field).and_then(|v| v.as_str()).map(String::from)
}

fn get_thinking_mode(canonical_id: &str, value: &Value) -> Option<ThinkingMode> {
    value
        .get("thinking_mode")
        .and_then(|v| v.as_str())
        .and_then(|mode| serde_json::from_value(Value::String(mode.to_string())).ok())
        .or_else(|| inferred_thinking_mode(canonical_id))
}

fn inferred_thinking_mode(canonical_id: &str) -> Option<ThinkingMode> {
    match canonical_id {
        "anthropic/claude-fable-5" => Some(ThinkingMode::AlwaysOnAdaptive),
        "anthropic/claude-fable-5.1" => Some(ThinkingMode::AlwaysOnAdaptive),
        "anthropic/claude-opus-5" => Some(ThinkingMode::Adaptive),
        "anthropic/claude-opus-5.5" => Some(ThinkingMode::AlwaysOnAdaptive),
        "anthropic/claude-opus-4.6" => Some(ThinkingMode::Adaptive),
        "anthropic/claude-opus-4.7" => Some(ThinkingMode::Adaptive),
        "anthropic/claude-opus-4.8" => Some(ThinkingMode::Adaptive),
        "anthropic/claude-sonnet-4.6" => Some(ThinkingMode::Adaptive),
        "anthropic/claude-sonnet-5" => Some(ThinkingMode::Adaptive),
        _ => None,
    }
}

fn parse_modalities(model_data: &Value, field: &str) -> Vec<Modality> {
    model_data
        .get("modalities")
        .and_then(|m| m.get(field))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .filter_map(|s| {
                    serde_json::from_value(serde_json::Value::String(s.to_string())).ok()
                })
                .collect()
        })
        .unwrap_or_else(|| vec![Modality::Text])
}

/// Convert ONE models.dev model object into its canonical form. Returns the
/// registry key (the model name with the provider prefix stripped) alongside
/// the [`CanonicalModel`].
pub fn process_model(
    model_id: &str,
    model_data: &Value,
    normalized_provider: &str,
) -> Result<(String, CanonicalModel)> {
    let name = model_data["name"]
        .as_str()
        .with_context(|| format!("Model {} missing name", model_id))?;

    let canonical_id = canonical_name(normalized_provider, model_id);

    let modalities = Modalities {
        input: parse_modalities(model_data, "input"),
        output: parse_modalities(model_data, "output"),
    };

    let cost = match model_data.get("cost") {
        Some(c) if !c.is_null() => Pricing {
            input: c.get("input").and_then(|v| v.as_f64()),
            output: c.get("output").and_then(|v| v.as_f64()),
            cache_read: c.get("cache_read").and_then(|v| v.as_f64()),
            cache_write: c.get("cache_write").and_then(|v| v.as_f64()),
        },
        _ => Pricing {
            input: None,
            output: None,
            cache_read: None,
            cache_write: None,
        },
    };

    let limit = Limit {
        context: model_data
            .get("limit")
            .and_then(|l| l.get("context"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_CONTEXT_LIMIT as u64) as usize,
        output: model_data
            .get("limit")
            .and_then(|l| l.get("output"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
    };

    let canonical_model = CanonicalModel {
        id: canonical_id.clone(),
        name: name.to_string(),
        family: get_string(model_data, "family"),
        attachment: model_data.get("attachment").and_then(|v| v.as_bool()),
        reasoning: model_data.get("reasoning").and_then(|v| v.as_bool()),
        thinking_mode: get_thinking_mode(&canonical_id, model_data),
        tool_call: model_data
            .get("tool_call")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        temperature: model_data.get("temperature").and_then(|v| v.as_bool()),
        knowledge: get_string(model_data, "knowledge"),
        release_date: get_string(model_data, "release_date"),
        last_updated: get_string(model_data, "last_updated"),
        modalities,
        open_weights: model_data.get("open_weights").and_then(|v| v.as_bool()),
        cost,
        limit,
    };

    let model_name = canonical_id
        .strip_prefix(&format!("{}/", normalized_provider))
        .unwrap_or(model_id)
        .to_string();

    Ok((model_name, canonical_model))
}

/// When several models.dev ids collapse onto the same canonical key, pick the
/// winning variant: the SHORTEST id (the un-dated family name), then the most
/// recently updated, then the most recently released, then lexicographic for
/// determinism. Returns the index into `variants`.
pub fn pick_winning_variant(variants: &[(String, CanonicalModel)]) -> usize {
    variants
        .iter()
        .enumerate()
        .min_by(|(_, (id_a, a)), (_, (id_b, b))| {
            id_a.len()
                .cmp(&id_b.len())
                .then_with(|| b.last_updated.cmp(&a.last_updated))
                .then_with(|| b.release_date.cmp(&a.release_date))
                .then_with(|| id_a.cmp(id_b))
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variant(id: &str, release: Option<&str>, updated: Option<&str>) -> (String, CanonicalModel) {
        (
            id.to_string(),
            CanonicalModel {
                id: format!("openai/{}", id),
                name: id.to_string(),
                family: None,
                attachment: None,
                reasoning: None,
                thinking_mode: None,
                tool_call: false,
                temperature: None,
                knowledge: None,
                release_date: release.map(String::from),
                last_updated: updated.map(String::from),
                modalities: Modalities::default(),
                open_weights: None,
                cost: Pricing::default(),
                limit: Limit::default(),
            },
        )
    }

    #[test]
    fn shortest_variant_wins() {
        let variants = vec![
            variant("gpt-4o-2024-08-06", Some("2024-08-06"), Some("2024-08-06")),
            variant("gpt-4o", Some("2024-05-13"), Some("2024-08-06")),
            variant("gpt-4o-2024-11-20", Some("2024-11-20"), Some("2024-11-20")),
            variant("gpt-4o-2024-05-13", Some("2024-05-13"), Some("2024-05-13")),
        ];
        let idx = pick_winning_variant(&variants);
        assert_eq!(variants[idx].0, "gpt-4o");

        let variants = vec![
            variant(
                "claude-haiku-4-5-20251001",
                Some("2025-10-16"),
                Some("2025-10-16"),
            ),
            variant("claude-haiku-4-5", Some("2025-10-16"), Some("2025-10-16")),
        ];
        let idx = pick_winning_variant(&variants);
        assert_eq!(variants[idx].0, "claude-haiku-4-5");
    }

    #[test]
    fn normalize_provider_name_maps_the_known_publishers() {
        assert_eq!(normalize_provider_name("llama"), "meta-llama");
        assert_eq!(normalize_provider_name("xai"), "x-ai");
        assert_eq!(normalize_provider_name("mistral"), "mistralai");
        assert_eq!(normalize_provider_name("openai"), "openai");
    }

    #[test]
    fn process_model_maps_a_known_fragment() {
        let data = serde_json::json!({
            "name": "Widget One",
            "family": "widget",
            "reasoning": true,
            "tool_call": true,
            "release_date": "2026-01-02",
            "last_updated": "2026-02-03",
            "modalities": { "input": ["text", "image"], "output": ["text"] },
            "cost": { "input": 2.0, "output": 8.0 },
            "limit": { "context": 200000, "output": 16000 }
        });
        let (key, model) = process_model("widget-1", &data, "acme").unwrap();
        assert_eq!(key, "widget-1");
        assert_eq!(model.id, "acme/widget-1");
        assert_eq!(model.name, "Widget One");
        assert_eq!(model.limit.context, 200_000);
        assert_eq!(model.limit.output, Some(16_000));
        assert_eq!(model.cost.input, Some(2.0));
        assert!(model.modalities.input.contains(&Modality::Image));
    }

    #[test]
    fn process_model_defaults_an_undeclared_context_limit() {
        let data = serde_json::json!({ "name": "Bare" });
        let (_, model) = process_model("bare", &data, "acme").unwrap();
        assert_eq!(model.limit.context, DEFAULT_CONTEXT_LIMIT);
        assert_eq!(model.limit.output, None);
        // An undeclared modality list defaults to text-only input.
        assert_eq!(model.modalities.input, vec![Modality::Text]);
    }
}
