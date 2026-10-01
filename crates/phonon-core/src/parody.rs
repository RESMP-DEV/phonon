//! Deterministic-first local parody transforms.
//!
//! Parody is an explicit post-dictation transform. It must preserve protected
//! spans and always leave the caller a way to recover the source text. The
//! first engine is deliberately deterministic; a model-backed implementation
//! can replace [`transform`] only through the same contract and gates.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

/// Versioned identifier for a built-in or user-derived style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StyleId {
    Pirate,
    Noir,
}

impl StyleId {
    pub const fn version(self) -> u32 {
        1
    }
}

impl fmt::Display for StyleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Pirate => "pirate",
            Self::Noir => "noir",
        };
        f.write_str(value)
    }
}

/// The strength of a requested transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Intensity {
    Low,
    Normal,
    High,
}

/// Whether output can be emitted directly or only shown for approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interaction {
    Selected,
    PreviewOnly,
}

/// A complete, auditable parody request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParodyRequest {
    pub source_text: String,
    pub style: StyleId,
    pub intensity: Intensity,
    pub interaction: Interaction,
}

impl ParodyRequest {
    pub fn new(
        source_text: impl Into<String>,
        style: StyleId,
        intensity: Intensity,
        interaction: Interaction,
    ) -> Self {
        Self {
            source_text: source_text.into(),
            style,
            intensity,
            interaction,
        }
    }
}

/// A protected substring in source-order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtectedSpan {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub kind: ProtectedKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtectedKind {
    Number,
    Identifier,
    Url,
}

/// The output of a successful transform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParodyResult {
    pub source_text: String,
    pub text: String,
    pub style: StyleId,
    pub style_version: u32,
    pub intensity: Intensity,
    pub interaction: Interaction,
    pub protected_spans: Vec<ProtectedSpan>,
}

/// Why a candidate was refused before any output was produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParodyRefusal {
    EmptySource,
    ProtectedSpanOverlap { text: String },
    UnsupportedStyle,
}

/// Transform a request locally, or refuse without returning candidate text.
pub fn transform(request: &ParodyRequest) -> Result<ParodyResult, ParodyRefusal> {
    if request.source_text.trim().is_empty() {
        return Err(ParodyRefusal::EmptySource);
    }

    let protected = protected_spans(&request.source_text);
    let mut out = String::with_capacity(request.source_text.len() + 48);
    let mut source_at = 0;
    for span in &protected {
        let before = &request.source_text[source_at..span.start];
        out.push_str(&replace_plain(before, request.style, request.intensity));
        out.push_str(&span.text);
        source_at = span.end;
    }
    out.push_str(&replace_plain(
        &request.source_text[source_at..],
        request.style,
        request.intensity,
    ));
    if request.intensity == Intensity::High {
        out = frame_style(out, request.style);
    }
    if !protected_spans_preserved(&protected, &out) {
        return Err(ParodyRefusal::ProtectedSpanOverlap {
            text: request.source_text.clone(),
        });
    }
    Ok(ParodyResult {
        source_text: request.source_text.clone(),
        text: out,
        style: request.style,
        style_version: request.style.version(),
        intensity: request.intensity,
        interaction: request.interaction,
        protected_spans: protected,
    })
}

/// Verify ordered semantic identity and exact occurrence counts.
fn protected_spans_preserved(protected: &[ProtectedSpan], output: &str) -> bool {
    let semantic = protected_spans(output)
        .into_iter()
        .map(|span| (span.text, span.kind))
        .collect::<Vec<_>>();
    let expected = protected
        .iter()
        .map(|span| (span.text.clone(), span.kind))
        .collect::<Vec<_>>();
    if semantic != expected {
        return false;
    }
    let mut output_counts = std::collections::HashMap::new();
    for span in protected_spans(output) {
        *output_counts.entry(span.text).or_insert(0_usize) += 1;
    }
    let mut source_counts = std::collections::HashMap::new();
    for span in protected {
        *source_counts.entry(span.text.clone()).or_insert(0_usize) += 1;
    }
    output_counts == source_counts
}

/// Extract spans that must survive byte-for-byte.
///
/// URLs are extracted before identifiers so `https://a.b/c` is not split apart.
/// Identifiers are intentionally conservative: words with `_`, internal dots,
/// hyphens, `::`, or mixed alphabetic and digit characters.
pub fn protected_spans(text: &str) -> Vec<ProtectedSpan> {
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if text[i..].starts_with("http://") || text[i..].starts_with("https://") {
            let start = i;
            i = text[start..]
                .find(char::is_whitespace)
                .map(|offset| start + offset)
                .unwrap_or(text.len());
            spans.push(ProtectedSpan {
                start,
                end: i,
                text: text[start..i].to_owned(),
                kind: ProtectedKind::Url,
            });
            continue;
        }
        if bytes[i].is_ascii_alphanumeric() {
            let start = i;
            let mut last_word_byte = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric()
                    || matches!(bytes[i], b'_' | b'.' | b'-' | b':'))
            {
                if bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' {
                    last_word_byte = i;
                }
                i += 1;
            }
            i = last_word_byte + 1;
            let word = &text[start..i];
            let identifier = word.bytes().any(|b| b == b'_')
                || word.bytes().any(|b| b.is_ascii_digit())
                    && word.bytes().any(|b| b.is_ascii_alphabetic());
            let number = word.bytes().all(|b| b.is_ascii_digit());
            let kind = if identifier {
                Some(ProtectedKind::Identifier)
            } else if number {
                Some(ProtectedKind::Number)
            } else {
                None
            };
            if let Some(kind) = kind {
                spans.push(ProtectedSpan {
                    start,
                    end: i,
                    text: word.to_owned(),
                    kind,
                });
            }
            continue;
        }
        i += 1;
    }
    spans
}

fn replace_plain(source: &str, style: StyleId, intensity: Intensity) -> String {
    match style {
        StyleId::Pirate => pirate_plain(source, intensity),
        StyleId::Noir => noir_plain(source, intensity),
    }
}

fn pirate_plain(source: &str, intensity: Intensity) -> String {
    let replacements: &[(&str, &str)] = match intensity {
        Intensity::Low => &[("hello", "ahoy")],
        Intensity::Normal | Intensity::High => &[
            ("hello", "ahoy"),
            ("my", "me"),
            ("yes", "aye"),
            ("friend", "matey"),
        ],
    };
    replace_words(source, replacements)
}

fn noir_plain(source: &str, intensity: Intensity) -> String {
    let replacements: &[(&str, &str)] = match intensity {
        Intensity::Low => &[("night", "neon night")],
        Intensity::Normal | Intensity::High => &[
            ("the", "the rain-slick"),
            ("night", "neon night"),
            ("city", "city of secrets"),
        ],
    };
    replace_words(source, replacements)
}

fn frame_style(source: String, style: StyleId) -> String {
    match style {
        StyleId::Pirate => format!("Arrr. {source} Yarr!"),
        StyleId::Noir => format!("The fog remembered this: {source} The case stayed open."),
    }
}

fn replace_words(source: &str, replacements: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(source.len() + 32);
    let mut chars = source.char_indices().peekable();
    while let Some((start, first)) = chars.next() {
        if !first.is_alphabetic() {
            out.push(first);
            continue;
        }
        let mut end = source.len();
        while let Some(&(index, ch)) = chars.peek() {
            if ch.is_alphabetic() {
                chars.next();
            } else {
                end = index;
                break;
            }
        }
        let word = &source[start..end];
        let key = word.to_lowercase();
        if let Some((_, replacement)) = replacements.iter().find(|(from, _)| *from == key) {
            if word.chars().next().is_some_and(char::is_uppercase) {
                out.push_str(&capitalize_first(replacement));
            } else {
                out.push_str(replacement);
            }
        } else {
            out.push_str(word);
        }
    }
    out
}

fn capitalize_first(value: &str) -> String {
    let mut chars = value.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => value.to_owned(),
    }
}

/// Unique protected span texts, useful in compact event records.
pub fn protected_vocabulary(spans: &[ProtectedSpan]) -> BTreeSet<&str> {
    spans.iter().map(|span| span.text.as_str()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pirate_transform_preserves_numbers_identifiers_and_urls() {
        let request = ParodyRequest::new(
            "hello my friend, deploy phonon_v2 at 09:30 using https://example.test/a_b",
            StyleId::Pirate,
            Intensity::Normal,
            Interaction::PreviewOnly,
        );
        let result = transform(&request).unwrap();
        assert_eq!(
            result.text,
            "ahoy me matey, deploy phonon_v2 at 09:30 using https://example.test/a_b"
        );
        assert!(result.source_text.starts_with("hello my friend"));
    }

    #[test]
    fn noir_transform_only_rewrites_plain_language() {
        let request = ParodyRequest::new(
            "the night city meeting kept fix_42 and 7 candles",
            StyleId::Noir,
            Intensity::Normal,
            Interaction::Selected,
        );
        let result = transform(&request).unwrap();
        assert_eq!(
            result.text,
            "the rain-slick neon night city of secrets meeting kept fix_42 and 7 candles"
        );
        assert_eq!(result.interaction, Interaction::Selected);
    }

    #[test]
    fn refuses_empty_source_without_candidate_text() {
        let request = ParodyRequest::new(
            "  ",
            StyleId::Pirate,
            Intensity::Low,
            Interaction::PreviewOnly,
        );
        assert_eq!(transform(&request), Err(ParodyRefusal::EmptySource));
    }

    #[test]
    fn parody_preserves_negation_order_and_source() {
        let source = "No, deploy first_v2 before second_v3. Do not reverse 1 and 2.";
        let pirate = transform(&ParodyRequest::new(
            source,
            StyleId::Pirate,
            Intensity::High,
            Interaction::PreviewOnly,
        ))
        .unwrap();
        assert!(pirate.text.contains("No,"));
        assert!(pirate.text.contains("Do not"));
        assert!(pirate.text.find("first_v2").unwrap() < pirate.text.find("second_v3").unwrap());
        assert_eq!(pirate.source_text, source);

        let noir = transform(&ParodyRequest::new(
            source,
            StyleId::Noir,
            Intensity::High,
            Interaction::Selected,
        ))
        .unwrap();
        assert!(noir.text.contains("No,"));
        assert!(noir.text.contains("Do not"));
        assert!(noir.text.find("first_v2").unwrap() < noir.text.find("second_v3").unwrap());
        assert_eq!(noir.source_text, source);
    }

    #[test]
    fn word_replacement_respects_boundaries_and_capitalization() {
        let request = ParodyRequest::new(
            "Hello my friend! Yes, no and never.",
            StyleId::Pirate,
            Intensity::Normal,
            Interaction::PreviewOnly,
        );
        assert_eq!(
            transform(&request).unwrap().text,
            "Ahoy me matey! Aye, no and never."
        );
    }

    #[test]
    fn intensity_changes_output_without_touching_protected_spans() {
        let source = "hello my friend, version_v2 stayed at 10:30";
        let low = transform(&ParodyRequest::new(
            source,
            StyleId::Pirate,
            Intensity::Low,
            Interaction::PreviewOnly,
        ))
        .unwrap();
        let high = transform(&ParodyRequest::new(
            source,
            StyleId::Pirate,
            Intensity::High,
            Interaction::PreviewOnly,
        ))
        .unwrap();
        assert_eq!(low.text, "ahoy my friend, version_v2 stayed at 10:30");
        assert_eq!(
            high.text,
            "Arrr. ahoy me matey, version_v2 stayed at 10:30 Yarr!"
        );
        assert_eq!(low.protected_spans, high.protected_spans);
    }

    #[test]
    fn adjacent_protected_values_do_not_merge() {
        let spans = protected_spans("first_v2 second_v3 10 20");
        let values: Vec<&str> = spans.iter().map(|span| span.text.as_str()).collect();
        assert_eq!(values, ["first_v2", "second_v3", "10", "20"]);
    }

    #[test]
    fn protected_urls_win_over_identifiers() {
        let spans = protected_spans("visit https://example.test/a_b now");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].kind, ProtectedKind::Url);
        assert_eq!(spans[0].text, "https://example.test/a_b");
    }
}
