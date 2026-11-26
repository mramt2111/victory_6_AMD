//! Every `VICTORY_*` switch the documents name is read by the code (#49).
//!
//! A switch is read through a string literal of its whole name
//! (`std::env::var("VICTORY_X")`, `crate::switch::on("VICTORY_X")`,
//! `crate::ab::pick("VICTORY_X")`), so a name the Markdown files give must
//! appear quoted in at least one Rust file. xtask's own files are left out
//! of the search: this module's tests name switches of their own.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

use anyhow::Result;

/// List tracked and untracked-but-not-ignored files in `root`, falling back to
/// a directory walk (skipping `target` and `.git`) outside a git repository.
fn list_files(root: &Path) -> Result<Vec<String>> {
    if let Ok(out) = Command::new("git")
        .args(["ls-files", "-z", "--cached", "--others", "--exclude-standard"])
        .current_dir(root)
        .output()
        && out.status.success()
    {
        let mut tracked: Vec<String> = out
            .stdout
            .split(|&b| b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).replace('\\', "/"))
            .collect();
        tracked.sort();
        tracked.dedup();
        return Ok(tracked);
    }
    Ok(walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !matches!(e.file_name().to_str(), Some("target" | ".git")))
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| {
            e.path()
                .strip_prefix(root)
                .ok()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
        })
        .collect())
}

/// The `VICTORY_*` names in `text`, without trailing underscores (a prefix
/// such as `VICTORY_` alone or `VICTORY_...` is not a name).
fn names(text: &str) -> BTreeSet<String> {
    const PREFIX: &str = "VICTORY_";
    let mut found = BTreeSet::new();
    let mut rest = text;
    while let Some(at) = rest.find(PREFIX) {
        let tail = &rest[at + PREFIX.len()..];
        let len = tail
            .bytes()
            .take_while(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || *b == b'_')
            .count();
        let suffix = tail[..len].trim_end_matches('_');
        if !suffix.is_empty() {
            found.insert(format!("{PREFIX}{suffix}"));
        }
        rest = &tail[len..];
    }
    found
}

/// Retired diagnostic switches still mentioned in archived historical docs
/// (such as `docs/README-fxceless.md`) that are intentionally no longer read
/// by Rust source.
const RETIRED_DOC_SWITCHES: &[&str] = &["VICTORY_MERGE_RACE"];

/// The names of `documented` that no text of `sources` holds in quotes.
fn unread<'a>(documented: &'a BTreeSet<String>, sources: &[String]) -> Vec<&'a str> {
    documented
        .iter()
        .filter(|name| {
            let quoted = format!("\"{name}\"");
            !sources.iter().any(|s| s.contains(&quoted))
        })
        .map(String::as_str)
        .collect()
}

#[test]
fn documented_switches_are_read() {
    let root = crate::workspace_root();
    let files = list_files(&root).unwrap();
    let read = |path: &String| std::fs::read_to_string(root.join(path)).ok();
    let mut documented = BTreeSet::new();
    let mut doc_of = Vec::new();
    for path in files.iter().filter(|p| p.ends_with(".md")) {
        for name in names(&read(path).unwrap_or_default()) {
            doc_of.push((name.clone(), path.clone()));
            documented.insert(name);
        }
    }
    let sources: Vec<String> = files
        .iter()
        .filter(|p| p.ends_with(".rs") && !p.starts_with("xtask/"))
        .filter_map(read)
        .collect();
    assert!(
        !documented.is_empty(),
        "no VICTORY_* names found in the Markdown files"
    );
    let dead: Vec<&str> = unread(&documented, &sources)
        .into_iter()
        .filter(|name| !RETIRED_DOC_SWITCHES.contains(name))
        .collect();
    let report: Vec<String> = dead
        .iter()
        .map(|name| {
            let docs: Vec<&str> = doc_of
                .iter()
                .filter(|(n, _)| n == name)
                .map(|(_, p)| p.as_str())
                .collect();
            format!("{name} (named in {})", docs.join(", "))
        })
        .collect();
    assert!(
        dead.is_empty(),
        "switches the documents name but no Rust source reads: {}",
        report.join("; ")
    );
}

#[test]
fn finds_whole_names_only() {
    let text =
        "`VICTORY_LATE_WAIT=3`, VICTORY_LATE_WAITS, `VICTORY_...`, VICTORY_ alone, VICTORY_X_ end";
    let found: Vec<String> = names(text).into_iter().collect();
    assert_eq!(
        found,
        ["VICTORY_LATE_WAIT", "VICTORY_LATE_WAITS", "VICTORY_X"]
    );
}

#[test]
fn a_longer_name_does_not_read_a_shorter_one() {
    let documented = names("VICTORY_LATE_WAIT and VICTORY_WIDE_BACK");
    let sources = [
        "std::env::var_os(\"VICTORY_LATE_WAITS\")".to_owned(),
        "on(\"VICTORY_WIDE_BACK\")".to_owned(),
    ];
    assert_eq!(unread(&documented, &sources), ["VICTORY_LATE_WAIT"]);
}
