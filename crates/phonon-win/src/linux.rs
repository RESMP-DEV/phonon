//! Linux child-process setup. The downloaded runtimes carry their own libraries.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};

/// Set the loader path only on the child. sherpa uses `lib/`; llama.cpp keeps
/// its libraries beside the executable. Preserve any caller-supplied paths.
pub fn runtime_libraries(command: &mut Command, root: &Path) -> Result<()> {
    let mut paths = vec![root.join("lib"), root.to_path_buf()];
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    command.env(
        "LD_LIBRARY_PATH",
        std::env::join_paths(paths).context("construct the runtime library path")?,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loader_paths_are_set_on_the_child_in_priority_order() {
        let mut command = Command::new("unused");
        runtime_libraries(&mut command, Path::new("/tmp/runtime with spaces")).unwrap();
        let value = command
            .get_envs()
            .find(|(key, _)| *key == "LD_LIBRARY_PATH")
            .unwrap()
            .1
            .unwrap();
        let paths: Vec<_> = std::env::split_paths(value).collect();
        assert_eq!(paths[0], Path::new("/tmp/runtime with spaces/lib"));
        assert_eq!(paths[1], Path::new("/tmp/runtime with spaces"));
    }
}
