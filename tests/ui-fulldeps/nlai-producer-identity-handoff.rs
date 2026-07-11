//@ run-pass
//@ edition: 2021
//@ only-macos
//@ ignore-cross-compile
//@ ignore-stage1 (requires matching rustc-private artifacts in the tested sysroot)
//@ needs-dynamic-linking
//@ needs-git-hash
//@ run-flags: {{sysroot-base}} {{target}}

use std::ffi::OsStr;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

const BACKEND_SOURCE: &str = include_str!("codegen-backend/auxiliary/nlai_handoff_backend.rs");
const SOURCE: &str = "#![feature(no_core)]\n#![no_core]\n";
const REPLACEMENT: &[u8] = b"not the mapped rustc image\n";
const RUSTC_PRIVATE_CRATES: [&str; 5] = [
    "rustc_codegen_ssa",
    "rustc_data_structures",
    "rustc_metadata",
    "rustc_middle",
    "rustc_session",
];

#[derive(Debug, PartialEq, Eq)]
struct Observation {
    identity: [String; 3],
    current_path_len: u64,
}

fn main() {
    let sysroot = PathBuf::from(std::env::args_os().nth(1).expect("missing stage sysroot"));
    let target = std::env::args_os().nth(2).expect("missing host target");
    let host_build = sysroot
        .parent()
        .expect("stage sysroot must be inside the host build directory");
    let rustc = sysroot.join("bin").join(format!("rustc{}", std::env::consts::EXE_SUFFIX));
    assert!(rustc.is_file(), "missing tested rustc: {}", rustc.display());
    let (rustc_private_metadata, rustc_runtime_libdir) =
        query_rustc_library_layout(&rustc, &sysroot, &target);
    let (rustc_private_deps, rustc_private_rlibs) =
        resolve_rustc_private_rlibs(&rustc_private_metadata, host_build, Path::new(&target));
    let test_dir =
        std::env::current_dir().unwrap().join(format!("nlai-handoff-{}", std::process::id()));
    if test_dir.exists() {
        std::fs::remove_dir_all(&test_dir).unwrap();
    }
    std::fs::create_dir(&test_dir).unwrap();

    let backend_source = test_dir.join("backend.rs");
    std::fs::write(&backend_source, BACKEND_SOURCE).unwrap();
    let backend = test_dir.join("libnlai_handoff_backend.dylib");
    let mut backend_build = Command::new(&rustc);
    backend_build
        .arg(&backend_source)
        .arg("--crate-type=dylib")
        .arg("-Crpath")
        .arg("--sysroot")
        .arg(&sysroot)
        .arg("-L")
        .arg(format!("dependency={}", rustc_private_deps.display()));
    // Use the exact rlibs matching the tested sysroot's rmeta files. Omitting `rustc_driver`
    // deliberately creates a second rustc_codegen_ssa instance inside the backend dylib.
    for (crate_name, rlib) in rustc_private_rlibs {
        backend_build.arg("--extern").arg(format!("{crate_name}={}", rlib.display()));
    }
    let backend_build = backend_build
        .arg("-o")
        .arg(&backend)
        .env("NLAI", "0")
        .output()
        .expect("compile dynamic handoff backend");
    require_success("compile dynamic handoff backend", &backend_build);
    let dependencies = Command::new("otool")
        .arg("-L")
        .arg(&backend)
        .output()
        .expect("inspect backend dependencies");
    require_success("inspect backend dependencies", &dependencies);
    assert!(
        !String::from_utf8_lossy(&dependencies.stdout).contains("librustc_driver"),
        "test backend must carry its own rustc_codegen_ssa copy"
    );
    let symbols = Command::new("nm")
        .arg(&backend)
        .output()
        .expect("inspect backend symbols");
    require_success("inspect backend symbols", &symbols);
    assert!(
        String::from_utf8_lossy(&symbols.stdout).contains("initialize_nlai_producer_identity"),
        "test backend must contain a backend-local producer-identity initializer"
    );

    let input = test_dir.join("input.rs");
    std::fs::write(&input, SOURCE).unwrap();
    let launcher = test_dir.join("rustc-under-test");
    std::fs::copy(&rustc, &launcher).unwrap();
    let launcher_len = std::fs::metadata(&launcher).unwrap().len();

    let baseline_dir = test_dir.join("baseline");
    std::fs::create_dir(&baseline_dir).unwrap();
    let baseline_marker = baseline_dir.join("backend-observation");
    let baseline =
        configured_rustc(
            &launcher,
            &backend,
            &sysroot,
            &rustc_runtime_libdir,
            &input,
            &baseline_dir,
            &baseline_marker,
        )
        .output()
        .expect("run baseline copied rustc");
    require_success("baseline copied rustc", &baseline);
    let baseline = read_observation(&baseline_marker);
    assert_eq!(baseline.current_path_len, launcher_len);

    // Logger initialization is environment-driven and may open/truncate an arbitrary path. Point
    // it at a distinct copy of the launcher itself: the driver must have captured the original
    // image identity before the logger truncates that vnode.
    let logger_launcher = test_dir.join("rustc-logger-target");
    std::fs::copy(&rustc, &logger_launcher).unwrap();
    let logger_dir = test_dir.join("logger-target");
    std::fs::create_dir(&logger_dir).unwrap();
    let logger_marker = logger_dir.join("backend-observation");
    let logger = configured_rustc(
        &logger_launcher,
        &backend,
        &sysroot,
        &rustc_runtime_libdir,
        &input,
        &logger_dir,
        &logger_marker,
    )
    .env("RUSTC_LOG_OUTPUT_TARGET", &logger_launcher)
    .output()
    .expect("run copied rustc with its launcher as logger target");
    require_success("copied rustc with its launcher as logger target", &logger);
    let logger = read_observation(&logger_marker);
    assert_eq!(logger.identity, baseline.identity);
    assert_eq!(logger.current_path_len, 0);
    assert_eq!(std::fs::metadata(&logger_launcher).unwrap().len(), 0);

    let fifo = test_dir.join("blocked-input.rs");
    let mkfifo = Command::new("mkfifo").arg(&fifo).output().expect("create source FIFO");
    require_success("create source FIFO", &mkfifo);
    let raced_dir = test_dir.join("raced");
    std::fs::create_dir(&raced_dir).unwrap();
    let raced_marker = raced_dir.join("backend-observation");
    let mut raced =
        configured_rustc(
            &launcher,
            &backend,
            &sysroot,
            &rustc_runtime_libdir,
            &fifo,
            &raced_dir,
            &raced_marker,
        )
        .spawn()
        .expect("spawn FIFO-gated copied rustc");

    // The writer open completes only after rustc has opened the FIFO for reading. Driver startup
    // and its one executable measurement are then complete, while parsing still waits for bytes.
    let mut writer = wait_for_fifo_reader(&mut raced, &fifo);
    let replacement = test_dir.join("replacement-image");
    std::fs::write(&replacement, REPLACEMENT).unwrap();
    std::fs::rename(&replacement, &launcher).unwrap();
    writer.write_all(SOURCE.as_bytes()).expect("feed source after launcher replacement");
    drop(writer);

    let raced = raced.wait_with_output().expect("wait for FIFO-gated copied rustc");
    require_success("FIFO-gated copied rustc", &raced);
    let raced = read_observation(&raced_marker);

    assert_eq!(
        raced.identity, baseline.identity,
        "backend must consume the early session snapshot"
    );
    assert_eq!(raced.current_path_len, REPLACEMENT.len() as u64);
    assert_ne!(raced.current_path_len, launcher_len);

    std::fs::remove_dir_all(test_dir).unwrap();
}

fn query_rustc_library_layout(rustc: &Path, sysroot: &Path, target: &OsStr) -> (PathBuf, PathBuf) {
    let output = Command::new(rustc)
        .arg("--print")
        .arg("target-libdir")
        .arg("--target")
        .arg(target)
        .arg("--sysroot")
        .arg(sysroot)
        .env("NLAI", "0")
        .output()
        .expect("query tested rustc target library directory");
    require_success("query tested rustc target library directory", &output);
    assert!(
        output.stderr.is_empty(),
        "target-libdir query emitted unexpected stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let printed = String::from_utf8(output.stdout)
        .expect("tested rustc target library directory must be valid UTF-8");
    let printed = printed
        .strip_suffix('\n')
        .expect("target-libdir query must end with one newline");
    let printed = printed.strip_suffix('\r').unwrap_or(printed);
    assert!(
        !printed.is_empty() && !printed.contains('\n') && !printed.contains('\r'),
        "target-libdir query must contain exactly one nonempty path"
    );
    let target_libdir = PathBuf::from(printed);
    assert!(target_libdir.is_absolute(), "target-libdir must be absolute: {target_libdir:?}");
    assert!(target_libdir.is_dir(), "target-libdir does not exist: {target_libdir:?}");

    let target_dir = target_libdir.parent().expect("target-libdir must have a target parent");
    assert_eq!(target_dir.file_name(), Some(target));
    let rustlib_dir = target_dir.parent().expect("target directory must be inside rustlib");
    assert_eq!(rustlib_dir.file_name(), Some(OsStr::new("rustlib")));
    let runtime_libdir = rustlib_dir
        .parent()
        .expect("rustlib must be inside the runtime libdir")
        .to_owned();
    assert!(runtime_libdir.is_dir());
    (target_libdir, runtime_libdir)
}

fn resolve_rustc_private_rlibs(
    metadata_dir: &Path,
    host_build: &Path,
    target: &Path,
) -> (PathBuf, Vec<(&'static str, PathBuf)>) {
    let rlib_names = RUSTC_PRIVATE_CRATES.map(|crate_name| {
        let prefix = format!("lib{crate_name}-");
        let mut matches = std::fs::read_dir(metadata_dir)
            .unwrap_or_else(|error| panic!("read tested-sysroot metadata directory: {error}"))
            .map(|entry| {
                entry
                    .unwrap_or_else(|error| panic!("read tested-sysroot metadata entry: {error}"))
                    .file_name()
            })
            .filter(|name| {
                let name = name.to_string_lossy();
                name.starts_with(&prefix) && name.ends_with(".rmeta")
            })
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1, "expected one tested-sysroot rmeta for {crate_name}");
        let mut rlib_name = PathBuf::from(matches.pop().unwrap());
        rlib_name.set_extension("rlib");
        (crate_name, rlib_name)
    });

    // An uplifted stage-N sysroot can contain a compiler built in stage N-1. Resolve the build
    // directory by the exact rustc-private crate hashes installed in the tested sysroot instead of
    // assuming that the sysroot and artifact-stage numbers (or Cargo profiles) match.
    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(host_build)
        .unwrap_or_else(|error| panic!("read host build directory: {error}"))
    {
        let entry = entry.unwrap_or_else(|error| panic!("read host build entry: {error}"));
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("stage") || !name.ends_with("-rustc") {
            continue;
        }
        for profile in ["release", "debug"] {
            let deps = entry.path().join(target).join(profile).join("deps");
            if deps.is_dir() && rlib_names.iter().all(|(_, name)| deps.join(name).is_file()) {
                candidates.push(deps);
            }
        }
    }
    candidates.sort();
    candidates.dedup();
    assert_eq!(
        candidates.len(),
        1,
        "expected one rustc artifact directory matching the tested-sysroot metadata; candidates: \
         {candidates:?}"
    );
    let dependency_dir = candidates.pop().unwrap();
    let rlibs = rlib_names
        .into_iter()
        .map(|(crate_name, rlib_name)| (crate_name, dependency_dir.join(rlib_name)))
        .collect();
    (dependency_dir, rlibs)
}

fn configured_rustc(
    launcher: &Path,
    backend: &Path,
    sysroot: &Path,
    rustc_runtime_libdir: &Path,
    input: &Path,
    output_dir: &Path,
    marker: &Path,
) -> Command {
    let mut dylib_paths = vec![rustc_runtime_libdir.to_owned()];
    if let Some(existing) = std::env::var_os("DYLD_LIBRARY_PATH") {
        dylib_paths.extend(std::env::split_paths(&existing));
    }

    let mut command = Command::new(launcher);
    command
        .arg(input)
        .arg("--crate-type=rlib")
        .arg("--edition=2021")
        .arg("--sysroot")
        .arg(sysroot)
        .arg("--out-dir")
        .arg(output_dir)
        .arg(format!("-Zcodegen-backend={}", backend.display()))
        .env("DYLD_LIBRARY_PATH", std::env::join_paths(dylib_paths).unwrap())
        .env("NLAI", "1")
        .env("NLAI_OUTPUT_DIR", output_dir)
        .env("NLAI_HANDOFF_MARKER", marker)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

fn wait_for_fifo_reader(child: &mut Child, fifo: &Path) -> std::fs::File {
    let (sender, receiver) = mpsc::sync_channel(1);
    let fifo = fifo.to_owned();
    thread::spawn(move || {
        let result = OpenOptions::new().write(true).open(fifo);
        let _ = sender.send(result);
    });

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match receiver.recv_timeout(Duration::from_millis(10)) {
            Ok(Ok(writer)) => return writer,
            Ok(Err(error)) => panic!("open FIFO writer after rustc reader: {error}"),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => panic!("FIFO writer thread disconnected"),
        }
        if let Some(status) = child.try_wait().expect("poll FIFO-gated rustc") {
            panic!("FIFO-gated rustc exited before opening its source FIFO: {status}");
        }
        if Instant::now() >= deadline {
            child.kill().expect("kill rustc after FIFO-open timeout");
            panic!("rustc did not open its source FIFO within 30 seconds");
        }
    }
}

fn require_success(context: &str, output: &Output) {
    assert!(
        output.status.success(),
        "{context} failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn read_observation(path: &Path) -> Observation {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let identity = [
        take_field(&mut lines, "commit"),
        take_field(&mut lines, "source"),
        take_field(&mut lines, "executable"),
    ];
    let current_path_len = take_field(&mut lines, "current_path_len").parse().unwrap();
    assert_eq!(lines.next(), None, "unexpected trailing backend observation data");
    Observation { identity, current_path_len }
}

fn take_field<'a>(lines: &mut impl Iterator<Item = &'a str>, name: &str) -> String {
    lines
        .next()
        .unwrap_or_else(|| panic!("missing {name} observation"))
        .strip_prefix(&format!("{name}="))
        .unwrap_or_else(|| panic!("malformed {name} observation"))
        .to_owned()
}
