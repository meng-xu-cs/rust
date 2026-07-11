use std::fmt;
use std::fs::{self, File, Metadata};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use rustc_middle::bug;

use super::{Activation, activation_from_environment};

/// Validated identity of the rustc image that started this process.
///
/// Construction is deliberately confined to this module: an embedding can carry or inspect a
/// genuine startup snapshot, but cannot manufacture one from caller-controlled strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NlaiProducerIdentity {
    commit: String,
    source_state_fingerprint: String,
    executable_fingerprint: String,
}

impl NlaiProducerIdentity {
    pub fn commit(&self) -> &str {
        &self.commit
    }

    pub fn source_state_fingerprint(&self) -> &str {
        &self.source_state_fingerprint
    }

    pub fn executable_fingerprint(&self) -> &str {
        &self.executable_fingerprint
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        commit: impl Into<String>,
        source_state_fingerprint: impl Into<String>,
        executable_fingerprint: impl Into<String>,
    ) -> Self {
        Self {
            commit: commit.into(),
            source_state_fingerprint: source_state_fingerprint.into(),
            executable_fingerprint: executable_fingerprint.into(),
        }
    }
}

/// BLAKE3 derive-key context for fingerprints of the exact rustc executable bytes.
pub(crate) const RUSTC_EXECUTABLE_FINGERPRINT_CONTEXT: &str = "nlai.rustc-executable.blake3.v1";

const FINGERPRINT_LENGTH: usize = 64;
const READ_BUFFER_SIZE: usize = 64 * 1024;

#[derive(Debug)]
enum ProducerIdentityError {
    CompiledIdentity { field: &'static str, cause: String },
    CurrentExecutable(io::Error),
    ExecutableIo { path: PathBuf, operation: &'static str, source: io::Error },
    ExecutableNotRegular { path: PathBuf, observation: &'static str },
    ExecutableChanged { path: PathBuf, cause: &'static str },
    ExecutableLengthMismatch { path: PathBuf, expected: u64, read: u64 },
}

impl fmt::Display for ProducerIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CompiledIdentity { field, cause } => {
                write!(f, "invalid compiled {field}: {cause}")
            }
            Self::CurrentExecutable(error) => {
                write!(f, "unable to discover the exact running rustc executable: {error}")
            }
            Self::ExecutableIo { path, operation, source } => {
                write!(f, "unable to {operation} running rustc executable {path:?}: {source}")
            }
            Self::ExecutableNotRegular { path, observation } => write!(
                f,
                "running rustc executable {path:?} was not a regular file at {observation}"
            ),
            Self::ExecutableChanged { path, cause } => write!(
                f,
                "running rustc executable {path:?} changed while its identity was measured: {cause}"
            ),
            Self::ExecutableLengthMismatch { path, expected, read } => write!(
                f,
                "running rustc executable {path:?} reported {expected} bytes but yielded {read} bytes while hashing"
            ),
        }
    }
}

impl std::error::Error for ProducerIdentityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CurrentExecutable(error) | Self::ExecutableIo { source: error, .. } => {
                Some(error)
            }
            Self::CompiledIdentity { .. }
            | Self::ExecutableNotRegular { .. }
            | Self::ExecutableChanged { .. }
            | Self::ExecutableLengthMismatch { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExecutableFileStamp {
    len: u64,
    platform: PlatformFileStamp,
}

#[cfg(unix)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct PlatformFileStamp {
    device: u64,
    inode: u64,
    mode: u32,
    links: u64,
    owner: u32,
    group: u32,
    special_device: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[cfg(windows)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct PlatformFileStamp {
    attributes: u32,
    creation_time: u64,
    last_write_time: u64,
}

#[cfg(not(any(unix, windows)))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct PlatformFileStamp;

trait ExecutableProbe {
    type Reader;

    fn current_exe(&mut self) -> io::Result<PathBuf>;
    fn open_loaded(&mut self, path: &Path) -> io::Result<Self::Reader>;
    fn loaded_stamp(&mut self, reader: &Self::Reader) -> io::Result<Option<ExecutableFileStamp>>;
    fn opened_stamp(&mut self, reader: &Self::Reader) -> io::Result<Option<ExecutableFileStamp>>;
    fn path_stamp(&mut self, path: &Path) -> io::Result<Option<ExecutableFileStamp>>;
    fn read(&mut self, reader: &mut Self::Reader, buffer: &mut [u8]) -> io::Result<usize>;
}

struct FileSystemProbe;

impl ExecutableProbe for FileSystemProbe {
    type Reader = File;

    fn current_exe(&mut self) -> io::Result<PathBuf> {
        std::env::current_exe()
    }

    fn open_loaded(&mut self, path: &Path) -> io::Result<Self::Reader> {
        loaded_executable_file(path)
    }

    fn loaded_stamp(&mut self, _reader: &Self::Reader) -> io::Result<Option<ExecutableFileStamp>> {
        #[cfg(target_vendor = "apple")]
        {
            apple_loaded_executable_stamp()
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            executable_stamp(_reader.metadata()?)
        }
    }

    fn opened_stamp(&mut self, reader: &Self::Reader) -> io::Result<Option<ExecutableFileStamp>> {
        executable_stamp(reader.metadata()?)
    }

    fn path_stamp(&mut self, path: &Path) -> io::Result<Option<ExecutableFileStamp>> {
        executable_stamp(fs::metadata(path)?)
    }

    fn read(&mut self, reader: &mut Self::Reader, buffer: &mut [u8]) -> io::Result<usize> {
        reader.read(buffer)
    }
}

fn executable_stamp(metadata: Metadata) -> io::Result<Option<ExecutableFileStamp>> {
    if !metadata.file_type().is_file() {
        return Ok(None);
    }
    #[cfg(unix)]
    let platform = {
        use std::os::unix::fs::MetadataExt;

        PlatformFileStamp {
            device: metadata.dev(),
            inode: metadata.ino(),
            mode: metadata.mode(),
            links: metadata.nlink(),
            owner: metadata.uid(),
            group: metadata.gid(),
            special_device: metadata.rdev(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    };

    #[cfg(windows)]
    let platform = {
        use std::os::windows::fs::MetadataExt;

        PlatformFileStamp {
            attributes: metadata.file_attributes(),
            creation_time: metadata.creation_time(),
            last_write_time: metadata.last_write_time(),
        }
    };

    #[cfg(not(any(unix, windows)))]
    return Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "the host platform does not expose stable executable file identity",
    ));

    #[cfg(any(unix, windows))]
    Ok(Some(ExecutableFileStamp { len: metadata.len(), platform }))
}

/// Open a kernel path for the process image where the host exposes one. Unlike the string returned
/// by `current_exe`, these procfs links continue to denote the loaded vnode after a rename. Apple
/// hosts have no equivalent openable handle, so their opened path is checked against the vnode
/// backing the in-memory Mach-O header by `apple_loaded_executable_stamp` before any byte is trusted.
fn loaded_executable_file(_current_path: &Path) -> io::Result<File> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let loaded_path = Path::new("/proc/self/exe");
    #[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
    let loaded_path = Path::new("/proc/curproc/file");
    #[cfg(target_os = "netbsd")]
    let loaded_path = Path::new("/proc/curproc/exe");
    #[cfg(any(target_os = "solaris", target_os = "illumos"))]
    let loaded_path = Path::new("/proc/self/path/a.out");

    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "netbsd",
        target_os = "solaris",
        target_os = "illumos"
    ))]
    return File::open(loaded_path);

    // Darwin's mapped-vnode comparison below closes the pathname race.
    #[cfg(target_vendor = "apple")]
    return File::open(_current_path);

    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "netbsd",
        target_os = "solaris",
        target_os = "illumos",
        target_vendor = "apple"
    )))]
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "the host does not expose a trustworthy handle to the loaded executable image",
    ))
}

#[cfg(target_vendor = "apple")]
#[repr(C)]
struct AppleProcRegionInfo {
    protection: u32,
    max_protection: u32,
    inheritance: u32,
    flags: u32,
    offset: u64,
    behavior: u32,
    user_wired_count: u32,
    user_tag: u32,
    pages_resident: u32,
    pages_shared_now_private: u32,
    pages_swapped_out: u32,
    pages_dirtied: u32,
    reference_count: u32,
    shadow_depth: u32,
    share_mode: u32,
    private_pages_resident: u32,
    shared_pages_resident: u32,
    object_id: u32,
    depth: u32,
    address: u64,
    size: u64,
}

#[cfg(target_vendor = "apple")]
#[repr(C)]
struct AppleProcRegionWithPathInfo {
    region: AppleProcRegionInfo,
    vnode: libc::vnode_info_path,
}

#[cfg(target_vendor = "apple")]
fn apple_loaded_executable_stamp() -> io::Result<Option<ExecutableFileStamp>> {
    use std::ffi::CStr;
    use std::mem::{MaybeUninit, size_of};

    const PROC_PIDREGIONPATHINFO: libc::c_int = 8;

    // Resolve from the main image explicitly. A direct extern reference from rustc_codegen_ssa's
    // dylib would be bound in that dylib's two-level namespace instead of the rustc executable.
    // SAFETY: the symbol name is NUL-terminated, and `RTLD_MAIN_ONLY` restricts lookup to the main
    // Mach-O image. `dlerror`'s pointer is consumed immediately when present.
    let execute_header = unsafe {
        libc::dlerror();
        let symbol = libc::dlsym(libc::RTLD_MAIN_ONLY, c"_mh_execute_header".as_ptr());
        if symbol.is_null() {
            let detail = libc::dlerror();
            let detail = if detail.is_null() {
                "dynamic loader returned no diagnostic".to_owned()
            } else {
                CStr::from_ptr(detail).to_string_lossy().into_owned()
            };
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("unable to resolve the main Mach-O header: {detail}"),
            ));
        }
        symbol
    };

    let mut info = MaybeUninit::<AppleProcRegionWithPathInfo>::uninit();
    let buffer_size = libc::c_int::try_from(size_of::<AppleProcRegionWithPathInfo>())
        .expect("Apple process-region structure size fits c_int");
    // SAFETY: `_mh_execute_header` is dyld's symbol for the main executable's mapped Mach-O
    // header. `info` is writable for exactly `buffer_size` bytes, and the result is initialized only
    // after libproc reports that it filled the complete structure.
    let written = unsafe {
        libc::proc_pidinfo(
            libc::getpid(),
            PROC_PIDREGIONPATHINFO,
            execute_header.addr() as u64,
            info.as_mut_ptr().cast(),
            buffer_size,
        )
    };
    if written < 0 {
        return Err(io::Error::last_os_error());
    }
    if written != buffer_size {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "libproc returned {written} of {buffer_size} bytes for the loaded executable region"
            ),
        ));
    }
    // SAFETY: the exact-size success check above establishes full initialization.
    let info = unsafe { info.assume_init() };
    let stat = &info.vnode.vip_vi.vi_stat;
    let mode = u32::from(stat.vst_mode);
    if mode & u32::from(libc::S_IFMT) != u32::from(libc::S_IFREG) {
        return Ok(None);
    }
    let len = u64::try_from(stat.vst_size).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("loaded executable region reported negative file size {}", stat.vst_size),
        )
    })?;
    Ok(Some(ExecutableFileStamp {
        len,
        platform: PlatformFileStamp {
            device: u64::from(stat.vst_dev),
            inode: stat.vst_ino,
            mode,
            links: u64::from(stat.vst_nlink),
            owner: stat.vst_uid,
            group: stat.vst_gid,
            special_device: u64::from(stat.vst_rdev),
            modified_seconds: stat.vst_mtime,
            modified_nanoseconds: stat.vst_mtimensec,
            changed_seconds: stat.vst_ctime,
            changed_nanoseconds: stat.vst_ctimensec,
        },
    }))
}

fn canonical_compiled_identity<'a>(
    field: &'static str,
    value: Option<&'a str>,
    lengths: &[usize],
) -> Result<&'a str, ProducerIdentityError> {
    let value = value.ok_or_else(|| ProducerIdentityError::CompiledIdentity {
        field,
        cause: "the field is unavailable".to_owned(),
    })?;
    if value == "unknown" {
        return Err(ProducerIdentityError::CompiledIdentity {
            field,
            cause: "the field is the explicit unknown sentinel".to_owned(),
        });
    }
    if !lengths.contains(&value.len())
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(ProducerIdentityError::CompiledIdentity {
            field,
            cause: format!(
                "expected lowercase hexadecimal text with byte length in {lengths:?}, got {value:?}"
            ),
        });
    }
    Ok(value)
}

fn executable_fingerprint_with<P: ExecutableProbe>(
    probe: &mut P,
) -> Result<String, ProducerIdentityError> {
    let path = probe.current_exe().map_err(ProducerIdentityError::CurrentExecutable)?;
    let mut reader =
        probe.open_loaded(&path).map_err(|source| ProducerIdentityError::ExecutableIo {
            path: path.clone(),
            operation: "open the loaded image for",
            source,
        })?;
    let loaded_before = probe
        .loaded_stamp(&reader)
        .map_err(|source| ProducerIdentityError::ExecutableIo {
            path: path.clone(),
            operation: "read initial loaded-image metadata for",
            source,
        })?
        .ok_or_else(|| ProducerIdentityError::ExecutableNotRegular {
            path: path.clone(),
            observation: "the initial loaded-image observation",
        })?;
    let opened_before = probe
        .opened_stamp(&reader)
        .map_err(|source| ProducerIdentityError::ExecutableIo {
            path: path.clone(),
            operation: "read initial opened-file metadata for",
            source,
        })?
        .ok_or_else(|| ProducerIdentityError::ExecutableNotRegular {
            path: path.clone(),
            observation: "the initial opened-file observation",
        })?;
    let path_before = probe
        .path_stamp(&path)
        .map_err(|source| ProducerIdentityError::ExecutableIo {
            path: path.clone(),
            operation: "read initial path metadata for",
            source,
        })?
        .ok_or_else(|| ProducerIdentityError::ExecutableNotRegular {
            path: path.clone(),
            observation: "the initial path observation",
        })?;
    if loaded_before != opened_before || loaded_before != path_before {
        return Err(ProducerIdentityError::ExecutableChanged {
            path,
            cause: "the loaded image, opened handle, and executable path initially named different file states",
        });
    }

    let mut hasher = blake3::Hasher::new_derive_key(RUSTC_EXECUTABLE_FINGERPRINT_CONTEXT);
    let mut buffer = [0_u8; READ_BUFFER_SIZE];
    let mut total_read = 0_u64;
    loop {
        let read = probe.read(&mut reader, &mut buffer).map_err(|source| {
            ProducerIdentityError::ExecutableIo {
                path: path.clone(),
                operation: "read bytes from",
                source,
            }
        })?;
        if read == 0 {
            break;
        }
        total_read = total_read
            .checked_add(u64::try_from(read).expect("read count is bounded by the fixed buffer"))
            .ok_or_else(|| ProducerIdentityError::ExecutableIo {
                path: path.clone(),
                operation: "count bytes read from",
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "executable byte count overflowed u64",
                ),
            })?;
        if total_read > loaded_before.len {
            return Err(ProducerIdentityError::ExecutableLengthMismatch {
                path,
                expected: loaded_before.len,
                read: total_read,
            });
        }
        hasher.update(&buffer[..read]);
    }

    let loaded_after = probe
        .loaded_stamp(&reader)
        .map_err(|source| ProducerIdentityError::ExecutableIo {
            path: path.clone(),
            operation: "read final loaded-image metadata for",
            source,
        })?
        .ok_or_else(|| ProducerIdentityError::ExecutableNotRegular {
            path: path.clone(),
            observation: "the final loaded-image observation",
        })?;
    let opened_after = probe
        .opened_stamp(&reader)
        .map_err(|source| ProducerIdentityError::ExecutableIo {
            path: path.clone(),
            operation: "read final opened-file metadata for",
            source,
        })?
        .ok_or_else(|| ProducerIdentityError::ExecutableNotRegular {
            path: path.clone(),
            observation: "the final opened-file observation",
        })?;
    let path_after = probe
        .path_stamp(&path)
        .map_err(|source| ProducerIdentityError::ExecutableIo {
            path: path.clone(),
            operation: "read final path metadata for",
            source,
        })?
        .ok_or_else(|| ProducerIdentityError::ExecutableNotRegular {
            path: path.clone(),
            observation: "the final path observation",
        })?;
    if loaded_before != loaded_after || loaded_before != opened_after || loaded_before != path_after
    {
        return Err(ProducerIdentityError::ExecutableChanged {
            path,
            cause: "loaded-image, opened-file, or path metadata changed between the initial and final observations",
        });
    }
    if total_read != loaded_before.len {
        return Err(ProducerIdentityError::ExecutableLengthMismatch {
            path,
            expected: loaded_before.len,
            read: total_read,
        });
    }

    Ok(hasher.finalize().to_hex().to_string())
}

fn derive_producer_identity<P: ExecutableProbe>(
    probe: &mut P,
    compiled_commit: Option<&str>,
    compiled_source_state: Option<&str>,
) -> Result<NlaiProducerIdentity, ProducerIdentityError> {
    let commit = canonical_compiled_identity("Rust commit", compiled_commit, &[40, 64])?;
    let source_state_fingerprint = canonical_compiled_identity(
        "NLAI source-state fingerprint",
        compiled_source_state,
        &[FINGERPRINT_LENGTH],
    )?;
    let executable_fingerprint = executable_fingerprint_with(probe)?;
    Ok(NlaiProducerIdentity {
        commit: commit.to_owned(),
        source_state_fingerprint: source_state_fingerprint.to_owned(),
        executable_fingerprint,
    })
}

static PRODUCER_IDENTITY: LazyLock<NlaiProducerIdentity> = LazyLock::new(|| {
    derive_producer_identity(
        &mut FileSystemProbe,
        rustc_session::nlai_rust_commit_hash(),
        rustc_session::nlai_rust_source_state_fingerprint(),
    )
    .unwrap_or_else(|error| bug!("[invariant] unable to establish NLAI producer identity: {error}"))
});

/// Force the process-wide snapshot before compiler inputs are observed when NLAI is enabled.
/// Invalid or non-Unicode values fail as user input before any identity operation.
pub(crate) fn initialize_if_requested() -> Option<NlaiProducerIdentity> {
    if activation_from_environment() == Some(Activation::Enabled) {
        Some(producer_identity().clone())
    } else {
        None
    }
}

/// Return the validated compile-time identity and measured bytes of this exact rustc process.
/// Invocation environment variables cannot influence any field.
fn producer_identity() -> &'static NlaiProducerIdentity {
    &PRODUCER_IDENTITY
}

#[cfg(test)]
mod tests;
