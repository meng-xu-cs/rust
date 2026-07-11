use std::collections::VecDeque;
use std::io;
use std::path::{Path, PathBuf};

use super::*;

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const SOURCE_STATE: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const VECTOR_BYTES: &[u8] = b"NLAI rustc executable vector\n";

#[test]
fn u1_2c2b1_only_canonical_true_values_request_the_eager_snapshot() {
    for value in ["1", "true", "yes", "on"] {
        assert_eq!(super::super::parse_activation(value), Some(super::super::Activation::Enabled));
    }
    for value in ["0", "false", "no", "off"] {
        assert_eq!(super::super::parse_activation(value), Some(super::super::Activation::Disabled));
    }
    for value in ["", "TRUE", "unexpected"] {
        assert_eq!(super::super::parse_activation(value), None);
    }
}

struct FakeReader {
    bytes: &'static [u8],
    offset: usize,
    fail: bool,
}

struct FakeProbe {
    current_exe_error: bool,
    open_error: bool,
    reader: Option<FakeReader>,
    stamps: VecDeque<io::Result<Option<ExecutableFileStamp>>>,
}

impl FakeProbe {
    fn valid(bytes: &'static [u8]) -> Self {
        let stamp = fake_stamp(bytes.len() as u64, 1);
        Self {
            current_exe_error: false,
            open_error: false,
            reader: Some(FakeReader { bytes, offset: 0, fail: false }),
            stamps: VecDeque::from([
                Ok(Some(stamp.clone())),
                Ok(Some(stamp.clone())),
                Ok(Some(stamp.clone())),
                Ok(Some(stamp.clone())),
                Ok(Some(stamp.clone())),
                Ok(Some(stamp)),
            ]),
        }
    }
}

impl ExecutableProbe for FakeProbe {
    type Reader = FakeReader;

    fn current_exe(&mut self) -> io::Result<PathBuf> {
        if self.current_exe_error {
            Err(io::Error::new(io::ErrorKind::NotFound, "injected current_exe failure"))
        } else {
            Ok(PathBuf::from("/fake/rustc"))
        }
    }

    fn open_loaded(&mut self, _path: &Path) -> io::Result<Self::Reader> {
        if self.open_error {
            Err(io::Error::new(io::ErrorKind::PermissionDenied, "injected open failure"))
        } else {
            Ok(self.reader.take().expect("one fake open"))
        }
    }

    fn loaded_stamp(&mut self, _reader: &Self::Reader) -> io::Result<Option<ExecutableFileStamp>> {
        self.stamps.pop_front().expect("one injected loaded-image stamp")
    }

    fn opened_stamp(&mut self, _reader: &Self::Reader) -> io::Result<Option<ExecutableFileStamp>> {
        self.stamps.pop_front().expect("one injected opened-file stamp")
    }

    fn path_stamp(&mut self, _path: &Path) -> io::Result<Option<ExecutableFileStamp>> {
        self.stamps.pop_front().expect("one injected path stamp")
    }

    fn read(&mut self, reader: &mut Self::Reader, buffer: &mut [u8]) -> io::Result<usize> {
        if reader.fail {
            return Err(io::Error::other("injected read failure"));
        }
        if reader.offset == reader.bytes.len() {
            return Ok(0);
        }
        let count = (reader.bytes.len() - reader.offset).min(buffer.len());
        buffer[..count].copy_from_slice(&reader.bytes[reader.offset..reader.offset + count]);
        reader.offset += count;
        Ok(count)
    }
}

fn fake_stamp(len: u64, generation: u64) -> ExecutableFileStamp {
    ExecutableFileStamp {
        len,
        #[cfg(unix)]
        platform: PlatformFileStamp {
            device: 1,
            inode: generation,
            mode: 0o100755,
            links: 1,
            owner: 1,
            group: 1,
            special_device: 0,
            modified_seconds: generation as i64,
            modified_nanoseconds: 0,
            changed_seconds: generation as i64,
            changed_nanoseconds: 0,
        },
        #[cfg(windows)]
        platform: PlatformFileStamp {
            attributes: 0,
            creation_time: 1,
            last_write_time: generation,
        },
        #[cfg(not(any(unix, windows)))]
        platform: PlatformFileStamp,
    }
}

#[test]
fn u1_2c2b1_fixed_executable_fingerprint_vector_pins_context_and_bytes() {
    let fingerprint = executable_fingerprint_with(&mut FakeProbe::valid(VECTOR_BYTES)).unwrap();
    assert_eq!(fingerprint, "5260f9e20f4c29837f3181f8cf8064557d3be714a47fd0f8ee0d0a8f45a82e82");
}

#[test]
fn u1_2c2b1_producer_identity_uses_only_validated_compiled_fields() {
    let identity = derive_producer_identity(
        &mut FakeProbe::valid(VECTOR_BYTES),
        Some(COMMIT),
        Some(SOURCE_STATE),
    )
    .unwrap();
    assert_eq!(identity.commit, COMMIT);
    assert_eq!(identity.source_state_fingerprint, SOURCE_STATE);
    assert_eq!(identity.executable_fingerprint.len(), FINGERPRINT_LENGTH);

    let long_commit = "a".repeat(64);
    assert!(
        derive_producer_identity(
            &mut FakeProbe::valid(VECTOR_BYTES),
            Some(long_commit.as_str()),
            Some(SOURCE_STATE)
        )
        .is_ok()
    );

    let uppercase_commit = "A".repeat(40);
    for invalid in [
        None,
        Some("unknown"),
        Some(""),
        Some("g234567890123456789012345678901234567890"),
        Some(uppercase_commit.as_str()),
    ] {
        assert!(
            derive_producer_identity(
                &mut FakeProbe::valid(VECTOR_BYTES),
                invalid,
                Some(SOURCE_STATE)
            )
            .is_err()
        );
    }
    let uppercase_fingerprint = "A".repeat(64);
    for invalid in [None, Some("unknown"), Some("0"), Some(uppercase_fingerprint.as_str())] {
        assert!(
            derive_producer_identity(&mut FakeProbe::valid(VECTOR_BYTES), Some(COMMIT), invalid)
                .is_err()
        );
    }
}

#[test]
fn u1_2c2b1_executable_discovery_open_read_and_metadata_fail_loudly() {
    let mut current = FakeProbe::valid(VECTOR_BYTES);
    current.current_exe_error = true;
    assert!(matches!(
        executable_fingerprint_with(&mut current),
        Err(ProducerIdentityError::CurrentExecutable(_))
    ));

    let mut open = FakeProbe::valid(VECTOR_BYTES);
    open.open_error = true;
    assert!(matches!(
        executable_fingerprint_with(&mut open),
        Err(ProducerIdentityError::ExecutableIo { operation: "open the loaded image for", .. })
    ));

    let mut read = FakeProbe::valid(VECTOR_BYTES);
    read.reader.as_mut().unwrap().fail = true;
    assert!(matches!(
        executable_fingerprint_with(&mut read),
        Err(ProducerIdentityError::ExecutableIo { operation: "read bytes from", .. })
    ));

    let mut metadata = FakeProbe::valid(VECTOR_BYTES);
    metadata.stamps[0] = Err(io::Error::other("injected metadata failure"));
    assert!(matches!(
        executable_fingerprint_with(&mut metadata),
        Err(ProducerIdentityError::ExecutableIo {
            operation: "read initial loaded-image metadata for",
            ..
        })
    ));
}

#[test]
fn u1_2c2b1_nonregular_replacement_and_mutation_fail_loudly() {
    let mut nonregular = FakeProbe::valid(VECTOR_BYTES);
    nonregular.stamps[0] = Ok(None);
    assert!(matches!(
        executable_fingerprint_with(&mut nonregular),
        Err(ProducerIdentityError::ExecutableNotRegular { .. })
    ));

    let mut replacement = FakeProbe::valid(VECTOR_BYTES);
    replacement.stamps[1] = Ok(Some(fake_stamp(VECTOR_BYTES.len() as u64, 2)));
    replacement.stamps[2] = Ok(Some(fake_stamp(VECTOR_BYTES.len() as u64, 2)));
    assert!(matches!(
        executable_fingerprint_with(&mut replacement),
        Err(ProducerIdentityError::ExecutableChanged { .. })
    ));

    let mut mutation = FakeProbe::valid(VECTOR_BYTES);
    mutation.stamps[5] = Ok(Some(fake_stamp(VECTOR_BYTES.len() as u64, 2)));
    assert!(matches!(
        executable_fingerprint_with(&mut mutation),
        Err(ProducerIdentityError::ExecutableChanged { .. })
    ));
}

#[test]
fn u1_2c2b1_premature_eof_cannot_hash_only_an_executable_prefix() {
    let mut probe = FakeProbe::valid(VECTOR_BYTES);
    probe.reader.as_mut().unwrap().bytes = &VECTOR_BYTES[..5];

    assert!(matches!(
        executable_fingerprint_with(&mut probe),
        Err(ProducerIdentityError::ExecutableLengthMismatch {
            expected,
            read: 5,
            ..
        }) if expected == VECTOR_BYTES.len() as u64
    ));
}

#[test]
fn u1_2c2b1_excess_bytes_cannot_extend_the_stamped_executable() {
    let mut probe = FakeProbe::valid(VECTOR_BYTES);
    for stamp in &mut probe.stamps {
        stamp.as_mut().unwrap().as_mut().unwrap().len = 5;
    }

    assert!(matches!(
        executable_fingerprint_with(&mut probe),
        Err(ProducerIdentityError::ExecutableLengthMismatch {
            expected: 5,
            read,
            ..
        }) if read == VECTOR_BYTES.len() as u64
    ));
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
#[test]
fn u1_2c2b1_production_probe_hashes_the_running_test_executable() {
    let fingerprint = executable_fingerprint_with(&mut FileSystemProbe).unwrap();
    assert_eq!(fingerprint.len(), FINGERPRINT_LENGTH);
    assert!(fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));
}

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
#[test]
fn u1_2c2b1_production_probe_fails_closed_without_a_loaded_image_handle() {
    assert!(matches!(
        executable_fingerprint_with(&mut FileSystemProbe),
        Err(ProducerIdentityError::ExecutableIo {
            operation: "open the loaded image for",
            source,
            ..
        }) if source.kind() == io::ErrorKind::Unsupported
    ));
}
