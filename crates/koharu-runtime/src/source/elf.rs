use std::{
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

use anyhow::{Context, Result, ensure};

/// The kernel requires every `PT_LOAD` segment to satisfy
/// `p_offset % PAGE_SIZE == p_vaddr % PAGE_SIZE`.
const ELF_PAGE_SIZE: u64 = 4096;
const PT_LOAD: u32 = 1;

/// PyTorch's ROCm libtorch ships `libtorch_cpu.so` with a zero-size `PT_LOAD`
/// whose file offset is not page-aligned (an LLD linker artifact). Newer
/// kernels reject such a segment on `dlopen`, so realign it after extraction.
pub(crate) fn fix_load_alignment(directory: &Path) -> Result<()> {
    for entry in fs::read_dir(directory)
        .with_context(|| format!("failed to read {}", directory.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        fix_file(&path)?;
    }
    Ok(())
}

fn fix_file(path: &Path) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;

    let mut header = [0u8; 64];
    let read = file
        .read(&mut header)
        .with_context(|| format!("failed to read {}", path.display()))?;
    if read < 16
        || &header[0..4] != b"\x7fELF"
        || header[4] != 2
        || header[5] != 1
        || u16::from_le_bytes([header[16], header[17]]) != 3
    {
        return Ok(()); // not a little-endian 64-bit ELF shared object
    }

    let e_phoff = u64::from_le_bytes(header[32..40].try_into().expect("valid slice"));
    let e_phentsize = u16::from_le_bytes([header[54], header[55]]) as u64;
    let e_phnum = u16::from_le_bytes([header[56], header[57]]) as usize;
    ensure!(
        e_phentsize >= 56,
        "{} uses an unsupported program header size {e_phentsize}",
        path.display()
    );

    file.seek(SeekFrom::Start(e_phoff))
        .with_context(|| format!("failed to seek program headers of {}", path.display()))?;
    let mut phdrs = vec![0u8; e_phentsize as usize * e_phnum];
    file.read_exact(&mut phdrs)
        .with_context(|| format!("failed to read program headers of {}", path.display()))?;

    let mut patched = false;
    for index in 0..e_phnum {
        let offset = index * e_phentsize as usize;
        let p_type = u32::from_le_bytes(phdrs[offset..offset + 4].try_into().expect("valid slice"));
        if p_type != PT_LOAD {
            continue;
        }
        let p_offset =
            u64::from_le_bytes(phdrs[offset + 8..offset + 16].try_into().expect("valid slice"));
        let p_vaddr =
            u64::from_le_bytes(phdrs[offset + 16..offset + 24].try_into().expect("valid slice"));
        let p_filesz =
            u64::from_le_bytes(phdrs[offset + 32..offset + 40].try_into().expect("valid slice"));

        if p_offset % ELF_PAGE_SIZE == p_vaddr % ELF_PAGE_SIZE {
            continue;
        }
        ensure!(
            p_filesz == 0,
            "{} has a non-empty misaligned load segment (offset {p_offset:#x}, address {p_vaddr:#x})",
            path.display()
        );

        // Align the unused offset to the same page position as the address.
        let delta = (p_vaddr % ELF_PAGE_SIZE + ELF_PAGE_SIZE - p_offset % ELF_PAGE_SIZE)
            % ELF_PAGE_SIZE;
        phdrs[offset + 8..offset + 16].copy_from_slice(&(p_offset + delta).to_le_bytes());
        patched = true;
    }

    if patched {
        file.seek(SeekFrom::Start(e_phoff))
            .with_context(|| format!("failed to seek program headers of {}", path.display()))?;
        file.write_all(&phdrs)
            .with_context(|| format!("failed to patch {}", path.display()))?;
        file.flush()
            .with_context(|| format!("failed to flush {}", path.display()))?;
    }
    Ok(())
}
