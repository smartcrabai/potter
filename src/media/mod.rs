use crate::error::{ErrorCode, PotError, Result};

const MIN_SAMPLE_RATE: u32 = 8_000;
const MAX_SAMPLE_RATE: u32 = 384_000;
const MAX_CHANNELS: u16 = 32;

/// Interleaved floating-point audio samples.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

#[derive(Clone, Copy)]
struct WavFormat {
    code: u16,
    channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
}

/// Decode RIFF/WAVE PCM (8/16/24/32-bit) or IEEE float32 audio.
pub fn decode_wav(bytes: &[u8]) -> Result<AudioBuffer> {
    if bytes.get(..4) != Some(&b"RIFF"[..]) || bytes.get(8..12) != Some(&b"WAVE"[..]) {
        return Err(wav_import_error("input is not a RIFF/WAVE file"));
    }
    if bytes.len() < 12 {
        return Err(wav_import_error("RIFF/WAVE header is truncated"));
    }

    let riff_size = usize::try_from(read_u32(bytes, 4))
        .map_err(|_| wav_import_error("RIFF size is out of range"))?;
    let riff_end = riff_size
        .checked_add(8)
        .ok_or_else(|| wav_import_error("RIFF size overflows"))?;
    if riff_size < 4 || riff_end > bytes.len() {
        return Err(wav_import_error("RIFF size exceeds the available input"));
    }

    let mut position = 12_usize;
    let mut format = None;
    let mut data = None;
    while position < riff_end {
        if riff_end - position < 8 {
            return Err(wav_import_error("RIFF chunk header is truncated"));
        }
        let chunk_id = bytes
            .get(position..position + 4)
            .ok_or_else(|| wav_import_error("RIFF chunk header is truncated"))?;
        let chunk_size = usize::try_from(read_u32(bytes, position + 4))
            .map_err(|_| wav_import_error("RIFF chunk size is out of range"))?;
        let payload_start = position + 8;
        let payload_end = payload_start
            .checked_add(chunk_size)
            .ok_or_else(|| wav_import_error("RIFF chunk size overflows"))?;
        if payload_end > riff_end {
            return Err(wav_import_error("RIFF chunk exceeds its container"));
        }
        let payload = &bytes[payload_start..payload_end];

        if chunk_id == b"fmt " {
            if format.is_some() {
                return Err(wav_import_error("RIFF/WAVE contains duplicate fmt chunks"));
            }
            format = Some(parse_format(payload)?);
        } else if chunk_id == b"data" {
            if data.is_some() {
                return Err(wav_import_error("RIFF/WAVE contains duplicate data chunks"));
            }
            data = Some(payload);
        }

        position = payload_end
            .checked_add(chunk_size % 2)
            .ok_or_else(|| wav_import_error("RIFF chunk padding overflows"))?;
        if position > riff_end {
            return Err(wav_import_error("RIFF chunk padding is truncated"));
        }
    }

    let format = format.ok_or_else(|| wav_import_error("RIFF/WAVE fmt chunk is missing"))?;
    let data = data.ok_or_else(|| wav_import_error("RIFF/WAVE data chunk is missing"))?;
    let block_align = usize::from(format.block_align);
    if !data.len().is_multiple_of(block_align) {
        return Err(wav_import_error(
            "WAVE data is not aligned to complete sample frames",
        ));
    }

    let sample_width = usize::from(format.bits_per_sample / 8);
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(data.len() / sample_width)
        .map_err(|_| wav_import_error("WAVE sample allocation failed"))?;
    for sample in data.chunks_exact(sample_width) {
        let decoded = match (format.code, format.bits_per_sample) {
            (1, 8) => (f32::from(sample[0]) - 128.0) / 128.0,
            (1, 16) => {
                let value = i16::from_le_bytes([sample[0], sample[1]]);
                f32::from(value) / 32_768.0
            }
            (1, 24) => {
                let sign = if sample[2] & 0x80 == 0 { 0 } else { 0xff };
                let value = i32::from_le_bytes([sample[0], sample[1], sample[2], sign]);
                pcm_i32_to_f32(value, 8_388_608.0)
            }
            (1, 32) => {
                let value = i32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]);
                pcm_i32_to_f32(value, 2_147_483_648.0)
            }
            (3, 32) => f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]),
            _ => return Err(wav_import_error("WAVE encoding is not supported")),
        };
        samples.push(sanitize_sample(decoded));
    }

    Ok(AudioBuffer {
        sample_rate: format.sample_rate,
        channels: format.channels,
        samples,
    })
}

/// Encode interleaved audio as canonical 16-bit PCM RIFF/WAVE.
pub fn encode_wav(audio: &AudioBuffer) -> Result<Vec<u8>> {
    validate_audio_format(audio.sample_rate, audio.channels, ErrorCode::ExportFailed)?;
    if !audio
        .samples
        .len()
        .is_multiple_of(usize::from(audio.channels))
    {
        return Err(wav_export_error(
            "audio samples are not aligned to complete channel frames",
        ));
    }

    let data_size = audio
        .samples
        .len()
        .checked_mul(2)
        .ok_or_else(|| wav_export_error("WAVE data size overflows"))?;
    let riff_size = 36_usize
        .checked_add(data_size)
        .ok_or_else(|| wav_export_error("RIFF size overflows"))?;
    let data_size_u32 = u32::try_from(data_size)
        .map_err(|_| wav_export_error("WAVE data exceeds the RIFF size limit"))?;
    let riff_size_u32 = u32::try_from(riff_size)
        .map_err(|_| wav_export_error("WAVE data exceeds the RIFF size limit"))?;
    let capacity = 44_usize
        .checked_add(data_size)
        .ok_or_else(|| wav_export_error("WAVE output size overflows"))?;
    let block_align = audio
        .channels
        .checked_mul(2)
        .ok_or_else(|| wav_export_error("WAVE channel alignment overflows"))?;
    let byte_rate = audio
        .sample_rate
        .checked_mul(u32::from(block_align))
        .ok_or_else(|| wav_export_error("WAVE byte rate overflows"))?;

    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| wav_export_error("WAVE output allocation failed"))?;
    output.extend_from_slice(b"RIFF");
    output.extend_from_slice(&riff_size_u32.to_le_bytes());
    output.extend_from_slice(b"WAVEfmt ");
    output.extend_from_slice(&16_u32.to_le_bytes());
    output.extend_from_slice(&1_u16.to_le_bytes());
    output.extend_from_slice(&audio.channels.to_le_bytes());
    output.extend_from_slice(&audio.sample_rate.to_le_bytes());
    output.extend_from_slice(&byte_rate.to_le_bytes());
    output.extend_from_slice(&block_align.to_le_bytes());
    output.extend_from_slice(&16_u16.to_le_bytes());
    output.extend_from_slice(b"data");
    output.extend_from_slice(&data_size_u32.to_le_bytes());
    for sample in &audio.samples {
        let scaled = (f64::from(sanitize_sample(*sample)) * 32_768.0)
            .round()
            .clamp(-32_768.0, 32_767.0);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the PCM value is rounded and clamped to the i16 range before conversion"
        )]
        let pcm_sample = scaled as i16;
        output.extend_from_slice(&pcm_sample.to_le_bytes());
    }
    Ok(output)
}

/// Encode interleaved audio as a 24-bit PCM FLAC stream.
///
/// Input samples are clamped to `[-1, 1]` and rounded to signed 24-bit PCM.
/// Frames use the FLAC verbatim subframe, so encoding is independent of
/// external codecs and preserves the quantized PCM values exactly.
///
/// # Errors
///
/// Returns `ExportFailed` when the audio format is invalid, FLAC's channel or
/// sample-count limits are exceeded, or output allocation fails.
pub fn encode_flac(audio: &AudioBuffer) -> Result<Vec<u8>> {
    const BLOCK_SIZE: usize = 4096;
    const BLOCK_SIZE_U64: u64 = 4096;
    const BITS_PER_SAMPLE: u64 = 24;
    const BITS_PER_SAMPLE_CODE: u8 = 6;
    const MAX_SAMPLE_COUNT: u64 = 0x0f_ffff_ffff;

    validate_audio_format(audio.sample_rate, audio.channels, ErrorCode::ExportFailed)?;
    if audio.channels > 8 {
        return Err(wav_export_error("FLAC supports at most 8 channels"));
    }
    let channels = usize::from(audio.channels);
    if !audio.samples.len().is_multiple_of(channels) {
        return Err(wav_export_error(
            "audio samples are not aligned to complete channel frames",
        ));
    }
    let (sample_rate_code, sample_rate_bytes, sample_rate_extension) =
        flac_sample_rate_encoding(audio.sample_rate)?;

    let sample_count_usize = audio.samples.len() / channels;
    let sample_count = u64::try_from(sample_count_usize)
        .map_err(|_| wav_export_error("FLAC sample count is out of range"))?;
    if sample_count > MAX_SAMPLE_COUNT {
        return Err(wav_export_error(
            "FLAC sample count exceeds the format limit",
        ));
    }
    let frame_count = sample_count.div_ceil(BLOCK_SIZE_U64);
    let mut remaining_samples = sample_count_usize;
    let mut frame_bytes = 0_usize;
    let mut minimum_frame_size = u32::MAX;
    let mut maximum_frame_size = 0_u32;
    for frame_number in 0..frame_count {
        let block_size = remaining_samples.min(BLOCK_SIZE);
        let frame_size =
            flac_frame_size(block_size, channels, frame_number, sample_rate_extension)?;
        let frame_size_u32 = u32::try_from(frame_size)
            .map_err(|_| wav_export_error("FLAC frame size is out of range"))?;
        if frame_size_u32 > 0x00ff_ffff {
            return Err(wav_export_error(
                "FLAC frame size exceeds the metadata limit",
            ));
        }
        minimum_frame_size = minimum_frame_size.min(frame_size_u32);
        maximum_frame_size = maximum_frame_size.max(frame_size_u32);
        frame_bytes = frame_bytes
            .checked_add(frame_size)
            .ok_or_else(|| wav_export_error("FLAC output size overflows"))?;
        remaining_samples -= block_size;
    }
    if frame_count == 0 {
        minimum_frame_size = 0;
    }
    let capacity = 42_usize
        .checked_add(frame_bytes)
        .ok_or_else(|| wav_export_error("FLAC output size overflows"))?;

    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| wav_export_error("FLAC output allocation failed"))?;
    output.extend_from_slice(b"fLaC");
    output.extend_from_slice(&[0x80, 0, 0, 34]);
    let stream_block_size = u16::try_from(BLOCK_SIZE)
        .map_err(|_| wav_export_error("FLAC block size is out of range"))?;
    output.extend_from_slice(&stream_block_size.to_be_bytes());
    output.extend_from_slice(&stream_block_size.to_be_bytes());
    let minimum_frame_size_bytes = minimum_frame_size.to_be_bytes();
    output.extend_from_slice(&minimum_frame_size_bytes[1..]);
    let maximum_frame_size_bytes = maximum_frame_size.to_be_bytes();
    output.extend_from_slice(&maximum_frame_size_bytes[1..]);
    let stream_format = (u64::from(audio.sample_rate) << 44)
        | (u64::from(audio.channels - 1) << 41)
        | ((BITS_PER_SAMPLE - 1) << 36)
        | sample_count;
    output.extend_from_slice(&stream_format.to_be_bytes());
    output.extend_from_slice(&[0; 16]);

    let mut first_sample = 0_usize;
    let mut frame_number = 0_u64;
    let channel_assignment = (u8::try_from(audio.channels - 1)
        .map_err(|_| wav_export_error("FLAC channel count is out of range"))?
        << 4)
        | (BITS_PER_SAMPLE_CODE << 1);
    while first_sample < sample_count_usize {
        let remaining = sample_count_usize - first_sample;
        let block_size = remaining.min(BLOCK_SIZE);
        let (block_size_code, block_size_bytes, size_extension) =
            flac_block_size_encoding(block_size)?;
        let frame_start = output.len();
        output.extend_from_slice(&[0xff, 0xf8, (block_size_code << 4) | sample_rate_code]);
        output.push(channel_assignment);
        append_flac_utf8_uint(&mut output, frame_number)?;
        match size_extension {
            0 => {}
            1 => output.push(block_size_bytes[0]),
            _ => output.extend_from_slice(&block_size_bytes),
        }
        match sample_rate_extension {
            0 => {}
            1 => output.push(sample_rate_bytes[0]),
            _ => output.extend_from_slice(&sample_rate_bytes),
        }
        let header_crc = flac_crc8(&output[frame_start..]);
        output.push(header_crc);

        for channel in 0..channels {
            output.push(0x02);
            for sample_offset in 0..block_size {
                let interleaved_index = (first_sample + sample_offset) * channels + channel;
                let pcm_sample = quantize_to_i24(audio.samples[interleaved_index]);
                let bytes = pcm_sample.to_be_bytes();
                output.extend_from_slice(&bytes[1..]);
            }
        }
        let frame_crc = flac_crc16(&output[frame_start..]);
        output.extend_from_slice(&frame_crc.to_be_bytes());

        first_sample += block_size;
        frame_number += 1;
    }

    Ok(output)
}

fn flac_frame_size(
    block_size: usize,
    channels: usize,
    frame_number: u64,
    sample_rate_extension: usize,
) -> Result<usize> {
    let (_, _, size_extension) = flac_block_size_encoding(block_size)?;
    let pcm_size = block_size
        .checked_mul(channels)
        .and_then(|size| size.checked_mul(3))
        .ok_or_else(|| wav_export_error("FLAC frame size overflows"))?;
    7_usize
        .checked_add(flac_utf8_uint_length(frame_number)?)
        .and_then(|size| size.checked_add(size_extension))
        .and_then(|size| size.checked_add(sample_rate_extension))
        .and_then(|size| size.checked_add(channels))
        .and_then(|size| size.checked_add(pcm_size))
        .ok_or_else(|| wav_export_error("FLAC frame size overflows"))
}
fn flac_block_size_encoding(block_size: usize) -> Result<(u8, [u8; 2], usize)> {
    if block_size < 256 {
        return Ok((
            6,
            [
                u8::try_from(block_size - 1)
                    .map_err(|_| wav_export_error("FLAC block size is out of range"))?,
                0,
            ],
            1,
        ));
    }
    if block_size.is_power_of_two() && block_size <= 32_768 {
        let code = u8::try_from(block_size.trailing_zeros())
            .map_err(|_| wav_export_error("FLAC block size is out of range"))?;
        return Ok((code, [0, 0], 0));
    }
    Ok((
        7,
        u16::try_from(block_size - 1)
            .map_err(|_| wav_export_error("FLAC block size is out of range"))?
            .to_be_bytes(),
        2,
    ))
}

fn flac_sample_rate_encoding(sample_rate: u32) -> Result<(u8, [u8; 2], usize)> {
    let standard_code = match sample_rate {
        88_200 => 1,
        176_400 => 2,
        192_000 => 3,
        8_000 => 4,
        16_000 => 5,
        22_050 => 6,
        24_000 => 7,
        32_000 => 8,
        44_100 => 9,
        48_000 => 10,
        96_000 => 11,
        _ => 0,
    };
    if standard_code != 0 {
        return Ok((standard_code, [0, 0], 0));
    }
    if sample_rate.is_multiple_of(1_000) {
        let kilo_hertz = sample_rate / 1_000;
        if u8::try_from(kilo_hertz).is_ok() {
            return Ok((
                12,
                [
                    u8::try_from(kilo_hertz)
                        .map_err(|_| wav_export_error("FLAC sample rate is out of range"))?,
                    0,
                ],
                1,
            ));
        }
    }
    if u16::try_from(sample_rate).is_ok() {
        return Ok((
            13,
            u16::try_from(sample_rate)
                .map_err(|_| wav_export_error("FLAC sample rate is out of range"))?
                .to_be_bytes(),
            2,
        ));
    }
    if sample_rate.is_multiple_of(10) {
        let sample_rate_tens = sample_rate / 10;
        if u16::try_from(sample_rate_tens).is_ok() {
            return Ok((
                14,
                u16::try_from(sample_rate_tens)
                    .map_err(|_| wav_export_error("FLAC sample rate is out of range"))?
                    .to_be_bytes(),
                2,
            ));
        }
    }
    Ok((0, [0, 0], 0))
}

fn flac_utf8_uint_length(value: u64) -> Result<usize> {
    if value < 0x80 {
        Ok(1)
    } else if value < (1 << 11) {
        Ok(2)
    } else if value < (1 << 16) {
        Ok(3)
    } else if value < (1 << 21) {
        Ok(4)
    } else if value < (1 << 26) {
        Ok(5)
    } else if value < (1 << 31) {
        Ok(6)
    } else if value < (1 << 36) {
        Ok(7)
    } else {
        Err(wav_export_error(
            "FLAC frame number exceeds the format limit",
        ))
    }
}

fn append_flac_utf8_uint(output: &mut Vec<u8>, value: u64) -> Result<()> {
    let byte_count = flac_utf8_uint_length(value)?;
    if byte_count == 1 {
        output.push(
            u8::try_from(value)
                .map_err(|_| wav_export_error("FLAC frame number is out of range"))?,
        );
        return Ok(());
    }
    let continuation_count = byte_count - 1;
    let first_prefix = 0xff_u8 << (8 - byte_count);
    let first_payload = u8::try_from(value >> (continuation_count * 6))
        .map_err(|_| wav_export_error("FLAC frame number is out of range"))?;
    output.push(first_prefix | first_payload);
    for continuation_index in (0..continuation_count).rev() {
        let part = u8::try_from((value >> (continuation_index * 6)) & 0x3f)
            .map_err(|_| wav_export_error("FLAC frame number is out of range"))?;
        output.push(0x80 | part);
    }
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "sample is rounded and clamped to the signed 24-bit range before conversion"
)]
fn quantize_to_i24(sample: f32) -> i32 {
    (f64::from(sanitize_sample(sample)) * 8_388_608.0)
        .round()
        .clamp(-8_388_608.0, 8_388_607.0) as i32
}

fn flac_crc8(bytes: &[u8]) -> u8 {
    let mut crc = 0_u8;
    for byte in bytes {
        crc ^= byte;
        for _ in 0..8 {
            crc = if crc & 0x80 == 0 {
                crc << 1
            } else {
                (crc << 1) ^ 0x07
            };
        }
    }
    crc
}

fn flac_crc16(bytes: &[u8]) -> u16 {
    let mut crc = 0_u16;
    for byte in bytes {
        crc ^= u16::from(*byte) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 == 0 {
                crc << 1
            } else {
                (crc << 1) ^ 0x8005
            };
        }
    }
    crc
}

fn parse_format(payload: &[u8]) -> Result<WavFormat> {
    if payload.len() < 16 {
        return Err(wav_import_error("WAVE fmt chunk is shorter than 16 bytes"));
    }
    let format = WavFormat {
        code: read_u16(payload, 0),
        channels: read_u16(payload, 2),
        sample_rate: read_u32(payload, 4),
        byte_rate: read_u32(payload, 8),
        block_align: read_u16(payload, 12),
        bits_per_sample: read_u16(payload, 14),
    };
    validate_audio_format(format.sample_rate, format.channels, ErrorCode::ImportFailed)?;

    let supported_width = match format.code {
        1 => matches!(format.bits_per_sample, 8 | 16 | 24 | 32),
        3 => format.bits_per_sample == 32,
        _ => false,
    };
    if !supported_width {
        return Err(wav_import_error("WAVE sample format is not supported"));
    }

    let expected_align = format
        .channels
        .checked_mul(format.bits_per_sample / 8)
        .ok_or_else(|| wav_import_error("WAVE block alignment overflows"))?;
    let expected_byte_rate = format
        .sample_rate
        .checked_mul(u32::from(expected_align))
        .ok_or_else(|| wav_import_error("WAVE byte rate overflows"))?;
    if format.block_align != expected_align || format.byte_rate != expected_byte_rate {
        return Err(wav_import_error(
            "WAVE fmt chunk has inconsistent rates or alignment",
        ));
    }
    Ok(format)
}

fn validate_audio_format(sample_rate: u32, channels: u16, code: ErrorCode) -> Result<()> {
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&sample_rate) {
        return Err(PotError::new(
            code,
            "sample rate must be between 8000 and 384000 Hz",
        ));
    }
    if !(1..=MAX_CHANNELS).contains(&channels) {
        return Err(PotError::new(
            code,
            "channel count must be between 1 and 32",
        ));
    }
    Ok(())
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn sanitize_sample(sample: f32) -> f32 {
    if sample.is_nan() {
        0.0
    } else {
        sample.clamp(-1.0, 1.0)
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "integer PCM is intentionally normalized to f32 sample precision"
)]
fn pcm_i32_to_f32(sample: i32, scale: f32) -> f32 {
    sample as f32 / scale
}

fn wav_import_error(message: &str) -> PotError {
    PotError::new(ErrorCode::ImportFailed, message)
}

fn wav_export_error(message: &str) -> PotError {
    PotError::new(ErrorCode::ExportFailed, message)
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "test fixture construction and assertions"
    )]
    use crate::error::ErrorCode;

    use super::{AudioBuffer, decode_wav, encode_wav};

    fn make_wav(format: u16, channels: u16, sample_rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let block_align = channels * (bits / 8);
        let byte_rate = sample_rate * u32::from(block_align);
        let padded_len = data.len() + usize::from(!data.len().is_multiple_of(2));
        let mut wav = Vec::with_capacity(44 + padded_len);
        wav.extend_from_slice(b"RIFF");
        let riff_size = 36_u32 + u32::try_from(padded_len).expect("test data fits u32");
        wav.extend_from_slice(&riff_size.to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&format.to_le_bytes());
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&bits.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(
            &u32::try_from(data.len())
                .expect("test data fits u32")
                .to_le_bytes(),
        );
        wav.extend_from_slice(data);
        if !data.len().is_multiple_of(2) {
            wav.push(0);
        }
        wav
    }

    #[test]
    fn pcm_decode_preserves_interleaved_frame_alignment() {
        let wav = make_wav(1, 2, 48_000, 16, &[0, 0, 0, 64, 0, 192, 0, 224]);
        let decoded = decode_wav(&wav).expect("valid stereo PCM WAV should decode");
        assert_eq!(decoded.sample_rate, 48_000);
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.samples.len(), 4);
        assert!(decoded.samples[0].abs() < f32::EPSILON);
        assert!((decoded.samples[1] - 0.5).abs() < f32::EPSILON);
        assert!((decoded.samples[2] + 0.5).abs() < f32::EPSILON);
        assert!((decoded.samples[3] + 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn decodes_each_supported_pcm_width_and_float32() {
        let cases = [
            (8, vec![0, 128, 255], vec![-1.0, 0.0, 127.0 / 128.0]),
            (
                16,
                vec![0, 128, 0, 0, 255, 127],
                vec![-1.0, 0.0, 32_767.0 / 32_768.0],
            ),
            (
                24,
                vec![0, 0, 128, 0, 0, 0, 255, 255, 127],
                vec![-1.0, 0.0, 8_388_607.0 / 8_388_608.0],
            ),
            (
                32,
                vec![0, 0, 0, 128, 0, 0, 0, 0, 255, 255, 255, 127],
                vec![-1.0, 0.0, 2_147_483_647.0 / 2_147_483_648.0],
            ),
        ];
        for (bits, bytes, expected) in cases {
            let decoded = decode_wav(&make_wav(1, 1, 44_100, bits, &bytes))
                .expect("supported PCM WAV should decode");
            assert_eq!(decoded.samples.len(), expected.len());
            for (actual, expected) in decoded.samples.iter().zip(expected) {
                assert!((*actual - expected).abs() < f32::EPSILON);
            }
        }

        let float_bytes = [(-0.75_f32).to_le_bytes(), 0.25_f32.to_le_bytes()].concat();
        let decoded = decode_wav(&make_wav(3, 1, 44_100, 32, &float_bytes))
            .expect("float32 WAV should decode");
        let expected = [-0.75, 0.25];
        for (actual, expected) in decoded.samples.iter().zip(expected) {
            assert!((*actual - expected).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn float32_decode_clamps_out_of_range_and_non_finite_samples() {
        let float_bytes = [
            2.0_f32.to_le_bytes(),
            f32::INFINITY.to_le_bytes(),
            f32::NEG_INFINITY.to_le_bytes(),
            f32::NAN.to_le_bytes(),
        ]
        .concat();
        let decoded = decode_wav(&make_wav(3, 1, 44_100, 32, &float_bytes))
            .expect("valid float32 WAV should decode");
        let expected = [1.0, 1.0, -1.0, 0.0];
        for (actual, expected) in decoded.samples.iter().zip(expected) {
            assert!((*actual - expected).abs() < f32::EPSILON);
        }
        assert!(decoded.samples.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn encode_and_decode_round_trip_interleaved_pcm16_with_clamping() {
        let audio = AudioBuffer {
            sample_rate: 48_000,
            channels: 2,
            samples: vec![
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.5,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ],
        };
        let wav = encode_wav(&audio).expect("valid audio buffer should encode");
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(u32::from_le_bytes([wav[4], wav[5], wav[6], wav[7]]), 52);
        assert_eq!(u32::from_le_bytes([wav[16], wav[17], wav[18], wav[19]]), 16);
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            48_000
        );
        assert_eq!(
            u32::from_le_bytes([wav[28], wav[29], wav[30], wav[31]]),
            192_000
        );
        assert_eq!(u16::from_le_bytes([wav[32], wav[33]]), 4);
        assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 1);
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 2);
        assert_eq!(u16::from_le_bytes([wav[34], wav[35]]), 16);
        assert_eq!(u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]), 16);

        let decoded = decode_wav(&wav).expect("encoded WAV should decode");
        assert_eq!(decoded.sample_rate, audio.sample_rate);
        assert_eq!(decoded.channels, audio.channels);
        assert_eq!(decoded.samples.len(), 8);
        let expected = [-1.0, -0.5, 0.0, 0.5, 1.0, 0.0, 1.0, -1.0];
        for (actual, expected) in decoded.samples.iter().zip(expected) {
            assert!((*actual - expected).abs() <= 1.0 / 32_768.0);
        }
    }

    #[test]
    fn rejects_malformed_riff_chunks_and_misaligned_samples() {
        let valid = make_wav(1, 2, 48_000, 16, &[0; 4]);

        let mut truncated = valid.clone();
        truncated.truncate(43);
        assert!(matches!(
            decode_wav(&truncated),
            Err(error) if error.code == ErrorCode::ImportFailed
        ));

        let mut oversized_riff = valid.clone();
        oversized_riff[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_wav(&oversized_riff).is_err());

        let mut oversized_chunk = valid.clone();
        oversized_chunk[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_wav(&oversized_chunk).is_err());

        let misaligned = make_wav(1, 2, 48_000, 16, &[0; 3]);
        assert!(decode_wav(&misaligned).is_err());

        let unsupported = make_wav(6, 1, 48_000, 16, &[0; 2]);
        assert!(decode_wav(&unsupported).is_err());
        let invalid_rate = make_wav(1, 1, 7_999, 16, &[0; 2]);
        assert!(decode_wav(&invalid_rate).is_err());

        let invalid_channels = make_wav(1, 33, 48_000, 16, &[0; 66]);
        assert!(decode_wav(&invalid_channels).is_err());
    }
    #[test]
    fn wav_chunk_padding_and_format_boundaries_are_checked() {
        let valid = make_wav(1, 1, 8_000, 8, &[0, 128]);
        let mut with_odd_unknown_chunk = Vec::with_capacity(valid.len() + 10);
        with_odd_unknown_chunk.extend_from_slice(&valid[..12]);
        with_odd_unknown_chunk.extend_from_slice(b"JUNK");
        with_odd_unknown_chunk.extend_from_slice(&1_u32.to_le_bytes());
        with_odd_unknown_chunk.extend_from_slice(&[42, 0]);
        with_odd_unknown_chunk.extend_from_slice(&valid[12..]);
        let riff_size =
            u32::try_from(with_odd_unknown_chunk.len() - 8).expect("fixture size fits u32");
        with_odd_unknown_chunk[4..8].copy_from_slice(&riff_size.to_le_bytes());
        let decoded =
            decode_wav(&with_odd_unknown_chunk).expect("unknown padded chunks should be skipped");
        assert_eq!(decoded.samples, vec![-1.0, 0.0]);

        let mut missing_padding = valid.clone();
        missing_padding.extend_from_slice(b"JUNK");
        missing_padding.extend_from_slice(&1_u32.to_le_bytes());
        missing_padding.push(42);
        let riff_size = u32::try_from(missing_padding.len() - 8).expect("fixture size fits u32");
        missing_padding[4..8].copy_from_slice(&riff_size.to_le_bytes());
        assert!(decode_wav(&missing_padding).is_err());

        let upper_boundary = decode_wav(&make_wav(1, 32, 384_000, 8, &[128; 32]))
            .expect("maximum sample-rate and channel boundaries should decode");
        assert_eq!(upper_boundary.sample_rate, 384_000);
        assert_eq!(upper_boundary.channels, 32);
        assert_eq!(upper_boundary.samples, vec![0.0; 32]);
    }

    #[test]
    fn flac_encoder_writes_pcm_stream_metadata_and_frames() {
        let audio = AudioBuffer {
            sample_rate: 48_000,
            channels: 2,
            samples: vec![-1.0, 0.5, 0.0, -0.25, 1.0, 0.0],
        };
        let flac = super::encode_flac(&audio).expect("valid audio should encode as FLAC");
        assert_eq!(&flac[..4], b"fLaC");
        assert_eq!(flac[4], 0x80);
        assert_eq!(&flac[5..8], &[0, 0, 34]);
        assert_eq!(u16::from_be_bytes([flac[8], flac[9]]), 4096);
        assert_eq!(u16::from_be_bytes([flac[10], flac[11]]), 4096);
        assert_eq!(&flac[12..15], &[0, 0, 29]);
        assert_eq!(&flac[15..18], &[0, 0, 29]);
        let stream_format = u64::from_be_bytes(flac[18..26].try_into().expect("8-byte field"));
        assert_eq!(stream_format >> 44, 48_000);
        assert_eq!((stream_format >> 41) & 0x7, 1);
        assert_eq!((stream_format >> 36) & 0x1f, 23);
        assert_eq!(stream_format & 0x0f_ffff_ffff, 3);

        let frame = &flac[42..];
        assert_eq!(&frame[..2], &[0xff, 0xf8]);
        assert_eq!(frame[2] >> 4, 6);
        assert_eq!(frame[2] & 0x0f, 10);
        assert_eq!(frame[3], 0x1c);
        assert_eq!(frame[4], 0);
        assert_eq!(frame[5], 2);
        assert_eq!(frame[7], 0x02);
        assert_eq!(&frame[8..17], &[0x80, 0, 0, 0, 0, 0, 0x7f, 0xff, 0xff]);
        assert_eq!(frame[17], 0x02);
        assert_eq!(&frame[18..27], &[0x40, 0, 0, 0xe0, 0, 0, 0, 0, 0]);
        let header_crc = super::flac_crc8(&frame[..6]);
        assert_eq!(frame[6], header_crc);
        assert_eq!(
            u16::from_be_bytes([frame[27], frame[28]]),
            super::flac_crc16(&frame[..27])
        );
        assert_eq!(super::flac_crc8(b"123456789"), 0xf4);
        assert_eq!(super::flac_crc16(b"123456789"), 0xfee8);
        let rfc_frame = [
            0xff, 0xf8, 0x69, 0x18, 0x00, 0x00, 0xbf, 0x03, 0x58, 0xfd, 0x03, 0x12, 0x8b, 0xaa,
            0x9a,
        ];
        assert_eq!(super::flac_crc8(&rfc_frame[..6]), 0xbf);
        assert_eq!(super::flac_crc16(&rfc_frame[..13]), 0xaa9a);

        let long_stream = super::encode_flac(&AudioBuffer {
            sample_rate: 8_000,
            channels: 1,
            samples: vec![0.0; 4_097],
        })
        .expect("multi-frame audio should encode as FLAC");
        assert_eq!(long_stream[44] >> 4, 12);
        let second_frame_start = 42 + 6 + 1 + 4_096 * 3 + 2;
        assert_eq!(
            &long_stream[second_frame_start..second_frame_start + 2],
            &[0xff, 0xf8]
        );
        assert_eq!(long_stream[second_frame_start + 2] >> 4, 6);
        assert_eq!(long_stream[second_frame_start + 4], 1);
        assert_eq!(long_stream[second_frame_start + 5], 0);
        let uncommon_rate = super::encode_flac(&AudioBuffer {
            sample_rate: 384_000,
            channels: 1,
            samples: vec![0.0],
        })
        .expect("uncommon rates should be explicitly encoded in FLAC frames");
        let uncommon_frame = &uncommon_rate[42..];
        assert_eq!(uncommon_frame[2], 0x6e);
        assert_eq!(&uncommon_frame[6..8], &[0x96, 0]);
        let empty_stream = super::encode_flac(&AudioBuffer {
            sample_rate: 48_000,
            channels: 1,
            samples: Vec::new(),
        })
        .expect("empty audio should encode as an empty FLAC stream");
        assert_eq!(empty_stream.len(), 42);
        let empty_stream_format =
            u64::from_be_bytes(empty_stream[18..26].try_into().expect("8-byte field"));
        assert_eq!(empty_stream_format & 0x0f_ffff_ffff, 0);
        let unsupported_channels = AudioBuffer {
            sample_rate: 48_000,
            channels: 9,
            samples: vec![0.0; 9],
        };
        assert!(matches!(
            super::encode_flac(&unsupported_channels),
            Err(error) if error.code == ErrorCode::ExportFailed
        ));
    }

    #[test]
    fn rejects_invalid_audio_buffer_format_and_alignment() {
        for audio in [
            AudioBuffer {
                sample_rate: 0,
                channels: 1,
                samples: vec![0.0],
            },
            AudioBuffer {
                sample_rate: 384_001,
                channels: 1,
                samples: vec![0.0],
            },
            AudioBuffer {
                sample_rate: 48_000,
                channels: 0,
                samples: vec![],
            },
            AudioBuffer {
                sample_rate: 48_000,
                channels: 33,
                samples: vec![0.0; 33],
            },
            AudioBuffer {
                sample_rate: 48_000,
                channels: 2,
                samples: vec![0.0],
            },
        ] {
            assert!(encode_wav(&audio).is_err());
        }
    }
}
